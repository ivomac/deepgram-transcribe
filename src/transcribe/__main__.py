#!/usr/bin/env python

import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from os import environ
from pathlib import Path
import httpx
import asyncio
import re

from deepgram import (
    DeepgramClient,
    FileSource,
    LiveOptions,
    LiveTranscriptionEvents,
    Microphone,
    PrerecordedOptions,
)

PID_FILE = Path(environ["HOME"]) / ".cache" / "transcribe.pid"

# Constants for configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SLEEP_BASE = 1.0
POLLING_INTERVAL = 0.1
FINAL_WAIT = 0.5

# LLM configuration constants
DEFAULT_LLM_TIMEOUT = 60.0
DEFAULT_LLM_RETRIES = 2
DEFAULT_MAX_PARALLEL = 3

# Deepgram API constants
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_CHANNELS = 1
DEEPGRAM_SAMPLE_RATE = 16000
DEEPGRAM_UTTERANCE_END_MS = "1000"
DEEPGRAM_ENDPOINTING = 300
DEEPGRAM_API_VERSION = "1"


def chunk_transcript(transcript: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    Split transcript into chunks by sentences/paragraphs.
    Tries to keep sentences and paragraphs together.
    """
    if len(transcript) <= max_chunk_size:
        return [transcript]

    chunks = []
    current_chunk = ""

    paragraphs = transcript.split("\n\n")

    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        else:
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def notify(title: str, message: str):
    """Send desktop notification."""
    try:
        subprocess.run(
            [
                "notify-send",
                f"--app-name={title}",
                "--icon=/usr/share/icons/Papirus/128x128/apps/gtranscribe.svg",
                title,
                message,
            ],
            check=False,
        )
    except FileNotFoundError:
        pass


def copy_to_clipboard(text: str):
    """Copy text to clipboard using pyperclip."""
    try:
        import pyperclip

        pyperclip.copy(text)
    except Exception as e:
        print(f"Warning: Failed to copy to clipboard: {e}", file=sys.stderr)


async def cleanup_chunk(
    chunk: str,
    model: str,
    timeout: float,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    chunk_index: int,
    total_chunks: int,
    formatting_str: str,
) -> str:
    """
    Clean up a single chunk of transcript using litellm.
    """
    try:
        import litellm

        system_prompt = f"""
You are a transcript formatter. You will be given an AI transcript of speech.

The transcript uses the following format per sentence:

{formatting_str}

PRESERVE:
- Speaker IDs and timing markers if included in the original sentence formatting
- Original message and intent

FORMAT:
- Add proper punctuation and capitalization
- Combine adjacent sentences from the same speaker into paragraphs

REMOVE:
- Pure filler words ("um", "uh", "like, you know" if meaningless)
- False starts where speaker immediately restarts
- Meaningless repetitions

FIX:
- Obvious transcription errors or mishearings
- Grammar if it doesn't change meaning

DO NOT:
- Add any intro, outro, or explanation
- Use markdown formatting (**, ##, etc.)
- Editorialize or interpret
"""

        message = (
            f"Please format this transcript chunk ({chunk_index + 1}/{total_chunks}):\n\n{chunk}"
        )
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    response = await litellm.acompletion(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": message,
                            },
                        ],
                        timeout=timeout,
                    )

                return str(response.choices[0].message.content)

            except (httpx.ReadTimeout, httpx.TimeoutException, litellm.exceptions.Timeout):
                if attempt < max_retries:
                    print(
                        f"Timeout on chunk {chunk_index + 1}, attempt {attempt + 1}, retrying...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(DEFAULT_SLEEP_BASE * (attempt + 1))
                else:
                    print(
                        f"Failed chunk {chunk_index + 1} after {max_retries + 1} attempts",
                        file=sys.stderr,
                    )
            except Exception as e:
                if attempt < max_retries and isinstance(e, (httpx.ConnectError, httpx.ReadError)):
                    print(
                        f"Failure on chunk {chunk_index + 1}, attempt {attempt + 1}, retrying...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(DEFAULT_SLEEP_BASE * (attempt + 1))
                else:
                    print(f"AI cleanup failed for chunk {chunk_index + 1}: {e}", file=sys.stderr)
                    break

        print(
            f"Using original content for chunk {chunk_index + 1} after all retries failed",
            file=sys.stderr,
        )
        return chunk

    except ImportError:
        print("Warning: litellm not installed, skipping AI cleanup", file=sys.stderr)
        return chunk
    except Exception as e:
        print(f"Warning: AI cleanup failed for chunk {chunk_index + 1}: {e}", file=sys.stderr)
        return chunk


async def cleanup_transcript_async(
    transcript: str,
    model: str,
    timeout: float,
    max_retries: int,
    max_parallel: int,
    formatting_str: str,
) -> str:
    """
    Use deepseek-chat via litellm to cleanup and format transcript in parallel chunks.
    """
    if not transcript.strip():
        return transcript

    chunks = chunk_transcript(transcript)
    total_chunks = len(chunks)

    if total_chunks == 1:
        return await cleanup_chunk(
            chunks[0], model, timeout, max_retries, asyncio.Semaphore(1), 0, 1, formatting_str
        )

    print(
        f"Processing transcript in {total_chunks} chunks with max {max_parallel} parallel calls...",
        file=sys.stderr,
    )

    semaphore = asyncio.Semaphore(max_parallel)

    tasks = []
    for i, chunk in enumerate(chunks):
        task = cleanup_chunk(
            chunk, model, timeout, max_retries, semaphore, i, total_chunks, formatting_str
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    cleaned_chunks = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Chunk {i + 1} failed with error: {result}", file=sys.stderr)
            cleaned_chunks.append(chunks[i])
        else:
            cleaned_chunks.append(result)

    return "\n\n".join(cleaned_chunks)


def cleanup_transcript(
    transcript: str,
    model: str,
    timeout: float,
    max_retries: int,
    max_parallel: int,
    formatting_str: str,
) -> str:
    """
    Synchronous wrapper for async cleanup_transcript_async.
    """
    try:
        return asyncio.run(
            cleanup_transcript_async(
                transcript, model, timeout, max_retries, max_parallel, formatting_str
            )
        )
    except Exception as e:
        print(f"Error in async cleanup: {e}", file=sys.stderr)
        return transcript


def summarize_transcript(
    transcript: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> str:
    """
    Generate a summary of the transcript using litellm.
    """
    try:
        import litellm

        system_prompt = """You are a transcript summarizer. You will be given a transcript of speech.

Create a concise summary that captures:
1. Main topics discussed
2. Key decisions or conclusions
3. Important points or arguments
4. Open questions and required actions

Do not include any concluding or introductory text like "Here is a summary:",
just provide the summary directly."""

        message = f"Please summarize this transcript:\n\n{transcript}"

        for attempt in range(max_retries + 1):
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                    timeout=timeout,
                )

                return str(response.choices[0].message.content)

            except (httpx.ReadTimeout, httpx.TimeoutException, litellm.exceptions.Timeout):
                if attempt < max_retries:
                    print(
                        f"Timeout on summary, attempt {attempt + 1}, retrying...",
                        file=sys.stderr,
                    )
                    time.sleep(DEFAULT_SLEEP_BASE * (attempt + 1))
                else:
                    print(
                        f"Failed to generate summary after {max_retries + 1} attempts",
                        file=sys.stderr,
                    )
            except Exception as e:
                if attempt < max_retries and isinstance(e, (httpx.ConnectError, httpx.ReadError)):
                    print(
                        f"Failure on summary, attempt {attempt + 1}, retrying...",
                        file=sys.stderr,
                    )
                    time.sleep(DEFAULT_SLEEP_BASE * (attempt + 1))
                else:
                    print(f"AI summary failed: {e}", file=sys.stderr)
                    break

        print(
            "Using empty summary after all retries failed",
            file=sys.stderr,
        )
        return ""

    except ImportError:
        print("Warning: litellm not installed, skipping summary", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Warning: AI summary failed: {e}", file=sys.stderr)
        return ""


def get_running_pid() -> int | None:
    """Get PID of running transcription process if it exists."""
    if not PID_FILE.exists():
        return None

    try:
        pid = int(PID_FILE.read_text().strip())
        subprocess.run(["kill", "-0", str(pid)], check=True, capture_output=True)
        return pid
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError):
        PID_FILE.unlink(missing_ok=True)
        return None


def write_pid():
    """Write current process PID to file."""
    import os

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def cleanup_pid():
    """Remove PID file."""
    PID_FILE.unlink(missing_ok=True)


def signal_handler(*_):
    global RUNNING_STATE
    RUNNING_STATE = False


RUNNING_STATE = True

signal.signal(signal.SIGUSR1, signal_handler)


class Transcriber:
    client: DeepgramClient = DeepgramClient(environ["DEEPGRAM_API_KEY"])

    def from_mic(self) -> str:
        options = LiveOptions(
            model=DEEPGRAM_MODEL,
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=DEEPGRAM_CHANNELS,
            sample_rate=DEEPGRAM_SAMPLE_RATE,
            interim_results=True,
            utterance_end_ms=DEEPGRAM_UTTERANCE_END_MS,
            vad_events=True,
            endpointing=DEEPGRAM_ENDPOINTING,
        )
        addons = {"no_delay": "true"}

        transcript = []
        finals = []

        def on_message(_, result, **kwargs):
            if kwargs:
                pass
            nonlocal finals, transcript
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            if result.is_final:
                finals.append(sentence)
                if result.speech_final:
                    transcript.append(" ".join(finals))
                    finals.clear()
            return

        connection = self.client.listen.websocket.v(DEEPGRAM_API_VERSION)
        connection.on(LiveTranscriptionEvents.Transcript, on_message)
        connection.start(options, addons=addons)

        microphone = Microphone(connection.send)
        microphone.start()

        while RUNNING_STATE:
            time.sleep(POLLING_INTERVAL)
        time.sleep(FINAL_WAIT)

        microphone.finish()
        connection.finish()

        return " ".join(transcript)

    def from_file(self, audio_file: str, fmt: str) -> str:
        with open(audio_file, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options: PrerecordedOptions = PrerecordedOptions(
            model=DEEPGRAM_MODEL,
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
        )

        data = self.client.listen.rest.v(DEEPGRAM_API_VERSION).transcribe_file(payload, options)

        sentences = []

        for utt in data.results.utterances:
            fields = {
                "text": utt.transcript.strip(),
                "start": utt.start,
                "end": utt.end,
                "speaker": utt.speaker,
            }

            sentences.append(fmt.format(**fields))

        return "\n".join(sentences)


def main():
    """Start transcribe command."""
    parser = ArgumentParser(
        description="Transcribe audio using Deepgram and process output with litellm calls"
    )
    parser.add_argument("--file", "-f", default=None, help="Path to audio file")
    parser.add_argument(
        "--format",
        "-F",
        default="{text}",
        help="""Only works with --file. Examples:
    "{text}"
    "{start:.2f}-{end:.2f} {text}"
    "S{speaker} [{start:.2f}-{end:.2f}]: {text}"
        """,
    )
    parser.add_argument(
        "--llm",
        "-l",
        action="store_true",
        default=False,
        help="Use llm for cleanup and formatting",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="deepseek/deepseek-chat",
        help="Choose litellm model",
    )
    parser.add_argument(
        "--keep",
        "-k",
        action="store_true",
        default=False,
        help="Keep original transcript when using --llm",
    )
    parser.add_argument(
        "--timeout",
        "-T",
        type=float,
        default=DEFAULT_LLM_TIMEOUT,
        help=f"Timeout in seconds for LLM requests (default: {DEFAULT_LLM_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=DEFAULT_LLM_RETRIES,
        help=f"Number of retries for LLM requests (default: {DEFAULT_LLM_RETRIES})",
    )
    parser.add_argument(
        "--copy",
        "-c",
        action="store_true",
        default=False,
        help="Copy transcript to clipboard",
    )
    parser.add_argument(
        "--max-parallel",
        "-p",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=f"Maximum parallel LLM calls for cleanup (default: {DEFAULT_MAX_PARALLEL})",
    )
    parser.add_argument(
        "--summarize",
        "-s",
        action="store_true",
        default=False,
        help="Generate a summary of the transcript and prepend it",
    )
    args = parser.parse_args()

    running_pid = get_running_pid()

    try:
        if running_pid:
            subprocess.run(["kill", "-USR1", str(running_pid)], check=True)
            return

        write_pid()

        transcriber = Transcriber()
        notify("Transcribe", "Transcribing...")
        if args.file:
            transcript = transcriber.from_file(args.file, args.format)
        else:
            transcript = transcriber.from_mic()

        if transcript.strip():
            if args.llm:
                notify("Transcribe", "Cleaning up transcript...")
                cleaned_transcript = cleanup_transcript(
                    transcript,
                    model=args.model,
                    timeout=args.timeout,
                    formatting_str=args.format,
                    max_retries=args.retries,
                    max_parallel=args.max_parallel,
                )
                if args.keep:
                    transcript = (
                        f"{cleaned_transcript}\n\n=== ORIGINAL TRANSCRIPT ===\n\n{transcript}"
                    )
                else:
                    transcript = cleaned_transcript

            if args.summarize:
                notify("Transcribe", "Generating summary...")
                summary = summarize_transcript(
                    transcript,
                    model=args.model,
                    timeout=args.timeout,
                    max_retries=args.retries,
                )
                if summary:
                    transcript = (
                        f"=== SUMMARY ===\n\n{summary}\n\n=== TRANSCRIPT ===\n\n{transcript}"
                    )
                else:
                    print("Warning: Failed to generate summary", file=sys.stderr)

            print(transcript)
            if args.copy:
                copy_to_clipboard(transcript)
                notify("Transcribe", "Transcription copied to clipboard")
            else:
                notify("Transcribe", "Transcription complete")
        else:
            notify("Transcribe", "No transcription recorded")

    finally:
        cleanup_pid()


if __name__ == "__main__":
    main()
