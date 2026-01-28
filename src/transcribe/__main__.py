#!/usr/bin/env python

import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from os import environ
from pathlib import Path
import httpx

from deepgram import (
    DeepgramClient,
    FileSource,
    LiveOptions,
    LiveTranscriptionEvents,
    Microphone,
    PrerecordedOptions,
)

PID_FILE = Path(environ["HOME"]) / ".cache" / "transcribe.pid"


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


def cleanup_transcript(
    transcript: str, model: str, timeout: float = 30.0, max_retries: int = 2
) -> str:
    """
    Use deepseek-chat via litellm to cleanup and format transcript.
    Includes summary if transcript is long (>500 words).
    """
    try:
        import litellm

        system_prompt = """You are a transcript formatter. Your task is to:
1. Fix any obvious transcription errors or mishearings
2. Add proper punctuation and capitalization
3. Break into clear paragraphs where appropriate
4. Remove filler words (um, uh, like) unless they're essential to meaning
5. Maintain the speaker's voice and intent"""

        word_count = len(transcript.split())

        if word_count > 500:
            system_prompt += "\n6. Add a summary at the beginning"

        for attempt in range(max_retries + 1):
            try:
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Please format this transcript:\n\n{transcript}",
                        },
                    ],
                    timeout=timeout,
                )

                return str(response.choices[0].message.content)

            except (httpx.ReadTimeout, httpx.TimeoutException, litellm.exceptions.Timeout):
                if attempt < max_retries:
                    print(f"Timeout on attempt {attempt + 1}, retrying...", file=sys.stderr)
                    time.sleep(1 * (attempt + 1))
                else:
                    print(
                        f"Failed after {max_retries + 1} attempts due to timeout", file=sys.stderr
                    )
            except Exception as e:
                if attempt < max_retries and isinstance(e, (httpx.ConnectError, httpx.ReadError)):
                    print(f"Network error on attempt {attempt + 1}, retrying...", file=sys.stderr)
                    time.sleep(1 * (attempt + 1))
                else:
                    print(f"AI cleanup failed: {e}", file=sys.stderr)
                    break

        notify("Transcription", "AI cleanup failed, using original transcript")
        return transcript

    except ImportError:
        print("Warning: litellm not installed, skipping AI cleanup", file=sys.stderr)
        return transcript
    except Exception as e:
        print(f"Warning: AI cleanup failed: {e}", file=sys.stderr)
        return transcript


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
            model="nova-3",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing=300,
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

        connection = self.client.listen.websocket.v("1")
        connection.on(LiveTranscriptionEvents.Transcript, on_message)
        connection.start(options, addons=addons)

        microphone = Microphone(connection.send)
        microphone.start()

        while RUNNING_STATE:
            time.sleep(0.1)
        time.sleep(0.5)

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
            model="nova-3",
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
        )

        data = self.client.listen.rest.v("1").transcribe_file(payload, options)

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
    parser = ArgumentParser(description="Transcribe audio using Deepgram")
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
        default=60.0,
        help="Timeout in seconds for LLM requests (default: 60)",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=2,
        help="Number of retries for LLM requests (default: 2)",
    )
    parser.add_argument(
        "--copy",
        "-c",
        action="store_true",
        default=False,
        help="Copy transcript to clipboard",
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
                    transcript, model=args.model, timeout=args.timeout, max_retries=args.retries
                )
                if args.keep:
                    transcript = f"{cleaned_transcript}\n\n--- Original below ---\n\n{transcript}"
                else:
                    transcript = cleaned_transcript

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
