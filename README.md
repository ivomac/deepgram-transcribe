# Transcribe

Transcription tool using Deepgram for live microphone input or audio files, with optional AI-powered cleanup and customizable formatting.

## Installation

Install system-wide using `uv tool`:

```bash
uv tool install .
```

This makes the `transcribe` command available globally.

## Usage

### Live Microphone Transcription (Toggle Mode)

Start transcription:
```bash
transcribe
```

Stop transcription (run the same command again):
```bash
transcribe
```

The transcription will be automatically copied to your clipboard and a notification will be shown.

### AI-Powered Cleanup

Use an LLM to cleanup and format your transcription:
```bash
transcribe --llm
```

The AI will:
- Fix transcription errors and mishearings
- Add proper punctuation and capitalization
- Break into clear paragraphs
- Remove filler words (um, uh, like)
- Add a summary for long transcripts (>500 words)

Choose a different model (defaults to `deepseek/deepseek-chat`):
```bash
transcribe --llm --model anthropic/claude-3-sonnet
transcribe --llm -m openai/gpt-4o
```

### File Transcription

Transcribe an audio file:
```bash
transcribe --file path/to/audio.mp3
```

Combine with LLM cleanup:
```bash
transcribe -f audio.mp3 --llm
```

### Custom Output Formatting

For file transcriptions, customize the output format with available fields: `{text}`, `{start}`, `{end}`, `{speaker}`

Basic text only (default):
```bash
transcribe -f audio.mp3
```

With timestamps:
```bash
transcribe -f audio.mp3 --format "{start:.2f}-{end:.2f} {text}"
```

With speaker labels and timestamps:
```bash
transcribe -f audio.mp3 -t "S{speaker} [{start:.2f}-{end:.2f}] {text}"
```

Combined with LLM cleanup:
```bash
transcribe -f audio.mp3 -t "S{speaker}: {text}" --llm
```

## How It Works

- **Toggle behavior**: Running `transcribe` starts a new transcription. Running it again stops the active transcription.
- **PID tracking**: Uses `~/.cache/transcribe.pid` to track running instances
- **Signal handling**: Stops gracefully using SIGUSR1 signal
- **Notifications**: Desktop notifications via `notify-send`
- **Clipboard**: Auto-copies to clipboard using `pyperclip` (cross-platform)
- **AI Cleanup**: Uses litellm to support any LLM model for intelligent transcript formatting
- **Custom Formatting**: Parse Deepgram's structured output with timestamps and speaker diarization

## Requirements

- Python 3.13+
- `notify-send` (for notifications)
- `DEEPGRAM_API_KEY` environment variable must be set
- API key for your chosen LLM provider (e.g., `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) - required only for `--llm` flag

## Uninstall

```bash
uv tool uninstall transcribe
```
