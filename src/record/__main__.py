#!/usr/bin/env python
"""Record audio from PulseAudio/PipeWire sources."""

import signal
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from os import environ

is_recording = True


def stop_recording(*_):
    """Signal handler to stop recording."""
    global is_recording
    is_recording = False


signal.signal(signal.SIGINT, stop_recording)
signal.signal(signal.SIGTERM, stop_recording)


def get_default_sources() -> tuple[str | None, str | None]:
    """Get default mic and system audio sources from PulseAudio."""
    result = subprocess.check_output(["pactl", "info"], text=True)

    default_sink = None
    default_source = None

    for line in result.split("\n"):
        if "Default Sink:" in line:
            default_sink = line.split(":", 1)[1].strip()
        elif "Default Source:" in line:
            default_source = line.split(":", 1)[1].strip()

    system_source = f"{default_sink}.monitor" if default_sink else None
    return default_source, system_source


def get_output_path() -> Path:
    """Get output file path in date-tagged folder."""
    date_folder = datetime.now().strftime("%y.%m.%d")
    output_dir = Path(environ["MEDIA"]) / "Records" / date_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    return output_dir / f"{timestamp}.wav"


def record_parec(
    source: str, output: Path, sample_rate: int = 16000
) -> tuple[subprocess.Popen, subprocess.Popen]:
    """Start parec process recording from source."""
    parec_cmd = [
        "parec",
        "--device",
        source,
        "--rate",
        str(sample_rate),
        "--channels",
        "1",
        "--format",
        "s16le",
    ]

    ffmpeg_cmd = [
        "ffmpeg",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-y",
        str(output),
    ]

    parec = subprocess.Popen(parec_cmd, stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=parec.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if parec.stdout:
        parec.stdout.close()

    return parec, ffmpeg


def wait_and_cleanup(processes: list):
    """Wait for processes and cleanup on interrupt."""
    try:
        for _, ffmpeg in processes:
            ffmpeg.wait()
    except KeyboardInterrupt:
        for parec, ffmpeg in processes:
            if parec:
                parec.terminate()
            if ffmpeg:
                ffmpeg.terminate()
        for parec, ffmpeg in processes:
            if parec:
                parec.wait()
            if ffmpeg:
                ffmpeg.wait()


def mix_audio_files(file1: Path, file2: Path, output: Path) -> bool:
    """Mix two audio files into one."""
    mix_cmd = [
        "ffmpeg",
        "-i",
        str(file1),
        "-i",
        str(file2),
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest[aout]",
        "-map",
        "[aout]",
        "-y",
        str(output),
    ]

    result = subprocess.run(
        mix_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return result.returncode == 0


def record_combined(mic: str, system: str, output: Path, sample_rate: int = 16000) -> Path | None:
    """Record both mic and system audio, then mix them."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1:
        mic_file = Path(tmp1.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
        sys_file = Path(tmp2.name)

    processes = [
        record_parec(mic, mic_file, sample_rate),
        record_parec(system, sys_file, sample_rate),
    ]

    wait_and_cleanup(processes)

    success = mix_audio_files(mic_file, sys_file, output)

    mic_file.unlink(missing_ok=True)
    sys_file.unlink(missing_ok=True)

    return output if success and output.exists() else None


def record_single(source: str, output: Path, sample_rate: int = 16000) -> Path | None:
    """Record from single source."""
    processes = [record_parec(source, output, sample_rate)]
    wait_and_cleanup(processes)
    return output if output.exists() else None


def main():
    parser = ArgumentParser(description="Record audio from PulseAudio/PipeWire sources")
    parser.add_argument(
        "-c",
        "--combined",
        action="store_true",
        help="Record both mic and system audio (mixed). Default: system audio only",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (default: $MEDIA/Records/YY.MM.DD/HHMMSS.wav)",
    )
    args = parser.parse_args()

    mic_src, sys_src = get_default_sources()
    output_file = args.output or get_output_path()

    if not sys_src:
        print("Error: No default system audio source found", file=sys.stderr)
        sys.exit(1)

    print(f"Recording to: {output_file}", file=sys.stderr)
    print("Press Ctrl+C to stop recording\n", file=sys.stderr)

    if args.combined and mic_src:
        print(f"Mic source: {mic_src}", file=sys.stderr)
        print(f"System source: {sys_src}", file=sys.stderr)
        result = record_combined(mic_src, sys_src, output_file, args.sample_rate)
    else:
        if args.combined and not mic_src:
            print("Warning: No mic source found, recording system audio only", file=sys.stderr)
        print(f"System source: {sys_src}", file=sys.stderr)
        result = record_single(sys_src, output_file, args.sample_rate)

    if result:
        print(f"\nRecording saved: {result}", file=sys.stderr)
        print(result)
    else:
        print("\nError: Recording failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
