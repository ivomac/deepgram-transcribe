#!/usr/bin/env python

import signal
import subprocess
from datetime import datetime
from pathlib import Path
from os import environ
import tempfile

recording_data = []
is_recording = False


def audio_callback(indata, *_):
    """Callback for audio recording."""
    recording_data.append(indata.copy())


def stop_recording(*_):
    """Signal handler to stop recording."""
    global is_recording
    is_recording = False


signal.signal(signal.SIGINT, stop_recording)
signal.signal(signal.SIGTERM, stop_recording)


def get_default_sources():
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


def record_source(
    source: str,
    output_file: Path,
    sample_rate: int = 16000,
    channels: int = 1,
):
    try:
        parec_cmd = [
            "parec",
            "--device",
            source,
            "--rate",
            str(sample_rate),
            "--channels",
            str(channels),
            "--format",
            "s16le",
        ]

        parec_proc = subprocess.Popen(parec_cmd, stdout=subprocess.PIPE)

        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-y",
            str(output_file),
        ]

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=parec_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if parec_proc.stdout:
            parec_proc.stdout.close()

        ffmpeg_proc.wait()

    except KeyboardInterrupt:
        if parec_proc:
            parec_proc.terminate()
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
        if parec_proc:
            parec_proc.wait()
        if ffmpeg_proc:
            ffmpeg_proc.wait()

    if output_file.exists():
        return output_file
    return None


def record_combined(
    mic_source: str,
    system_source: str,
    output_file: Path,
    sample_rate: int = 16000,
):
    """Record both microphone and system audio, then mix them."""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as sys_tmp:
        sys_file = Path(sys_tmp.name)

    mic_cmd = [
        "parec",
        "--device",
        mic_source,
        "--rate",
        str(sample_rate),
        "--channels",
        "1",
        "--format",
        "s16le",
    ]

    sys_cmd = [
        "parec",
        "--device",
        system_source,
        "--rate",
        str(sample_rate),
        "--channels",
        "1",
        "--format",
        "s16le",
    ]

    mic_ffmpeg_cmd = [
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
        str(mic_file),
    ]

    sys_ffmpeg_cmd = [
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
        str(sys_file),
    ]

    mic_parec = None
    sys_parec = None
    mic_ffmpeg = None
    sys_ffmpeg = None

    try:
        mic_parec = subprocess.Popen(mic_cmd, stdout=subprocess.PIPE)
        sys_parec = subprocess.Popen(sys_cmd, stdout=subprocess.PIPE)

        mic_ffmpeg = subprocess.Popen(
            mic_ffmpeg_cmd,
            stdin=mic_parec.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sys_ffmpeg = subprocess.Popen(
            sys_ffmpeg_cmd,
            stdin=sys_parec.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if mic_parec.stdout:
            mic_parec.stdout.close()
        if sys_parec.stdout:
            sys_parec.stdout.close()

        mic_ffmpeg.wait()
        sys_ffmpeg.wait()

    except KeyboardInterrupt:
        for proc in [mic_parec, sys_parec, mic_ffmpeg, sys_ffmpeg]:
            if proc:
                proc.terminate()
        for proc in [mic_parec, sys_parec, mic_ffmpeg, sys_ffmpeg]:
            if proc:
                proc.wait()

    mix_cmd = [
        "ffmpeg",
        "-i",
        str(mic_file),
        "-i",
        str(sys_file),
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest[aout]",
        "-map",
        "[aout]",
        "-y",
        str(output_file),
    ]

    result = subprocess.run(mix_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    mic_file.unlink(missing_ok=True)
    sys_file.unlink(missing_ok=True)

    if result.returncode == 0 and output_file.exists():
        return output_file
    return None


def main():
    date_folder = datetime.now().strftime("%y.%m.%d")
    output_dir = Path(environ["MEDIA"]) / "Records" / date_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    output_file = output_dir / f"{timestamp}.wav"

    result_file = None

    mic_src, sys_src = get_default_sources()

    if mic_src and sys_src:
        result_file = record_combined(
            mic_src,
            sys_src,
            output_file,
        )
    if sys_src:

    print(f"\n{result_file}")


if __name__ == "__main__":
    main()
