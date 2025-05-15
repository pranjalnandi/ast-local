import os
import csv
import torch
import torchaudio
import numpy as np
import pyaudio
import gdown
from collections import deque
from rich.console import Console
from rich.table import Table
from rich import print

# ----------------------------
# Configuration and Constants
# ----------------------------
INPUT_TDIM = 1024
LABEL_DIM = 527
CHECKPOINT_PATH = "./pretrained_models/audio_mdl.pth"
MODEL_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
TOTAL_PRINTED_PREDICTIONS = 3

SAMPLE_RATE = 16000
CHUNK_DURATION = 5
OVERLAP_PERCENT = 0.20
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
OVERLAP_SIZE = int(CHUNK_SIZE * OVERLAP_PERCENT)
HOP_SIZE = CHUNK_SIZE - OVERLAP_SIZE  # New samples added each time
MEL_BINS = 128  # Number of Mel filter bank bins

console = Console()


def display_predictions(predictions):
    table = Table(title=f"Predictions")

    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Label", style="yellow")
    table.add_column("Confidence", justify="right", style="green")

    for rank, (label, confidence) in enumerate(predictions, start=1):
        table.add_row(str(rank), label, f"{confidence:.3f}")

    console.print(table)


# ----------------------------
# Feature Extraction Function
# ----------------------------
def make_features(waveform, sr, mel_bins, target_length=INPUT_TDIM):
    """
    Compute filter bank features using Kaldi-compatible settings.
    Pads or crops the feature matrix to a fixed target length.
    """
    assert sr == SAMPLE_RATE, "Input audio sampling rate must be 16kHz"

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = pad(fbank)
    elif p < 0:
        fbank = fbank[:target_length, :]

    # Normalize using constants from training
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def get_spectrogram(segment):
    feats = make_features(segment, SAMPLE_RATE, MEL_BINS)
    return feats


def download_model(model_url, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        gdown.download(model_url, checkpoint_path, quiet=False, fuzzy=True)


def main():
    p = pyaudio.PyAudio()
    try:
        default_input_info = p.get_default_input_device_info()
        input_device_index = default_input_info["index"]
        print("Using input device:", default_input_info["name"])
    except Exception as e:
        print("Error getting default input device:", e)
        return

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=HOP_SIZE,
        input_device_index=input_device_index,
    )

    print("Listening... Press Ctrl+C to stop.")

    audio_buffer = deque(maxlen=CHUNK_SIZE)

    try:
        while True:
            data = stream.read(HOP_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)
            audio_buffer.extend(samples)

            # When we have accumulated enough samples, process this segment
            if len(audio_buffer) >= CHUNK_SIZE:
                segment_np = np.array(audio_buffer, dtype=np.int16)
                # Normalize to [-1, 1] (16-bit PCM normalization)
                segment_tensor = (
                    torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0) / 32768.0
                )

                spectrogram = get_spectrogram(segment_tensor)
                # display_predictions(predictions)
                print("Spectrogram shape: ", spectrogram.shape)

                # Remove the oldest samples to maintain an overlap (keep OVERLAP_SIZE samples)
                for _ in range(HOP_SIZE):
                    audio_buffer.popleft()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
