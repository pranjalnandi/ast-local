import csv
import os
from collections import deque

import numpy as np
import pyaudio
import torch
import torchaudio
from rich.console import Console
from rich.table import Table
from torch.amp import autocast

from src.models import ASTModel

# ----------------------------
# Configuration and Constants
# ----------------------------
SAMPLE_RATE = 16000  # Target sample rate for model (16 kHz)
CHUNK_DURATION = 5  # Window duration (in seconds) for inference
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION  # Total samples per segment at 16 kHz
OVERLAP_PERCENT = 0.20  # Overlap between segments (20%)
OVERLAP_SIZE = int(CHUNK_SIZE * OVERLAP_PERCENT)
HOP_SIZE = CHUNK_SIZE - OVERLAP_SIZE  # New samples added each time
MEL_BINS = 128  # Number of Mel filter bank bins
INPUT_TDIM = 1024  # Time dimension expected by the model

# READ_DURATION: time (in seconds) for each read from each microphone.
# We choose a duration such that the resampled chunk length is consistent:
READ_DURATION = 0.064  # e.g. 0.064 seconds => int(16000*0.064)=1024 samples

console = Console()


def display_predictions(predictions):
    table = Table(title="Predictions (Merged Audio)")
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


# ----------------------------
# Label Loading Function
# ----------------------------
def load_labels(label_csv):
    """
    Load class labels from a CSV file.
    Assumes the CSV header exists and labels are in the third column.
    """
    with open(label_csv, "r") as f:
        reader = csv.reader(f, delimiter=",")
        lines = list(reader)
    return [line[2] for line in lines[1:]]


# ----------------------------
# Prediction Function
# ----------------------------
def predict_segment(segment, model, device, labels):
    """
    Run inference on a given audio segment (a tensor of shape [1, N]).
    Returns the top three predictions.
    Note: Expects the segment to be normalized floats in [-1,1].
    """
    feats = make_features(segment, SAMPLE_RATE, MEL_BINS)
    feats_data = feats.unsqueeze(0).to(device)  # shape: [batch, time, freq]
    with torch.no_grad():
        with autocast(device_type="cuda"):
            output = model(feats_data)
            output = torch.sigmoid(output)
    output_np = output.cpu().numpy()[0]
    sorted_idxs = np.argsort(output_np)[::-1]
    top_predictions = [(labels[i], output_np[i]) for i in sorted_idxs[:3]]
    return top_predictions


# ----------------------------
# Main Real-Time Inference Pipeline
# ----------------------------
def main():
    # Set up model storage directory
    os.environ["TORCH_HOME"] = "./pretrained_models"
    if not os.path.exists("./pretrained_models"):
        os.mkdir("./pretrained_models")

    # Load the AST model
    ast_model = ASTModel(
        label_dim=527,
        input_tdim=INPUT_TDIM,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    checkpoint_path = "./pretrained_models/audio_mdl.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model = torch.nn.DataParallel(ast_model, device_ids=[0])
    model.load_state_dict(checkpoint)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Load class labels
    label_csv = "./egs/audioset/data/class_labels_indices.csv"
    labels = load_labels(label_csv)
    print("Labels loaded.")

    # Initialize PyAudio and enumerate available USB input devices.
    p = pyaudio.PyAudio()
    mic_device_indices = []
    print("Available input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxInputChannels"] > 0:
            print(f"Device index {i}: {device_info['name']}")
            # Filter for USB devices; adjust the keyword as needed.
            if "USB" in device_info["name"]:
                mic_device_indices.append(i)

    if not mic_device_indices:
        print("No USB input devices found. Please check your microphone connections.")
        return
    else:
        print("Selected device indices:", mic_device_indices)

    # For each selected device, open a stream at its native rate and prepare a resampler.
    # We'll store a list of tuples: (stream, resampler, read_frames)
    stream_objects = []
    for dev_idx in mic_device_indices:
        try:
            info = p.get_device_info_by_index(dev_idx)
            native_rate = int(info["defaultSampleRate"])
            # Determine number of native frames to read for READ_DURATION seconds.
            read_frames = int(native_rate * READ_DURATION)
            print(
                f"Using input device {dev_idx}: {info['name']}, native_rate: {native_rate}, read_frames: {read_frames}"
            )
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=native_rate,
                input=True,
                frames_per_buffer=read_frames,
                input_device_index=dev_idx,
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=native_rate, new_freq=SAMPLE_RATE
            )
            stream_objects.append((stream, resampler, read_frames))
        except Exception as e:
            print(f"Error opening device {dev_idx}: {e}")

    if not stream_objects:
        print("No valid audio input streams. Exiting.")
        return

    print("Listening... Press Ctrl+C to stop.")

    audio_buffer = deque(maxlen=CHUNK_SIZE)

    try:
        while True:
            resampled_chunks = []
            for stream, resampler, read_frames in stream_objects:
                try:
                    data = stream.read(read_frames, exception_on_overflow=False)
                    # Convert raw bytes to numpy array of int16
                    native_samples = np.frombuffer(data, dtype=np.int16)
                    # Convert to float tensor normalized in [-1,1]
                    native_tensor = (
                        torch.tensor(native_samples, dtype=torch.float32).unsqueeze(0)
                        / 32768.0
                    )
                    # Resample to 16kHz; expected output length = int(SAMPLE_RATE * READ_DURATION)
                    resampled_tensor = resampler(native_tensor)
                    # Remove channel dimension and convert to numpy array
                    resampled_chunk = resampled_tensor.squeeze(0).numpy()
                    resampled_chunks.append(resampled_chunk)
                except Exception as e:
                    print("Error reading from stream:", e)
            if not resampled_chunks:
                continue

            # Merge the resampled chunks by averaging (elementwise)
            # All resampled chunks should have the same length (e.g. 1024 samples)
            merged_chunk = np.mean(resampled_chunks, axis=0).astype(np.float32)
            # Append the merged chunk to the global audio buffer
            audio_buffer.extend(merged_chunk.tolist())

            if len(audio_buffer) >= CHUNK_SIZE:
                # Convert the buffer to a numpy array (of floats in [-1,1])
                segment_np = np.array(audio_buffer, dtype=np.float32)
                # Create a tensor with shape [1, num_samples] for inference
                segment_tensor = torch.tensor(
                    segment_np, dtype=torch.float32
                ).unsqueeze(0)
                predictions = predict_segment(segment_tensor, model, device, labels)
                display_predictions(predictions)

                # Remove oldest samples to maintain the desired overlap
                for _ in range(HOP_SIZE):
                    audio_buffer.popleft()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for stream, _, _ in stream_objects:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
