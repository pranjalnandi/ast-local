import os
import csv
import torch
import torchaudio
import numpy as np
import pyaudio
from collections import deque
from torch.amp import autocast
from src.models import ASTModel
from rich.console import Console
from rich.table import Table

# ----------------------------
# Configuration and Constants
# ----------------------------
SAMPLE_RATE = 16000  # Expected sample rate (16 kHz)
CHUNK_DURATION = 5  # Window duration (in seconds) for inference
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION  # Total samples per segment
OVERLAP_PERCENT = 0.20  # Overlap between segments (20%)
OVERLAP_SIZE = int(CHUNK_SIZE * OVERLAP_PERCENT)
HOP_SIZE = CHUNK_SIZE - OVERLAP_SIZE  # New samples added each time
MEL_BINS = 128  # Number of Mel filter bank bins
INPUT_TDIM = 1024  # Time dimension expected by the model

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

    # Initialize PyAudio and open streams for multiple USB microphones.
    # Specify the device indices for your microphones.
    # mic_device_indices = [1, 2]  # <-- Update with your actual device indices


    # p = pyaudio.PyAudio()
    # streams = []
    # for dev_idx in mic_device_indices:
    #     try:
    #         info = p.get_device_info_by_index(dev_idx)
    #         print(f"Using input device {dev_idx}: {info['name']}")
    #         stream = p.open(
    #             format=pyaudio.paInt16,
    #             channels=1,
    #             rate=SAMPLE_RATE,
    #             input=True,
    #             frames_per_buffer=HOP_SIZE,
    #             input_device_index=dev_idx,
    #         )
    #         streams.append(stream)
    #     except Exception as e:
    #         print(f"Error opening device {dev_idx}: {e}")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Enumerate all devices and select those that are input devices
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
    else:
        print("Selected device indices:", mic_device_indices)

    # Continue with opening streams for each detected microphone
    streams = []
    for dev_idx in mic_device_indices:
        try:
            info = p.get_device_info_by_index(dev_idx)
            print(f"Using input device {dev_idx}: {info['name']}")
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=HOP_SIZE,
                input_device_index=dev_idx,
            )
            streams.append(stream)
        except Exception as e:
            print(f"Error opening device {dev_idx}: {e}")

    if not streams:
        print("No valid audio input streams. Exiting.")
        return

    print("Listening... Press Ctrl+C to stop.")
    # Use a deque as a circular buffer to hold CHUNK_SIZE samples for the merged audio
    audio_buffer = deque(maxlen=CHUNK_SIZE)

    try:
        while True:
            new_samples_list = []
            # Read HOP_SIZE samples from each stream
            for stream in streams:
                try:
                    data = stream.read(HOP_SIZE, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.int16)
                    new_samples_list.append(samples)
                except Exception as e:
                    print("Error reading from stream:", e)
            if not new_samples_list:
                continue
            # Merge by averaging samples from all sources (ensure same length)
            merged_samples = np.mean(new_samples_list, axis=0).astype(np.int16)
            # Append merged samples to the audio buffer
            audio_buffer.extend(merged_samples)
            # When we've accumulated enough samples, process this segment
            if len(audio_buffer) >= CHUNK_SIZE:
                segment_np = np.array(audio_buffer, dtype=np.int16)
                segment_tensor = (
                    torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0) / 32768.0
                )
                predictions = predict_segment(segment_tensor, model, device, labels)
                display_predictions(predictions)
                # Remove oldest samples to maintain the desired overlap
                for _ in range(HOP_SIZE):
                    audio_buffer.popleft()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for stream in streams:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
