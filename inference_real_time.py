import os
import csv
import torch
import torchaudio
import numpy as np
import pyaudio
from collections import deque
from torch.amp import autocast
from src.models import ASTModel

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
    Returns a formatted string of the top three predictions.
    """
    # Extract features (fbank) from the waveform segment
    feats = make_features(segment, SAMPLE_RATE, MEL_BINS)
    # The model expects input shape [batch, time, frequency]
    feats_data = feats.unsqueeze(0).to(device)
    # Run inference using automatic mixed precision if available
    with torch.no_grad():
        with autocast(device_type="cuda"):
            output = model(feats_data)
            output = torch.sigmoid(output)
    output_np = output.cpu().numpy()[0]
    sorted_idxs = np.argsort(output_np)[::-1]
    top_predictions = [(labels[i], output_np[i]) for i in sorted_idxs[:3]]
    formatted_predictions = " - ".join(
        f"{label} ({score:.4f})" for label, score in top_predictions
    )
    return formatted_predictions


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

    # Initialize PyAudio and select a valid input device (USB microphone)
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
    # Use a deque as a circular buffer to hold CHUNK_SIZE samples
    audio_buffer = deque(maxlen=CHUNK_SIZE)

    try:
        while True:
            # Read a chunk of data (HOP_SIZE samples) from the microphone
            data = stream.read(HOP_SIZE, exception_on_overflow=False)
            # Convert raw bytes to a NumPy array of int16 samples
            samples = np.frombuffer(data, dtype=np.int16)
            # Append the new samples to the audio buffer
            audio_buffer.extend(samples)
            # When we have accumulated enough samples, process this segment
            if len(audio_buffer) >= CHUNK_SIZE:
                # Convert the deque buffer to a NumPy array and then a tensor
                segment_np = np.array(audio_buffer, dtype=np.int16)
                # Normalize to [-1, 1] (16-bit PCM normalization)
                segment_tensor = (
                    torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0) / 32768.0
                )
                # Get predictions for the segment
                predictions = predict_segment(segment_tensor, model, device, labels)
                print("Predictions:", predictions)
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
