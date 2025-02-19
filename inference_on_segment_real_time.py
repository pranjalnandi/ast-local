import os
import torch
import torchaudio
import numpy as np
import time
from torch.amp import autocast
from src.models import ASTModel
import csv
import typer
import gdown

from rich.console import Console
from rich.table import Table
from rich import print

INPUT_TDIM = 1024
LABEL_DIM = 527
CHECKPOINT_PATH = "./pretrained_models/audio_mdl.pth"
MODEL_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
TOTAL_PRINTED_PREDICTION = 3

SAMPLE_RATE = 16000
CHUNK_DURATION = 5
OVERLAP_PERCENT = 0.20
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
OVERLAP_SIZE = int(CHUNK_SIZE * OVERLAP_PERCENT)
HOP_SIZE = CHUNK_SIZE - OVERLAP_SIZE  # New samples added each time
MEL_BINS = 128  # Number of Mel filter bank bins


console = Console()

def display_predictions(segment_number, predictions):
    table = Table(title=f"Segment {segment_number} Predictions")

    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Label", style="yellow")
    table.add_column("Confidence", justify="right", style="green")

    for rank, (label, confidence) in enumerate(predictions, start=1):
        table.add_row(str(rank), label, f"{confidence:.3f}")

    console.print(table)


# Function to extract features from audio
def make_features(waveform, sr, mel_bins, target_length=1024):
    assert sr == 16000, "Input audio sampling rate must be 16kHz"
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
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


# Function to load labels from a CSV file
def load_label(label_csv):
    with open(label_csv, "r") as f:
        reader = csv.reader(f, delimiter=",")
        lines = list(reader)
    labels = []
    ids = []
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


def download_model(model_url, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        gdown.download(model_url, checkpoint_path, quiet=False, fuzzy=True)


# Main function
def main(audio_file_name: str):
    if not os.path.exists("./pretrained_models"):
        os.mkdir("./pretrained_models")

    download_model(model_url=MODEL_URL, checkpoint_path=CHECKPOINT_PATH)

    # Load the model
    ast_mdl = ASTModel(
        label_dim=LABEL_DIM,
        input_tdim=INPUT_TDIM,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    print(f"[*INFO] Load checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()
    print("Model loaded! :boom:")
 
    # Load labels
    label_csv = "./egs/audioset/data/class_labels_indices.csv"
    labels = load_label(label_csv)
    print("Labels loaded! :boom:")

    # Load the audio file
    audio_path = f"./sample_audios/{audio_file_name}"

    waveform, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Input audio sampling rate must be 16kHz"

    # Process the audio in windows
    num_windows = (waveform.size(1) - OVERLAP_SIZE) // HOP_SIZE
    print(f"Processing {num_windows} windows of audio")

    for i in range(num_windows):
        start = i * HOP_SIZE
        end = start + CHUNK_SIZE
        segment = waveform[:, start:end]

        # Ensure the segment is the correct length
        if segment.size(1) < CHUNK_SIZE:
            padding = CHUNK_SIZE - segment.size(1)
            segment = torch.nn.functional.pad(segment, (0, padding))

        # Measure the start time
        start_time = time.time()

        # Extract features
        feats = make_features(segment, sr, mel_bins=MEL_BINS)
        feats_data = feats.expand(1, INPUT_TDIM, 128)
        feats_data = feats_data.to(torch.device("cuda:0"))

        with torch.no_grad():
            with autocast(device_type="cuda"):
                output = audio_model.forward(feats_data)
                output = torch.sigmoid(output)

        result_output = output.data.cpu().numpy()[0]
        sorted_indexes = np.argsort(result_output)[::-1]

        top_predictions = [
            (labels[sorted_indexes[j]], result_output[sorted_indexes[j]])
            for j in range(TOTAL_PRINTED_PREDICTION)
        ]

        display_predictions(i + 1, top_predictions)
        

        end_time = time.time()

        processing_time = end_time - start_time

        sleep_time = max(0, CHUNK_DURATION - processing_time - 1)
        time.sleep(sleep_time)


if __name__ == "__main__":
    typer.run(main)
