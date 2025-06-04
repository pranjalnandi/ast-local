import csv
import os
import zlib
from pathlib import Path

import gdown
import numpy as np
import torch
import torchaudio
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from torch.amp import autocast

from src.models import ASTModel

INPUT_TDIM = 1024
LABEL_DIM = 527
CHECKPOINT_PATH = "./pretrained_models/audio_mdl.pth"
# CHECKPOINT_PATH = "./src/ast_model.onnx"
MODEL_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
TOTAL_PRINTED_PREDICTION = 10
MEL_BINS = 128  # Number of Mel filter bank bins

console = Console()


def display_predictions(predictions):
    table = Table(title="Predictions (Single Audio)")

    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Label", style="yellow")
    table.add_column("Confidence", justify="right", style="green")

    for rank, (label, confidence) in enumerate(predictions, start=1):
        table.add_row(str(rank), label, f"{confidence:.3f}")

    console.print(table)


# Function to extract features from audio
def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, "Input audio sampling rate must be 16kHz"
    print("waveform shape: ", torch._shape_as_tensor(waveform))
    fbank_32 = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
        frame_length=25,
    )
    # convert to float16
    fbank_fp16 = fbank_32.to(torch.float16)

    size_kb = (fbank_fp16.numel() * fbank_fp16.element_size()) / 1024
    print(f"Feature size: {fbank_fp16.shape}, Size in KB: {size_kb:.2f} KB")

    # Convert to bytes and compress using zlib
    arr_bytes = fbank_fp16.cpu().numpy().tobytes()
    compressed = zlib.compress(arr_bytes, level=6)

    size_bytes = len(compressed)
    print(f"Compressed size in KB: {size_bytes / 1024:.2f} KB")

    # Decompress the data and convert back to float16
    decompressed = zlib.decompress(compressed)
    fbank_recon = np.frombuffer(decompressed, dtype=np.float16).reshape(
        fbank_fp16.shape
    )

    fbank = torch.tensor(fbank_recon, dtype=torch.float16)

    n_frames = fbank.shape[0]
    p = target_length - n_frames

    print("Number of positive values: ", torch.count_nonzero(fbank > 0))
    # print("fbank shape: ", torch._shape_as_tensor(fbank))
    # print("fbank: ", fbank[20:30, 20:30])
    # print(
    #     "Indices of positive values: ", torch.nonzero(fbank > 0, as_tuple=False)[0:10]
    # )
    # print("number of frames: ", n_frames)
    # print("padding: ", p)

    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    # print(
    #     "Number of positive values after padding and normalization: ",
    #     torch.count_nonzero(fbank > 0),
    # )
    # print("fbank element: ", fbank[500:502, :])
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


def main(audio_dir_name: str):
    if not os.path.exists("./pretrained_models"):
        os.mkdir("./pretrained_models")

    download_model(model_url=MODEL_URL, checkpoint_path=CHECKPOINT_PATH)

    # Load models
    ast_mdl = ASTModel(
        label_dim=LABEL_DIM,
        input_tdim=INPUT_TDIM,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    # print(ast_mdl)
    print(f"[*INFO] Load checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()

    # Load labels
    label_csv = "./egs/audioset/data/class_labels_indices.csv"
    labels = load_label(label_csv)

    audio_dir_path = f"./sample_audios/{audio_dir_name}"

    audio_paths = sorted(Path(audio_dir_path).glob("*.flac"))
    if not audio_paths:
        print(f"No .flac files found in {audio_dir_path}")
        return

    for audio_path in audio_paths:
        print(f"Processing audio file: {audio_path.name}")
        # Extract features
        feats = make_features(audio_path, mel_bins=MEL_BINS)  # feats-shape: (1024, 128)
        feats_data = feats.expand(
            1, INPUT_TDIM, 128
        )  # feats_data-shape: (1, 1024, 128)
        feats_data = feats_data.to(torch.device("cuda:0"))

        # Make predictions
        with torch.no_grad():
            with autocast(device_type="cuda"):
                output = audio_model.forward(feats_data)
                output = torch.sigmoid(output)
                # print("Sum of all elements in output: ", torch.sum(output))

        result_output = output.data.cpu().numpy()[0]
        sorted_indexes = np.argsort(result_output)[::-1]

        top_predictions = [
            (labels[sorted_indexes[j]], result_output[sorted_indexes[j]])
            for j in range(TOTAL_PRINTED_PREDICTION)
        ]

        display_predictions(top_predictions)


if __name__ == "__main__":
    typer.run(main)
