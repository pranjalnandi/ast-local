import base64
import csv
import json
import os
import sqlite3

import gdown
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.amp import autocast

from kafka import KafkaConsumer
from src.models import ASTModel

# ── Configuration ─────────────────────────────────────────────────────────────
BOOTSTRAP = "172.16.2.13:9092"
TOPIC = "spectrogram"
GROUP_ID = "spectrogram_consumers"

# Audio/Model constants
MEL_BINS = 128
INPUT_TDIM = 1024
LABEL_DIM = 527
CHECKPOINT_DIR = "./pretrained_models"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/audio_mdl.pth"
MODEL_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
LABEL_CSV = "./egs/audioset/data/class_labels_indices.csv"
TOP_K = 5

DB_PATH = "/home/pranjal/audio-app/predictions.db"
TABLE = "preds"
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db_cursor = db_conn.cursor()

console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────
def download_model():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        console.log(f"Downloading AST checkpoint…")
        gdown.download(MODEL_URL, CHECKPOINT_PATH, fuzzy=True)


def load_labels():
    with open(LABEL_CSV, "r") as f:
        rows = list(csv.reader(f))
    # skip header, col2 = index, col3 = human label
    return [r[2] for r in rows[1:]]


def display_predictions(predictions):
    table = Table(show_header=True, header_style="bold blue")

    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Label", style="yellow")
    table.add_column("Confidence", justify="right", style="green")

    for rank, (label, confidence) in enumerate(predictions, start=1):
        table.add_row(str(rank), label, f"{confidence:.3f}")

    console.print(table)


def build_model(device):
    download_model()
    # instantiate
    mdl = ASTModel(
        label_dim=LABEL_DIM,
        input_tdim=INPUT_TDIM,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    # wrap + load
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    mdl = torch.nn.DataParallel(mdl).to(device)
    mdl.load_state_dict(checkpoint)
    mdl.eval()
    return mdl


def preprocess_fbank(fbank: torch.Tensor):
    """
    pad/truncate to INPUT_TDIM x MEL_BINS and normalize.
    Expects fbank shape [T, 128].
    Returns Tensor [1, INPUT_TDIM, 128] on same device.
    """
    T = fbank.size(0)
    delta = INPUT_TDIM - T
    if delta > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, delta))
    elif delta < 0:
        fbank = fbank[:INPUT_TDIM]
    # normalize with same stats as pretrain
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank.unsqueeze(0)  # [1, D, H]


# ── Consumer setup ────────────────────────────────────────────────────────────
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP,
    group_id=GROUP_ID,
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
)

# Load once
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
audio_model = build_model(device)
labels = load_labels()
console.log(f"Model loaded on {device}, ready to consume.")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    console.log("Starting Kafka consumer…")
    try:
        for msg in consumer:
            raw = msg.value
            if raw is None:
                continue

            # deserialize tensor
            try:
                data_bytes = base64.b64decode(raw["data"])
                arr = np.frombuffer(data_bytes, dtype=raw["dtype"]).reshape(
                    raw["shape"]
                )
                fbank = torch.tensor(arr, device=device)

                timestamp = raw.get("timestamp", None)
                wav_path = raw.get("filename", None)
            except Exception as e:
                console.log(
                    f"Failed to deserialize message at offset {msg.offset}: {e}"
                )
                continue

            # preprocess
            feats = preprocess_fbank(fbank)

            # predict
            with torch.no_grad(), autocast(device_type=device.type):
                logits = audio_model(feats)
                probs = torch.sigmoid(logits)[0]

            # top-K
            topk_idx = torch.topk(probs, TOP_K).indices.cpu().numpy()
            topk = [(labels[i], float(probs[i])) for i in topk_idx]

            l1, l2, l3 = topk[0][0], topk[1][0], topk[2][0]
            c1, c2, c3 = topk[0][1], topk[1][1], topk[2][1]
            # save to db if only l1,l2 or l3 contains "door", "Toilet Flush" or "Electric Toothbrush"
            if (
                "Door" in l1
                or "Door" in l2
                or "Door" in l3
                or "Toilet flush" in l1
                or "Toilet flush" in l2
                or "Toilet flush" in l3
                or "Electric toothbrush" in l1
                or "Electric toothbrush" in l2
                or "Electric toothbrush" in l3
            ):
                console.log("Saving to DB...")
                db_cursor.execute(
                    f"INSERT INTO {TABLE} (timestamp, label1, label2, label3, conf1, conf2, conf3) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, l1, l2, l3, c1, c2, c3),
                )
                db_conn.commit()

            console.print(
                f"\n📨 Msg offset={msg.offset} (File name={wav_path}) (at={timestamp}) → Top {TOP_K} predictions:"
            )
            display_predictions(topk)

            consumer.commit()
    except KeyboardInterrupt:
        console.log("Consumer interrupted. Exiting...")
    finally:
        consumer.close()
        db_conn.close()
        console.log("Consumer closed.")
        console.log("Bye! 👋")


if __name__ == "__main__":
    main()
