import torchaudio
import torch
import time
import numpy as np
import json
import base64
from kafka import KafkaProducer
import datetime
from pathlib import Path


INPUT_DIR = "./sample_audios/brushing_teeth"
MEL_BINS = 128
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "spectrogram"
SLEEP_SECS = 4


def make_features(wav_name, mel_bins):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, "Input audio sampling rate must be 16kHz"
    print("waveform shape: ", waveform.shape)
    fbank = torchaudio.compliance.kaldi.fbank(
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
    print("  fbank datatype: ", fbank.dtype)
    return fbank


def serialize_tensor(tensor, wav_path):
    """
    Serializes a PyTorch tensor to a JSON-compatible format.
    """
    array = tensor.numpy()
    data_b64 = base64.b64encode(array.tobytes()).decode("utf-8")
    return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "data": data_b64,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "filename": str(wav_path.name),
    }
    # data_b64 = base64.b64encode(array.tobytes()).decode("utf-8")
    # return {"shape": array.shape, "dtype": str(array.dtype), "data": data_b64}


def main():
    wav_paths = sorted(Path(INPUT_DIR).glob("*.flac"))
    if not wav_paths:
        print(f"No .flac files found in {INPUT_DIR}")
        return

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    for wav_path in wav_paths:
        start_time = time.time()
        print(f"Processing {wav_path.name}…")

        fbank = make_features(str(wav_path), mel_bins=MEL_BINS)
        print("  Spectrogram shape:", fbank.shape)

        message = serialize_tensor(fbank, wav_path=wav_path)
        print("  Serialized done.")

        producer.send(TOPIC, value=message).add_callback(
            lambda md: print(f"Sent to {md.topic}-{md.partition}@{md.offset}")
        ).add_errback(lambda e: print(f"Error: {e}"))

        print(f"  Sent {wav_path.name} features to Kafka topic '{TOPIC}'.")

        elapsed = time.time() - start_time
        print(f"  Time taken: {elapsed:.3f} s")
        print("=" * 40)
        time.sleep(SLEEP_SECS)

    producer.flush()
    producer.close()
    print("All messages sent. Producer closed.")


if __name__ == "__main__":
    main()
