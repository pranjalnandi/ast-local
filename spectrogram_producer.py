import torchaudio
import torch
import time
import numpy as np
import json
import base64
from kafka import KafkaProducer
import datetime


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


def serialize_tensor(tensor):
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
    }
    # data_b64 = base64.b64encode(array.tobytes()).decode("utf-8")
    # return {"shape": array.shape, "dtype": str(array.dtype), "data": data_b64}


def main():
    wav_names = [
        "./sample_audios/cough.flac",
        "./sample_audios/help.flac",
        "./sample_audios/finger_snap_clap.flac",
        "./sample_audios/clap_finger_snap.flac",
    ]
    mel_bins = 128

    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    topic = "spectrogram"

    for wav in wav_names:
        start_time = time.time()
        print(f"Processing {wav}…")

        fbank = make_features(wav, mel_bins=mel_bins)
        print("  Spectrogram shape:", fbank.shape)

        message = serialize_tensor(fbank)
        print("  Serialized done.")

        producer.send(topic, value=message).add_callback(
            lambda md: print(f"Sent to {md.topic}-{md.partition}@{md.offset}")
        ).add_errback(lambda e: print(f"Error: {e}"))
        # producer.send(topic, value=message)
        print(f"  Sent {wav} features to Kafka topic '{topic}'.")

        elapsed = time.time() - start_time
        print(f"  Time taken: {elapsed:.3f} s")
        print("=" * 40)
        time.sleep(4)

    producer.flush()
    producer.close()
    print("All messages sent. Producer closed.")


if __name__ == "__main__":
    main()
