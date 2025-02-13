import os
import torch
import torchaudio
import numpy as np
from torch.amp import autocast
from src.models import ASTModel


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
    import csv

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


# Main function
def main():
    # Set environment variables
    os.environ["TORCH_HOME"] = "./pretrained_models"
    if not os.path.exists("./pretrained_models"):
        os.mkdir("./pretrained_models")

    # Load the model
    input_tdim = 1024
    ast_mdl = ASTModel(
        label_dim=527,
        input_tdim=input_tdim,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    checkpoint_path = "./pretrained_models/audio_mdl.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()

    # Load labels
    label_csv = "./egs/audioset/data/class_labels_indices.csv"
    labels = load_label(label_csv)

    # Load the audio file
    audio_path = "./sample_audios/large_file2.flac"
    waveform, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Input audio sampling rate must be 16kHz"

    # Define windowing parameters
    window_size_sec = 5
    overlap_percent = 0.20  # 15% overlap
    window_size_samples = int(window_size_sec * sr)
    overlap_samples = int(overlap_percent * window_size_samples)
    hop_size_samples = window_size_samples - overlap_samples

    # Process the audio in windows
    num_windows = (waveform.size(1) - overlap_samples) // hop_size_samples
    print(f"Processing {num_windows} windows of audio")

    for i in range(num_windows):
        start = i * hop_size_samples
        end = start + window_size_samples
        segment = waveform[:, start:end]

        # Ensure the segment is the correct length
        if segment.size(1) < window_size_samples:
            padding = window_size_samples - segment.size(1)
            segment = torch.nn.functional.pad(segment, (0, padding))

        # Extract features
        feats = make_features(segment, sr, mel_bins=128)
        feats_data = feats.expand(1, input_tdim, 128)
        feats_data = feats_data.to(torch.device("cuda:0"))

        # Make predictions
        with torch.no_grad():
            with autocast(device_type="cuda"):
                output = audio_model.forward(feats_data)
                output = torch.sigmoid(output)

        result_output = output.data.cpu().numpy()[0]
        sorted_indexes = np.argsort(result_output)[::-1]

        # # Print top prediction for this segment
        # top_label = labels[sorted_indexes[0]]
        # confidence = result_output[sorted_indexes[0]]
        # second_label = labels[sorted_indexes[1]]
        # second_confidence = result_output[sorted_indexes[1]]
        # third_label = labels[sorted_indexes[2]]
        # third_confidence = result_output[sorted_indexes[2]]
        # print(f'Segment {i+1}/{num_windows}: {top_label} ({confidence:.4f}) - {second_label} ({second_confidence:.4f}) - {third_label} ({third_confidence:.4f})')

        top_predictions = [
            (labels[sorted_indexes[i]], result_output[sorted_indexes[i]])
            for i in range(3)
        ]
        formatted_predictions = " - ".join(
            f"{label} ({confidence:.4f})" for label, confidence in top_predictions
        )
        print(f"Segment {i+1}/{num_windows}: {formatted_predictions}")


if __name__ == "__main__":
    main()
