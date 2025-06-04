import os
import csv
import torch
import torchaudio
import numpy as np
import pyaudio
import gdown
from collections import deque
from torch.amp import autocast
from src.models import ASTModel
from rich.console import Console
from rich.table import Table
from rich import print
import matplotlib.pyplot as plt

# ----------------------------
# Configuration and Constants
# ----------------------------
INPUT_TDIM = 1024
LABEL_DIM = 527
CHECKPOINT_PATH = "./pretrained_models/audio_mdl.pth"
MODEL_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
TOTAL_PRINTED_PREDICTIONS = 3

SAMPLE_RATE = 16000

# We still want 5 sec chunks with 20% overlap:
CHUNK_DURATION = 5  # seconds
OVERLAP_PERCENT = 0.20  # 20%
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION  # 80_000 samples
OVERLAP_SIZE = int(CHUNK_SIZE * OVERLAP_PERCENT)  # 16_000 samples
HOP_SIZE = CHUNK_SIZE - OVERLAP_SIZE  # 64_000 samples

# But for the denoiser, we run in small hops (128 samples ≈ 8 ms):
N_FFT = 512  # 32 ms window
HOP_LENGTH = 128  # 8 ms hop → ~8 ms algorithmic latency
WIN_LENGTH = N_FFT

MEL_BINS = 128  # for your mel-spectrogram images

SPECTROGRAM_FOLDER = "./spectrograms"
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

console = Console()


# ----------------------------
# ─── LOW-LATENCY DENOISER ────────────────────────────────────────────────────
# ----------------------------
# (This code is adapted from the previous streaming example.)


def np_to_torch(x_np: np.ndarray) -> torch.Tensor:
    """
    Convert a 1D float32 NumPy array in [-1,+1] to a torch Tensor of shape (1, n).
    """
    return torch.from_numpy(x_np).to(device).unsqueeze(0)


def torch_to_np(x_t: torch.Tensor) -> np.ndarray:
    """
    Convert a torch Tensor of shape (1, n) with float32 in [-1,+1] back to a 1D float32 NumPy array.
    """
    return x_t.squeeze(0).cpu().numpy()


def estimate_noise_spectrum(noise_waveform: torch.Tensor) -> torch.Tensor:
    """
    Given noise_waveform of shape (1, num_samples), float32 in [-1,+1],
    compute its average magnitude spectrum via STFT.
    Returns a Tensor of shape (freq_bins, 1), clamped ≥1e-8.
    """
    stft = torch.stft(
        noise_waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH).to(device),
        return_complex=True,
    )  # shape = (1, freq_bins, time_frames)

    stft = stft.squeeze(0)  # → (freq_bins, time_frames)
    mag = stft.abs()  # → (freq_bins, time_frames)
    noise_mag = mag.mean(dim=1, keepdim=True)  # → (freq_bins,1)
    return noise_mag.clamp(min=1e-8)


class StreamingDenoiser:
    def __init__(self, noise_mag: torch.Tensor):
        """
        noise_mag: Tensor of shape (freq_bins, 1) computed via estimate_noise_spectrum.
        """
        self.noise_mag = noise_mag.to(device)  # (freq_bins,1)
        self.window = torch.hann_window(WIN_LENGTH).to(device)

        # Input buffer: keep exactly N_FFT samples (starts as zeros).
        self.in_buffer = torch.zeros((1, N_FFT), device=device)

        # Overlap buffer: (N_FFT - HOP_LENGTH) zeros → we will overlap-add from ISTFT.
        self.out_overlap = torch.zeros((1, N_FFT - HOP_LENGTH), device=device)

    def process_frame(self, new_frame_np: np.ndarray) -> np.ndarray:
        """
        new_frame_np: NumPy float32 array of length HOP_LENGTH, in [-1,+1].
        Returns: denoised NumPy float32 array of length HOP_LENGTH, in [-1,+1].
        """
        # 1) Convert to torch → shape (1, HOP_LENGTH)
        new_frame = np_to_torch(new_frame_np)  # (1, 128)

        # 2) Slide input buffer: drop oldest HOP_LENGTH, append new_frame
        self.in_buffer = torch.cat(
            (self.in_buffer[:, HOP_LENGTH:], new_frame), dim=1
        )  # → (1, 512)

        # 3) STFT on the full buffer
        stft_full = torch.stft(
            self.in_buffer,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            return_complex=True,
        )  # shape = (1, freq_bins, 1)

        stft_frame = stft_full.squeeze(0).squeeze(1)  # → (freq_bins,)

        # 4) Magnitude/phase
        mag = stft_frame.abs().unsqueeze(1)  # (freq_bins,1)
        phase = torch.angle(stft_frame).unsqueeze(1)  # (freq_bins,1)

        # 5) Spectral subtraction: clamp at zero
        mag_denoised = (mag - self.noise_mag).clamp(min=0.0)

        # 6) Reconstruct complex spectrum
        real = mag_denoised * torch.cos(phase)
        imag = mag_denoised * torch.sin(phase)
        denoised_stft = torch.complex(real, imag).unsqueeze(1)  # (freq_bins,1)

        # 7) ISTFT → yields exactly N_FFT time-domain samples
        recon_wave = torch.istft(
            denoised_stft,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            length=N_FFT,
        ).unsqueeze(
            0
        )  # → (1, 512)

        # 8) Overlap-add
        full_frame = recon_wave + torch.cat(
            (self.out_overlap, torch.zeros((1, HOP_LENGTH), device=device)), dim=1
        )  # → (1, 512)

        to_emit = full_frame[:, :HOP_LENGTH]  # first 128 samples
        self.out_overlap = full_frame[:, HOP_LENGTH:]  # last 384 samples

        # 9) Return as NumPy float32 in [-1,+1]
        return torch_to_np(to_emit.clamp(-1.0, +1.0))  # shape=(128,)


# ----------------------------
# Feature Extraction Function
# ----------------------------
def make_features(waveform, sr, mel_bins, target_length=INPUT_TDIM):
    """
    Compute filter bank (fbank) features using Kaldi‐compatible settings.
    Pads/crops to a fixed time dimension.
    """
    assert sr == SAMPLE_RATE, "Input audio sampling rate must be 16 kHz"

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_bins,
        dither=0.0,
        frame_shift=10,
    )  # shape = (time_frames, mel_bins)

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
    Load class labels from a CSV. Assumes header and labels in third column.
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
    Run inference on a given audio segment (Tensor shape [1, N], float32 in [-1,+1]).
    Returns a list of (label, confidence) for top‐3.
    """
    feats = make_features(segment, SAMPLE_RATE, MEL_BINS)
    feats_data = feats.unsqueeze(0).to(device)  # → (1, time, freq)

    with torch.no_grad():
        with autocast(device_type="cuda"):
            output = model(feats_data)  # shape (1, LABEL_DIM)
            output = torch.sigmoid(output)

    output_np = output.cpu().numpy()[0]  # (LABEL_DIM,)
    sorted_idxs = np.argsort(output_np)[::-1]

    top_predictions = [
        (labels[i], float(output_np[i]))
        for i in sorted_idxs[:TOTAL_PRINTED_PREDICTIONS]
    ]
    return top_predictions


def display_predictions(predictions, chunk_counter):
    table = Table(title=f"Predictions – Chunk {chunk_counter}")
    table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
    table.add_column("Label", style="yellow")
    table.add_column("Confidence", justify="right", style="green")

    for rank, (label, confidence) in enumerate(predictions, start=1):
        table.add_row(str(rank), label, f"{confidence:.3f}")
    console.print(table)


# ----------------------------
# Mel Spectrogram Saving (for visualization)
# ----------------------------
def save_mel_spectrogram(waveform_np, sample_rate, folder, idx):
    """
    waveform_np: int16 NumPy array of shape (CHUNK_SIZE,)
    """
    # Convert to float in [-1,+1]
    waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0) / 32768.0

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=MEL_BINS,
        dither=0.0,
        frame_shift=10,
    )  # (time_frames, mel_bins)

    mel_spec = fbank.numpy().T  # → (mel_bins, time_frames)
    plt.figure(figsize=(6, 4))
    plt.imshow(mel_spec, origin="lower", aspect="auto")
    plt.axis("off")

    out_path = os.path.join(folder, f"mel_chunk_{idx:05d}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def download_model(model_url, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        gdown.download(model_url, checkpoint_path, quiet=False, fuzzy=True)


# ----------------------------
# Main Real-Time Inference + Denoising Pipeline
# ----------------------------
if __name__ == "__main__":
    #################################################
    # 1) Prepare pretrained_models folder + download checkpoint
    #################################################
    if not os.path.exists("./pretrained_models"):
        os.mkdir("./pretrained_models")
    download_model(model_url=MODEL_URL, checkpoint_path=CHECKPOINT_PATH)

    #################################################
    # 2) Load the AST model to GPU
    #################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ast_model = ASTModel(
        label_dim=LABEL_DIM,
        input_tdim=INPUT_TDIM,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    print(f"[*INFO] Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = torch.nn.DataParallel(ast_model, device_ids=[0])
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("[*INFO] Model loaded!")

    #################################################
    # 3) Load class labels
    #################################################
    label_csv = "./egs/audioset/data/class_labels_indices.csv"
    labels = load_labels(label_csv)
    print("[*INFO] Labels loaded!")

    #################################################
    # 4) Set up PyAudio for noise estimation (1 second)
    #################################################
    p = pyaudio.PyAudio()
    try:
        default_input_info = p.get_default_input_device_info()
        input_device_index = default_input_info["index"]
        print("[*INFO] Using input device:", default_input_info["name"])
    except Exception as e:
        print("[ERROR] Could not get default input device:", e)
        exit(1)

    # 4.1) Open a short stream to capture ~1 second of noise
    print("→ Please remain silent (or play only background noise) for 1 second…")
    noise_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=HOP_LENGTH,
        input_device_index=input_device_index,
    )

    noise_chunks = []
    num_noise_chunks = SAMPLE_RATE // HOP_LENGTH  # =16000/128 = 125 → ~1 sec
    for _ in range(num_noise_chunks):
        data = noise_stream.read(HOP_LENGTH, exception_on_overflow=False)
        np_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        noise_chunks.append(np_frame)
    noise_stream.stop_stream()
    noise_stream.close()

    noise_wave_np = np.concatenate(noise_chunks, axis=0)  # (≈16000,)
    noise_wave_t = np_to_torch(noise_wave_np)  # (1,16000)
    noise_mag = estimate_noise_spectrum(noise_wave_t)  # (freq_bins,1)
    print("[*INFO] Noise spectrum estimated.")

    #################################################
    # 5) Create the StreamingDenoiser instance
    #################################################
    denoiser = StreamingDenoiser(noise_mag)

    #################################################
    # 6) Open the “real-time” input stream (hop of 128 samples)
    #################################################
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=HOP_LENGTH,
        input_device_index=input_device_index,
    )
    print("[*INFO] Now listening and denoising in real time… Press Ctrl+C to stop.")

    #################################################
    # 7) Main loop: read 128 samples → denoise → push into deque
    #################################################
    audio_buffer = deque(maxlen=CHUNK_SIZE)  # holds the last 80 000 denoised samples
    chunk_counter = 0

    try:
        while True:
            # 7.1) Read one small hop (128 samples)
            data = stream.read(HOP_LENGTH, exception_on_overflow=False)
            in_frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # 7.2) Denoise that 128-sample hop
            denoised_hop = denoiser.process_frame(
                in_frame
            )  # float32 in [-1,+1], shape=(128,)

            # 7.3) Convert denoised float32 back to int16 (for consistent storage)
            denoised_int16 = (denoised_hop * 32767.0).astype(np.int16)

            # 7.4) Append each int16 sample into our deque
            #      (deque drops oldest if over maxlen)
            for s in denoised_int16:
                audio_buffer.append(int(s))

            # 7.5) Once we have CHUNK_SIZE samples stored (5 sec), run inference
            if len(audio_buffer) >= CHUNK_SIZE:
                # 7.5.1) Make a NumPy int16 array of length CHUNK_SIZE
                segment_np = np.array(audio_buffer, dtype=np.int16)

                # 7.5.2) Save the mel spectrogram image for visualization
                save_mel_spectrogram(
                    segment_np, SAMPLE_RATE, SPECTROGRAM_FOLDER, chunk_counter
                )

                # 7.5.3) Normalize to float32 in [-1,+1] for model input
                segment_tensor = (
                    torch.tensor(segment_np, dtype=torch.float32).unsqueeze(0) / 32768.0
                ).to(device)

                # 7.5.4) Run ASTModel inference
                predictions = predict_segment(segment_tensor, model, device, labels)
                display_predictions(predictions, chunk_counter)
                chunk_counter += 1

                # 7.5.5) Pop the oldest HOP_SIZE samples to maintain 20% overlap
                for _ in range(HOP_SIZE):
                    audio_buffer.popleft()

    except KeyboardInterrupt:
        print("\n[*INFO] Stopping...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
