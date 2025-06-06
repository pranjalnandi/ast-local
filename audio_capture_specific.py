import wave

import numpy as np
import pyaudio
import torch
import torchaudio

FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio
DESIRED_RATE = 16000  # Desired sample rate for processing (16 kHz)
CHUNK = 1024  # Buffer size
RECORD_SECONDS = 5  # Duration of recording
OUTPUT_NATIVE = "native_output.wav"
OUTPUT_RESAMPLED = "resampled_output.wav"


def record_audio(device_index):
    audio = pyaudio.PyAudio()

    # Get device info and its native sample rate.
    device_info = audio.get_device_info_by_index(device_index)
    native_rate = int(device_info["defaultSampleRate"])
    print(f"Using device index {device_index}: {device_info['name']}")
    print(f"Device native sample rate: {native_rate}")

    # Open the stream using the native sample rate.
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=native_rate,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_index,
    )

    print("Recording...")
    frames = []
    num_reads = int(native_rate / CHUNK * RECORD_SECONDS)
    for i in range(num_reads):
        # Use exception_on_overflow=False to avoid overflow errors.
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        if i % 50 == 0:
            print(f"Read {i+1}/{num_reads}")

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Combine frames and convert to a numpy array.
    audio_data = b"".join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Debug: Print basic statistics of the recorded data.
    print("Raw audio stats:", np.min(audio_np), np.max(audio_np), np.mean(audio_np))

    # Save native audio to file to verify capture.
    with wave.open(OUTPUT_NATIVE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(native_rate)
        wf.writeframes(audio_data)
    print(f"Native audio saved as {OUTPUT_NATIVE}")

    # Convert to a PyTorch tensor and normalize to [-1.0, 1.0].
    waveform = torch.tensor(audio_np, dtype=torch.float32) / 32768.0
    # Add a channel dimension: shape becomes [1, num_samples].
    waveform = waveform.unsqueeze(0)
    print("Waveform shape before resampling:", waveform.shape)

    # Resample if native rate differs from desired rate.
    if native_rate != DESIRED_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=native_rate, new_freq=DESIRED_RATE
        )
        waveform = resampler(waveform)
    print("Waveform shape after resampling:", waveform.shape)

    # Convert back to int16 for saving.
    waveform_int16 = (
        (waveform * 32768.0).clamp(-32768, 32767).short().squeeze(0).numpy()
    )

    # Save the resampled audio as a WAV file.
    with wave.open(OUTPUT_RESAMPLED, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(DESIRED_RATE)
        wf.writeframes(waveform_int16.tobytes())
    print(f"Resampled audio saved as {OUTPUT_RESAMPLED}")


if __name__ == "__main__":
    # For example, try recording from device index 24.
    record_audio(24)
