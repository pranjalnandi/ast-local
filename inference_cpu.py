import csv
import os
import sys
import time

import gdown
import numpy as np
import torch
import torchaudio

from src.models import ASTModel


# Function to set up the environment
def setup_environment():
    # Clone the repository if it doesn't exist
    if not os.path.exists("ast"):
        os.system("git clone https://github.com/YuanGongND/ast")
    sys.path.append("./ast")
    os.chdir("./ast")

    # Install required packages
    os.system("pip install timm==0.4.5")
    os.system("pip install wget")


# Custom ASTModel for visualization
class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list


# Function to extract features from audio
def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
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


def main():
    # Set environment variables
    os.environ["TORCH_HOME"] = "./pretrained_models"
    if not os.path.exists("./pretrained_models"):
        os.mkdir("./pretrained_models")

    # Download pretrained model if not already present
    audioset_mdl_url = (
        "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
    )
    checkpoint_path = "./pretrained_models/audio_mdl.pth"
    if not os.path.exists(checkpoint_path):
        gdown.download(audioset_mdl_url, checkpoint_path, quiet=False, fuzzy=True)

    # Load the model
    input_tdim = 1024
    ast_mdl = ASTModelVis(
        label_dim=527,
        input_tdim=input_tdim,
        imagenet_pretrain=False,
        audioset_pretrain=False,
    )
    print(f"[*INFO] Load checkpoint: {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create a new state_dict without 'module.' prefix
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load the state_dict into the model
    ast_mdl.load_state_dict(new_state_dict)

    # Move model to CPU
    audio_model = ast_mdl.to(torch.device("cpu"))
    audio_model.eval()

    # Load labels
    label_csv = "./egs/audioset/data/class_labels_indices.csv"
    labels = load_label(label_csv)

    audio_path = "./sample_audios/scream_1.flac"

    # Extract features
    feats = make_features(audio_path, mel_bins=128)
    feats_data = feats.expand(1, input_tdim, 128)
    feats_data = feats_data.to(torch.device("cpu"))

    # # Make predictions
    # with torch.no_grad():
    #     output = audio_model.forward(feats_data)
    #     output = torch.sigmoid(output)

    # result_output = output.data.cpu().numpy()[0]
    # sorted_indexes = np.argsort(result_output)[::-1]

    # Warm-up runs
    for _ in range(5):
        with torch.no_grad():
            _ = audio_model(feats_data)

    # Measure inference time
    num_iterations = 5
    total_time = 0.0

    for i in range(num_iterations):
        print(f"Loop {i} running")
        start_time = time.time()
        with torch.no_grad():
            output = audio_model(feats_data)
        end_time = time.time()
        total_time += end_time - start_time

    average_inference_time = total_time / num_iterations
    print(f"Average inference time: {average_inference_time:.6f} seconds")

    # Process the final output
    output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]

    # Print top 10 predictions
    print("Predicted results:")
    for k in range(20):
        print(
            "- {}: {:.4f}".format(
                np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]
            )
        )


if __name__ == "__main__":
    main()
