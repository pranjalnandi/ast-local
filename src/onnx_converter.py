import torch
from models import ASTModel
from torch.amp import autocast
import onnx
import onnxruntime
import numpy as np
from collections import OrderedDict
import time

LABEL_DIM = 527
INPUT_TDIM = 1024

model_path = "../pretrained_models/audio_mdl.pth"


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def convert_model_to_onnx(model, onnx_file_path, input_tdim=1024, frequency_bins=128):
    """
    Parameters:
      model (torch.nn.Module): The PyTorch model to export.
      onnx_file_path (str): File path to save the ONNX model.
      input_tdim (int): Number of time frames (default: 1024).
      frequency_bins (int): Number of frequency bins (default: 128).
    """
    model.eval()
    dummy_input = torch.randn(1, input_tdim, frequency_bins)

    with torch.no_grad():
        with autocast(device_type="cuda"):
            start = time.time()
            torch_out = model.forward(dummy_input)
            end = time.time()
            print(f"Inference of Pytorch model used {end - start} seconds")
            # torch_out = torch.sigmoid(output)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes={
        #     "input": {0: "batch_size", 1: "time_dim"},
        #     "output": {0: "batch_size"},
        # },  # Allow dynamic batch and time dimensions
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Model has been successfully exported to {onnx_file_path}")

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(
        onnx_file_path, providers=["CPUExecutionProvider"]
    )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    start = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end = time.time()
    print(f"Inference of ONNX model used {end - start} seconds")

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    model = ASTModel(
        label_dim=LABEL_DIM,
        input_tdim=INPUT_TDIM,
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict)

    convert_model_to_onnx(
        model, "ast_model_29_april.onnx", input_tdim=1024, frequency_bins=128
    )
