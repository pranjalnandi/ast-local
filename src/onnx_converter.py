import torch


def convert_model_to_onnx(model, onnx_file_path, input_tdim=1024, frequency_bins=128):
    """
    Convert a PyTorch model (e.g., your ASTModel) to the ONNX format.

    Parameters:
      model (torch.nn.Module): The PyTorch model to export.
      onnx_file_path (str): File path to save the ONNX model.
      input_tdim (int): Number of time frames (default: 1024).
      frequency_bins (int): Number of frequency bins (default: 128).
    """
    model.eval()
    # Create a dummy input tensor with shape [batch_size, time, frequency]
    dummy_input = torch.randn(1, input_tdim, frequency_bins)

    # Export the model to ONNX
    torch.onnx.export(
        model,  # The model being exported
        dummy_input,  # A sample input tensor
        onnx_file_path,  # Where to save the ONNX model
        export_params=True,  # Store the trained parameters within the model file
        opset_version=11,  # ONNX version to export to (adjust if needed)
        do_constant_folding=True,  # Enable constant folding for optimization
        input_names=["input"],  # Name of the model input
        output_names=["output"],  # Name of the model output
        dynamic_axes={
            "input": {0: "batch_size", 1: "time_dim"},
            "output": {0: "batch_size"},
        },  # Allow dynamic batch and time dimensions
    )
    print(f"Model has been successfully exported to {onnx_file_path}")


# Example usage:
if __name__ == "__main__":
    from models import ASTModel  # Ensure this imports your model definition

    # Create an instance of your model with desired parameters.
    model = ASTModel(input_tdim=1024)
    # Convert the model to ONNX and save it.
    convert_model_to_onnx(model, "ast_model.onnx", input_tdim=1024, frequency_bins=128)
