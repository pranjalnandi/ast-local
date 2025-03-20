import onnx
from onnx import shape_inference

# Load the ONNX model
model = onnx.load("ast_model_time_non_dynamic.onnx")

# Run shape inference on the model
inferred_model = shape_inference.infer_shapes(model)

# Iterate through all nodes and check for Reshape nodes
for node in inferred_model.graph.node:
    if node.op_type == "Reshape":
        print(f"Reshape Node: {node.name}")
        for output in node.output:
            # Look up the output's shape from value_info
            shape_found = False
            for value_info in inferred_model.graph.value_info:
                if value_info.name == output:
                    dims = []
                    for dim in value_info.type.tensor_type.shape.dim:
                        dims.append(dim.dim_value if dim.HasField("dim_value") else "?")
                    print(f"  Output {output} shape: {dims}")
                    shape_found = True
            if not shape_found:
                print(f"  Output {output} shape: not found (None?)")
