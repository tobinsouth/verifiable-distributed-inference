import onnx, numpy as np
import torch
import onnxruntime

session1 = onnxruntime.InferenceSession("proofs/model_part1.onnx")
session2 = onnxruntime.InferenceSession("proofs/model_part2.onnx")

input_data = torch.rand([1, 1, 28, 28]).numpy()
output1 = session1.run(None, {"input.1": input_data})[0]

intermediate_output = torch.flatten(torch.tensor(output1),1 ).numpy() # could be cleaner

output2 = session2.run(None, {"input.1": intermediate_output})[0]

