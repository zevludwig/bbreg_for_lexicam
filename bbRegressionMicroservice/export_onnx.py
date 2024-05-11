
import torch
from detection_model import initialize_detection_model
model, t = initialize_detection_model()


# allow different image shapes
dynamic_axes= {'input':{1:'width', 2:'height'}, 'output':{1:'width', 2:'height'}} 

torch.onnx.export(model, ([t[0]], None), 'out_dyn.onnx',  export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)



