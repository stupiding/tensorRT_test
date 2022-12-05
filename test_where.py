import time
import numbers
import numpy as np

import torch
import torch.onnx
from torch import nn

torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mask = x[:, :, :, -1].contiguous() > 0.5
        idcs = torch.nonzero(mask, as_tuple=True)

        xx = x[:, :, :, 0].clone() * 0
        xx = xx.index_put(idcs, idcs[1].float())

        return xx

#Function to Convert to ONNX
def Convert_ONNX(model, input_names, input_data):
    # Export the model
    torch.onnx.export(model,         # model being run
         tuple(input_data),       # model input (or a tuple for multiple inputs)
         "converted/test_where.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=17,    # the ONNX version to export the model to
         do_constant_folding=False,  # whether to execute constant folding for optimization
         input_names = input_names,   # the model's input names
         output_names = ['imgs'])
    print('Model has been converted to ONNX')


def main():
    model = Model().eval().cuda()

    x = torch.rand(1, 1024, 1024, 2).cuda().float()

    with torch.inference_mode():
        z = model(x)

    inp_names, inp_data = ['x'], [x]
    Convert_ONNX(model, inp_names, inp_data)

    np_data = [x.cpu().numpy()]
    np.save('test_data/test_where.npy', dict(zip(inp_names, np_data)))
    np.save('test_data/th_where.npy', dict(zip(['imgs'], [z.cpu().numpy()])))


if __name__ == '__main__':
    main()
