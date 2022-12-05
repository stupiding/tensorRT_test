
import os
import sys
import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit


class TensorRTInfer:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        # trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(i)

            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_tensor_shape(i)
            
            if is_input:
                self.batch_size = shape[0]

            size = dtype.itemsize
            for s in shape:
                size *= s
            
            allocation = cuda.mem_alloc(size)
            self.allocations.append(allocation)

            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            # print("{} '{}' with shape {} and dtype {}".format(
            #     "Input" if is_input else "Output",
            #     binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def infer(self, batch_data):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        for k,v in batch_data.items():
            for i in range(len(self.inputs)):
                if self.inputs[i]['name'] == k:
                    cuda.memcpy_htod(self.inputs[i]['allocation'], v)


        start_time = time.time()
        self.context.execute_v2(self.allocations)
        time_perf = time.time() - start_time
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs], time_perf

    def process(self, batch_data):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Run inference

        times = []
        outputs = self.infer(batch_data)
        for i in range(5):
            outputs, time_perf = self.infer(batch_data)
            times.append(time_perf)
        times = [f'{t:.4f}' for t in times]
        print(f'trt times: {times}')
        # print('trt infer time: ', (time.time() - start_time) / 10)
        trt_outputs = dict(zip([o['name'] for o in self.outputs], outputs))
        np.save('test_data/trt_output.npy', trt_outputs)

        org_outputs = np.load('test_data/th_where.npy', allow_pickle=True).item(0)
        for k in org_outputs:
            diff = org_outputs[k] - trt_outputs[k]
            # print('inputs: ', batch_data)
            # print('torch:  ', org_outputs[k])
            # print('tensor: ', trt_outputs[k])
            print(f'{k} max diff: {abs(diff).max()}, {trt_outputs[k].shape}')
            print(f'{k} diff count: {(abs(diff) > 0.001).sum() / (abs(org_outputs[k]) > 0.1).sum()} '
                    f'[{(abs(diff) > 0.001).sum()} / {(abs(org_outputs[k]) > 0.001).sum()}]')
            print(f'{k} diff ratio: {abs(diff).sum() / abs(org_outputs[k]).sum()} '
                    f'[{abs(diff).sum()} / {abs(org_outputs[k]).sum()}')  

def main(args):

    trt_infer = TensorRTInfer(args.engine)
    data = np.load('test_data/test_where.npy', allow_pickle=True).item(0)

    outputs = trt_infer.process(data)

    print()
    print("Finished Processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None, required=True,
                        help="The serialized TensorRT engine")
    args = parser.parse_args()
    main(args)
