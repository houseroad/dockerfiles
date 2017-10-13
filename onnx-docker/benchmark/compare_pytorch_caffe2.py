from argparse import ArgumentParser
import os
from timeit import Timer

import numpy as np
import onnx
import onnx_caffe2.backend
import torch
import torch.onnx
from torch.autograd import Variable
from torch.utils.model_zoo import load_url
from torch import nn
import torchvision

def create_model(model_name, batch_size):
    model = getattr(torchvision.models, model_name)(pretrained=False)
    model = model.eval()
    dummy_input = Variable(torch.rand(batch_size, 3, 224, 224))
    torch.onnx.export(model, dummy_input, "tmp.onnx", export_params=True)
    onnx_model = onnx.load("tmp.onnx")
    os.remove("tmp.onnx")
    return onnx_model


def benchmark_caffe2(model_name, batch_size, runs):
    model = create_model(model_name, batch_size)
    dummy_input = np.ndarray((batch_size, 3, 224, 224), dtype=np.float32)
    prepared_backend = onnx_caffe2.backend.prepare(model)
    def forward_model(x):
        w = {model.graph.input[0].name: x} # swap the input in the model for dummy input
        prepared_backend.run(w)[0]

    t = Timer(lambda: forward_model(dummy_input))
    avg_time = t.timeit(number=runs) / float(runs)  # temporarily disables GC
    return avg_time


def benchmark_pytorch(model_name, batch_size, runs):
    model = getattr(torchvision.models, model_name)(pretrained=False)
    model = model.eval()
    dummy_input = Variable(torch.rand(batch_size, 3, 224, 224))

    t = Timer(lambda: model(dummy_input))
    avg_time = t.timeit(number=runs) / float(runs)
    return avg_time


def run_all_benchmarks(model_name, batch_sizes, runs):
    for batch_size in batch_sizes:
        caffe2_avg_time = benchmark_caffe2(model_name, batch_size, runs)
        pytorch_avg_time = benchmark_pytorch(model_name, batch_size, runs)
        print("Batch Size: {} \t PyTorch: {:.4f} \t Caffe2: {:.4f} [{} runs]".format(batch_size,
                                                                                     pytorch_avg_time,
                                                                                     caffe2_avg_time,
                                                                                     runs))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str, help="torchvision model to run")
    parser.add_argument("--batch_sizes", nargs="+", type=int, help="batch sizes to evaluate")
    parser.add_argument("--runs", default=10, type=int, help="number of runs per batch size")

    opts = parser.parse_args()

    run_all_benchmarks(opts.model_name, opts.batch_sizes, opts.runs)
