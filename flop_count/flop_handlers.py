import torch
from collections import Counter

def elementwise_flop_jit(inputs, outputs):
    output = outputs[0]
    if output.isCompleteTensor():
        num_elements = output.type().sizes()
        flops = 1
        for dim in num_elements:
            flops *= dim
        return Counter({"elementwise": flops})
    return Counter()

def transpose_flop_jit(inputs, outputs):
    # transpose operations are generally considered to have zero FLOPs
    return Counter({"transpose": 0})

def softmax_flop_jit(inputs, outputs):
    input_tensor = inputs[0]
    if input_tensor.isCompleteTensor():
        sizes = input_tensor.type().sizes()
        N = sizes[0]
        D = sizes[1]
        flops = N * D * 3  # 2 FLOPs for exp and 1 for division
        return Counter({"softmax": flops})
    return Counter()


def gelu_flop_jit(inputs, outputs):
    input_tensor = inputs[0]
    if input_tensor.isCompleteTensor():
        num_elements = input_tensor.type().sizes()
        flops = 1
        for dim in num_elements:
            flops *= dim
        flops *= 6  # approximate FLOPs per element for GELU
        return Counter({"gelu": flops})
    return Counter()