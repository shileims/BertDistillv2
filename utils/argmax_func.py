import torch

class ArgMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        idx = torch.argmax(inputs, 1)
        output = torch.zeros_like(inputs)
        output.scatter_(1, idx, 1)
        return output
                                                
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
