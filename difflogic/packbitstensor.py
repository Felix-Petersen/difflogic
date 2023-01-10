import difflogic_cuda
import torch


class PackBitsTensor:
    def __init__(self, t: torch.BoolTensor, bit_count=32, device='cuda'):

        assert len(t.shape) == 2, t.shape

        self.bit_count = bit_count
        self.device = device

        if device == 'cuda':
            t = t.to(device).T.contiguous()
            self.t, self.pad_len = difflogic_cuda.tensor_packbits_cuda(t, self.bit_count)
        else:
            raise NotImplementedError(device)

    def group_sum(self, k):
        assert self.device == 'cuda', self.device
        return difflogic_cuda.groupbitsum(self.t, self.pad_len, k)
