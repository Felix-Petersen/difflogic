import difflogic_cuda
import torch
import numpy as np


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

    def flatten(self, start_dim=0, end_dim=-1, **kwargs):
        """
        Returns the PackBitsTensor object itself.
        Arguments are ignored.
        """
        return self

    def _get_member_repr(self, member):
        if len(member) <= 4:
            result = [(np.binary_repr(integer, width=self.bit_count))[::-1] for integer in member]
            return ' '.join(result)
        first_three = [(np.binary_repr(integer, width=self.bit_count))[::-1] for integer in member[:3]]
        sep = "..."
        final = np.binary_repr(member[-1], width=self.bit_count)[::-1]
        return f"{' '.join(first_three)} {sep} {final}"
    
    def __repr__(self):
        return '\n'.join([self._get_member_repr(item) for item in self.t])