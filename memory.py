import torch
from torch import nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, M, N, controller_out):
        super(Memory, self).__init__()

        self.N = N
        self.M = M
        self.read_lengths = self.N + 1 + 1 + 3 + 1
        self.write_lengths = self.N + 1 + 1 + 3 + 1 + self.N + self.N
        self.w_last = []
        self.reset_memory()

    def get_weights(self):
        return self.w_last

    def reset_memory(self):
        self.w_last = []
        self.w_last.append(torch.zeros([1, self.M], dtype=torch.float32))

    def address(self, key, stren, gate, shift, sharp, memory, w_last):
        wc = self._similarity(key, stren, memory)

        wg = self._interpolate(wc, gate, w_last)
        w_t = self._shift(wg, shift)
        w = self._sharpen(w_t, sharp)

        return w

    def _similarity(self, key, stren, memory):

        w = F.cosine_similarity(memory, key, -1, 1e-16)
        w = F.softmax(stren * w, dim=-1)
        return w

    def _interpolate(self, wc, gate, w_last):
        return gate * wc + (1 - gate) * w_last

    def _shift(self, wg, shift):
        result = torch.zeros(wg.size())
        result = _convolve(wg, shift)
        return result

    def _sharpen(self, w_t, sharp):
        w = w_t ** sharp
        w = torch.div(w, torch.sum(w, dim=-1) + 1e-16)
        return w


class ReadHead(Memory):

    def __init__(self, M, N, controller_out):
        super(ReadHead, self).__init__(M, N, controller_out)


        self.fc_read = nn.Linear(controller_out, self.read_lengths)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def read(self, memory, w):
        return torch.matmul(w, memory)

    def forward(self, x, memory):
        param = self.fc_read(x)
        key, stren, gate, shift, sharp = torch.split(param, [self.N, 1, 1, 3, 1], dim=1)

        key = torch.tanh(key)
        stren = F.softplus(stren)
        gate = torch.sigmoid(gate)
        shift = F.softmax(shift, dim=1)
        sharp = 1 + F.softplus(sharp)

        w = self.address(key, stren, gate, shift, sharp, memory, self.w_last[-1])
        self.w_last.append(w)
        mem = self.read(memory, w)
        return mem, w


class WriteHead(Memory):

    def __init__(self, M, N, controller_out):
        super(WriteHead, self).__init__(M, N, controller_out)


        self.fc_write = nn.Linear(controller_out, self.write_lengths)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def write(self, memory, w, e, a):
        """write to memory (according to section 3.2)."""
        w = torch.squeeze(w)
        e = torch.squeeze(e)
        a = torch.squeeze(a)

        erase = torch.ger(w, e)
        add = torch.ger(w, a)

        m_tilde = memory * (1 - erase)
        memory_update = m_tilde + add

        return memory_update

    def forward(self, x, memory):
        param = self.fc_write(x)

        key, stren, gate, shift, sharp, a, e = torch.split(param, [self.N, 1, 1, 3, 1, self.N, self.N], dim=1)

        key = torch.tanh(key)
        stren = F.softplus(stren)
        gate = torch.sigmoid(gate)
        shift = F.softmax(shift, dim=-1)
        sharp = 1 + F.softplus(sharp)
        a = torch.tanh(a)
        e = torch.sigmoid(e)

        w = self.address(key, stren, gate, shift, sharp, memory, self.w_last[-1])
        self.w_last.append(w)
        mem = self.write(memory, w, e, a)
        return mem, w


def _convolve(w, shift):
    b, d = shift.shape
    assert b == 1
    assert d == 3
    w = torch.squeeze(w)
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), shift.view(1, 1, -1)).view(b, -1)
    return c
