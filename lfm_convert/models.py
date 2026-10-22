"""Explicit graph wrappers used by the pinned LFM export recipe."""
import torch


class ExplicitConv(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.in_proj = original.in_proj
        self.out_proj = original.out_proj
        self.register_buffer("taps", original.conv.weight[:, 0, :].detach().clone())
        if original.bias:
            raise ValueError("The pinned LFM checkpoint requires bias-free Conv layers")

    def forward(self, hidden, state):
        b, c, x = self.in_proj(hidden).transpose(1, 2).chunk(3, dim=1)
        bx = b * x
        length = hidden.shape[1]
        history = torch.cat((state[:, :, -2:], bx), dim=2)
        filtered = sum(history[:, :, i:i + length] * self.taps[:, i].view(1, -1, 1)
                       for i in range(3))
        new_state = torch.cat((state, bx), dim=2)[:, :, -3:]
        return self.out_proj((c * filtered).transpose(1, 2)), new_state


class ExplicitConvDecoder(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.operator_norm = original.operator_norm
        self.ffn_norm = original.ffn_norm
        self.feed_forward = original.feed_forward
        self.mixer = ExplicitConv(original.conv)

    def forward(self, hidden, state):
        mixed, new_state = self.mixer(self.operator_norm(hidden), state)
        hidden = hidden + mixed
        return hidden + self.feed_forward(self.ffn_norm(hidden)), new_state


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Prepare(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.operator_norm
        self.attn = layer.self_attn

    def forward(self, hidden, cos, sin):
        x = self.norm(hidden)
        a = self.attn
        b, t, _ = x.shape
        q = a.q_layernorm(a.q_proj(x).view(b, t, 32, 64)).transpose(1, 2)
        k = a.k_layernorm(a.k_proj(x).view(b, t, 8, 64)).transpose(1, 2)
        v = a.v_proj(x).view(b, t, 8, 64).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v


class Tile(torch.nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

    def forward(self, query, keys, values, mask):
        b, h, t, d = keys.shape
        k = keys[:, :, None].expand(b, h, 4, t, d).reshape(b, 32, t, d)
        v = values[:, :, None].expand(b, h, 4, t, d).reshape(b, 32, t, d)
        scores = torch.matmul(query, k.transpose(-1, -2)) * self.scaling + mask
        maximum = scores.amax(dim=-1, keepdim=True)
        total = torch.exp(scores - maximum).sum(dim=-1, keepdim=True)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v), maximum, total


class Finish(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.out_proj = layer.self_attn.out_proj
        self.ffn_norm = layer.ffn_norm
        self.ffn = layer.feed_forward

    def forward(self, hidden, attention_output):
        b, h, t, d = attention_output.shape
        result = attention_output.transpose(1, 2).reshape(b, t, 2048)
        hidden = hidden + self.out_proj(result)
        return hidden + self.ffn(self.ffn_norm(hidden))


class Merge(torch.nn.Module):
    def forward(self, old_output, old_maximum, old_total, tile_output, tile_maximum, tile_total):
        maximum = torch.cat((old_maximum, tile_maximum), dim=-1).amax(dim=-1, keepdim=True)
        alpha = old_total * torch.exp(old_maximum - maximum)
        beta = tile_total * torch.exp(tile_maximum - maximum)
        total = alpha + beta
        output = (alpha * old_output + beta * tile_output) / total
        return output, maximum, total


class Head(torch.nn.Module):
    def __init__(self, norm, embedding):
        super().__init__()
        self.norm = norm
        self.projection = torch.nn.Linear(2048, 128000, bias=False)
        self.projection.weight.data.copy_(embedding)

    def forward(self, hidden):
        return self.projection(self.norm(hidden))
