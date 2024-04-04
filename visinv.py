from torch import Tensor, bmm
from torch.nn import Module, Linear, Dropout, ReLU, Sequential, Softmax


class VisualInversion(Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = [Linear(dim, middle_dim), Dropout(dropout), ReLU()]
            dim = middle_dim
            layers.append(Sequential(*block))        
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class MultiHeadCrossAttention(Module):
    def __init__(self, src_dim, tgt_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = tgt_dim // num_heads
        assert self.head_dim * self.num_heads == tgt_dim    # tgt_sim must be divisible by num_heads
        self.q_proj = Linear(src_dim, tgt_dim)
        self.k_proj = Linear(tgt_dim, tgt_dim)
        self.v_proj = Linear(tgt_dim, tgt_dim)
        self.attention_dropout = Dropout(0.1)
        self.softmax = Softmax(dim=-1)
        self.out_proj = Linear(tgt_dim, src_dim)

    def _shard(self, tensor, bs):
        return tensor.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, query, target):
        # using the output embedding of text encoder as query, shape is (batch_size, src_dim)
        # target shape is (batch_size, seq_len, tgt_dim)
        batch_size, seq_len, tgt_dim = query.shape

        # Project and shard the queries, keys, and values
        q = self._shard(self.q_proj(query), batch_size) # q: [bs, num_heads, seq_len, head_dim]
        k = self._shard(self.k_proj(target), batch_size)    # k: [bs, 1, num_heads, head_dim]
        v = self._shard(self.v_proj(target), batch_size)    # v: [bs, 1, num_heads, head_dim]

        # Calculate the attention scores
        q = q.view(batch_size, self.num_heads * seq_len, self.head_dim)   # [bs, num_heads * seq_len, head_dim]
        k = k.view(batch_size, self.num_heads, self.head_dim).transpose(1, 2)    # [bs, head_dim, num_heads]
        scores = bmm(q, k) / (self.head_dim ** 0.5)      # [bs, num_heads * seq_len, num_heads]
        scores = self.softmax(scores)
        scores = self.attention_dropout(scores)

        # Apply the attention to the values
        v = v.view(batch_size, self.num_heads, self.head_dim)  # [bs, num_heads, head_dim]
        context = bmm(scores, v)    # [bs, num_heads * seq_len, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [bs, seq_len, tgt_dim] todo is this correct?

        # Project the context to the original query shape
        attention_output = self.out_proj(context)
        return attention_output     # [bs, seq_len, tgt_dim]


class CrossAttention(Module):
    def __init__(self, src_dim, tgt_dim):
        # src_dim=1024, tgt_dim=768
        super().__init__()
        self.q_proj = Linear(src_dim, tgt_dim)
        self.k_proj = Linear(tgt_dim, tgt_dim)
        self.v_proj = Linear(tgt_dim, tgt_dim)
        self.softmax = Softmax(dim = -1)
        self.final_proj = Linear(tgt_dim, src_dim)
    
    def forward(self, query, target):
        q = self.q_proj(query)                  # [bs, 257, 768]
        k = self.k_proj(target).transpose(1, 2)  # [bs, 768, 77]
        v = self.v_proj(target)                  # [bs, 77, 768]
        scores = bmm(q, k)                      # [bs, 257, 77]
        scores = self.softmax(scores)           # [bs, 257, 77]
        context = bmm(scores, v)                # [bs, 257, 768]
        attn_out = self.final_proj(context)     # [bs, 257, 1024]
        return attn_out
