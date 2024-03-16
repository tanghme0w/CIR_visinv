import torch
from visinv import MultiHeadCrossAttention


mh_attn_layer = MultiHeadCrossAttention(src_dim=768, tgt_dim=1024, num_heads=8)
visual_hidden = torch.rand(size=[64, 257, 1024])
text_embedding = torch.rand(size=[64, 768])
output_hidden = mh_attn_layer(text_embedding, visual_hidden)
print(output_hidden.shape)
