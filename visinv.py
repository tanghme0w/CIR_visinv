from torch import Tensor
from torch.nn import Module, Linear, Dropout, ReLU, Sequential


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
