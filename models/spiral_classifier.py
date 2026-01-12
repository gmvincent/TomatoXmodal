import torch
import torch.nn.functional as F

from models.spiral_conv import SpiralConv

def scatter_add(src, index, dim=0, out=None, dim_size=None):
    # Handle dim_size to allocate output
    if out is None:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        
        size = list(src.shape)
        size[dim] = dim_size
        
        out = torch.zeros(size, dtype=src.dtype, device=src.device)

    # index_add_ requires index to match src.shape[dim]
    return out.index_add_(dim, index, src)

def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out

class SpiralBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SpiralConv(in_channels, out_channels)

    def forward(self, x, spiral_indices, down_transform):
        """
        spiral_indices: (N, seq_len)
        down_transform: sparse matrix (coo)
        """
        out = F.elu(self.conv(x, spiral_indices))    # ← pass indices here
        out = Pool(out, down_transform)
        return out
    
    
    
class SpiralClassifier(torch.nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()

        self.blocks = torch.nn.ModuleList()
        for i in range(len(channels)):
            cin = in_channels if i == 0 else channels[i - 1]
            cout = channels[i]
            self.blocks.append(SpiralBlock(cin, cout))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(channels[-1], channels[-1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x, spiral_indices, down_transform):
        for i, block in enumerate(self.blocks):
            s = spiral_indices[i]
            d = down_transform[i]

            assert x.size(1) == s.size(0), \
                f"Resolution mismatch: x has {x.size(1)} vertices, spirals[{i}] has {s.size(0)}"

            x = block(x, s, d)

        x = x.mean(dim=1)
        return self.classifier(x)

