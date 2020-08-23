import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from Utils.utils import *


class RGCNlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RGCNlayer, self).__init__()
        layer = [nn.Linear(in_dim, out_dim) for _ in range(NUM_BOND+1)]
        self.layer = nn.ModuleList(layer)
        self.dr = nn.Dropout(0.3)

    def forward(self, x):
        # x_adj = (batch_size, node_num, node_num, edge_feature_dim)
        # x_feat = (batch_size, node_num, node_feature_dim)
        x_adj, x_feat = x

        h = []
        for i in range(NUM_BOND):
            h.append(self.layer[i](x_feat))
        h = torch.stack(h, 1)
        h = torch.einsum("nijb,nbif->nbif", (x_adj, h))
        h = torch.sum(h, 1) + self.layer[NUM_BOND](x_feat)
        h = F.tanh(h)
        out = self.dr(h)

        return out


if __name__ == "__main__":
    X = torch.randn([32, 9, 5], device=device, dtype=dtype)
    A = torch.randn([32, 9, 9, 4], device=device, dtype=dtype)
    model = RGCNlayer(5, 64)
    model.to(device=device)

    y = model.forward([A, X])

    print(y.shape)

