import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.layer import RGCNlayer


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(32, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 512)
        self.dr = nn.Dropout(0.1)
        self.node_head = nn.Linear(512, MAX_NODE*NUM_ATOM)
        self.edge_head = nn.Linear(512, MAX_NODE*MAX_NODE*NUM_BOND)

    def forward(self, x):
        h = F.tanh(self.l1(x))
        h = self.dr(h)
        h = F.tanh(self.l2(h))
        h = self.dr(h)
        h = F.tanh(self.l3(h))
        h = self.dr(h)

        out_node = F.softmax(self.node_head(h).view(-1, MAX_NODE, NUM_ATOM), -1)
        # out_node = self.node_head(h).view(-1, MAX_NODE, NUM_ATOM)
        out_node = self.dr(out_node)

        out_edge = F.softmax(self.edge_head(h).view(-1, MAX_NODE, MAX_NODE, NUM_BOND), -1)
        # out_edge = self.edge_head(h).view(-1, MAX_NODE, MAX_NODE, NUM_BOND)
        out_edge = (out_edge + out_edge.permute(0, 2, 1, 3))/2
        out_edge = self.dr(out_edge)

        return out_edge, out_node


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rgcn1 = RGCNlayer(NUM_ATOM, 64)
        self.rgcn2 = RGCNlayer(64, 32)
        self.l1 = nn.Linear(32, 128)
        self.l2 = nn.Linear(32, 128)
        self.head1 = nn.Linear(128, 128)
        self.head2 = nn.Linear(128, 1)

    def forward(self, x):
        in_adj, in_node_feature = x
        h = self.rgcn1([in_adj, in_node_feature])
        h = self.rgcn2([in_adj, h])

        h1 = F.sigmoid(self.l1(h))
        h2 = F.tanh(self.l2(h))
        h = F.tanh(torch.sum(torch.mul(h1, h2), 1))

        out = F.tanh(self.head1(h))
        out = F.tanh(self.head2(out))

        return out


class RewardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()

    def forward(self, x):
        x = self.discriminator(x)
        out = F.sigmoid(x)

        return out


if __name__ == "__main__":
    # Generator
    # X = torch.randn(128, 32, device=device, dtype=dtype)
    # model = Generator()
    # X, A = model.forward(X)
    # print(X.shape, A.shape)

    # Descriminator
    X = torch.randn([32, 9, 5], device=device, dtype=dtype)
    A = torch.randn([32, 9, 9, 4], device=device, dtype=dtype)
    model = Discriminator()
    model.to(device=device)

    y = model.forward(A, None, X)

    print(y[0].size(), y[1].size())
