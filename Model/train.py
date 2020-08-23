import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import Generator, Discriminator, RewardNet
from Model.dataset import MolDataset


def gradient_penalty(y, x):
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.contiguous().view(dydx.size()[0], -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def wgan_loss(D, G, A, X, alpha=10, dim_latent=DIM_Z):
    pred_real = D.forward([A, X])
    L_real = torch.mean(pred_real)

    z = torch.randn([X.size(0), dim_latent], device=device, requires_grad=True)
    pred_fake = G(z)
    L_fake = torch.mean(D.forward(pred_fake))

    eps_A = torch.rand([A.size()[0], 1, 1, 1], device=device)
    eps_X = torch.rand([X.size()[0], 1, 1], device=device)
    A_hat = (eps_A*A + (1-eps_A)*pred_fake[0]).requires_grad_(True)
    X_hat = (eps_X*X + (1-eps_X)*pred_fake[1]).requires_grad_(True)

    pred_hat = D([A_hat, X_hat])

    L_grad = gradient_penalty(pred_hat, A_hat) + gradient_penalty(pred_hat, X_hat)

    return L_fake - L_real + alpha*L_grad


if __name__ == "__main__":
    # Training Loop
    num_epoch = 100
    batch_size = 64
    lr_G = 1e-4
    lr_D = 1e-4

    smiles_list = read_smilesset("data/zinc_under15.smi")
    mol_dataset = MolDataset(smiles_list)
    mol_dataloader = torch.utils.data.DataLoader(mol_dataset, batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    R = RewardNet().to(device)

    optimizaer_G = torch.optim.Adam(list(G.parameters())+list(R.parameters()), lr_G)
    optimizaer_D = torch.optim.Adam(D.parameters(), lr_D)

    for n in range(num_epoch):

        G_losses = []
        D_losses = []

        for i, (batch_A, batch_X) in enumerate(mol_dataloader):
            loss = wgan_loss(D, G, batch_A, batch_X)

            optimizaer_G.zero_grad()
            optimizaer_D.zero_grad()
            loss.backward(retain_graph=True)
            optimizaer_D.step()

            print("D %d %d: %f" % (n, i, loss))

            if i % 3 == 0:
                z = torch.randn([batch_size, DIM_Z], device=device, requires_grad=True, dtype=dtype)
                pred_G = G(z)
                L_fake = -torch.mean(D(pred_G))

                r_real = R([batch_A, batch_X])
                if n > 10:
                    r_fake = R(pred_G)
                    L_reward = torch.mean((r_fake-reward_mol(pred_G[0], pred_G[1]))**2
                                      + (r_real-reward_mol(batch_A, batch_X))**2)
                else:
                    L_reward = torch.mean((r_real - reward_mol(batch_A, batch_X)) ** 2)

                L_generator = L_fake + L_reward

                optimizaer_G.zero_grad()
                optimizaer_D.zero_grad()
                L_generator.backward()
                optimizaer_G.step()

                print("G %d: %f" % (i, L_generator))

    torch.save(G.state_dict(), "data/generator.pth")
    torch.save(D.state_dict(), "data/discriminator.pth")
    torch.save(R.state_dict(), "data/rewardnet.pth")
