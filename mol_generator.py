from Utils.utils import *
from Model.model import Generator, Discriminator


if __name__ == "__main__":
    G = Generator().to(device)
    G.load_state_dict(torch.load("data/generator.pth"))

    smiles_list = []
    num_sample = 10000
    batch_size = 100

    with torch.no_grad():
        for i in tqdm(range(int(num_sample/batch_size))):
            z = torch.randn([batch_size, DIM_Z], device=device, requires_grad=True)
            pred_A, pred_X = G(z)

            for j in range(pred_A.size()[0]):
                mol = mat2graph(pred_A[j, ], pred_X[j, ])
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    # print(Chem.GetAdjacencyMatrix(mol))
                    # print([atom.GetSymbol() for atom in mol.GetAtoms()])
                    # print(smiles)
                    smiles_list.append(smiles)

    with open("data/generatedSMILES.smi", mode="w") as f:
        for smiles in smiles_list:
            f.write(smiles+"\n")
