import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


class MolDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, item, training=True):
        smiles = self.smiles_list[item]
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol, clearAromaticFlags=False)

        A = generate_adj(mol)
        X = generate_feture(mol)

        A = torch.tensor(A, device=device, requires_grad=True, dtype=dtype)
        X = torch.tensor(X, device=device, requires_grad=True, dtype=dtype)

        if training:
            A = A + torch.randn(A.size(), device=device, requires_grad=True, dtype=dtype)
            X = X + torch.randn(X.size(), device=device, requires_grad=True, dtype=dtype)

        return A, X


def generate_adj(mol):
        A = np.zeros([MAX_NODE, MAX_NODE, NUM_BOND])
        A[:, :, 0] = 1

        if mol is not None:
            bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())]for b in mol.GetBonds()]
            for i, j, b_type in bonds:
                A[i, j, :] = 0
                A[j, i, :] = 0
                A[i, j, BOND_IDX[b_type]] = 1
                A[j, i, BOND_IDX[b_type]] = 1

        return A


def generate_feture(mol):
        X = np.zeros([MAX_NODE, NUM_ATOM])
        if mol is not None:
            atoms = [ATOM_IDX[atom.GetSymbol()] for atom in mol.GetAtoms()]
            for i, a_type in enumerate(atoms):
                X[i, a_type] = 1

            for i in range(MAX_NODE-len(atoms)):
                X[len(atoms), 0] = 1

        return X


if __name__ == "__main__":
    smiles = "C.CC1C(C)(C)C12CC21CC1.Cl.N.O"
    mol = Chem.MolFromSmiles(smiles)
    print(Descriptors.MolWt(mol))
    Chem.Kekulize(mol, clearAromaticFlags=False)

    A = generate_adj(mol)
    X = generate_feture(mol)

    for i in range(NUM_BOND):
        print(A[:, :, i])

    A = torch.tensor(A, device=device, requires_grad=True, dtype=dtype)
    X = torch.tensor(X, device=device, requires_grad=True, dtype=dtype)

    mol = mat2graph(A, X)

    if mol is not None:
        print(Chem.MolToSmiles(mol))



