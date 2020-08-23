import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.include import *


# maximum number of nodes in a graph
MAX_NODE = 15

#
NUM_ATOM = 10

#
NUM_BOND = 4

#
DIM_Z = 32

#
dtype = torch.float
device = torch.device("cuda")

# atom idx , E = empty
ATOM_IDX = {"E": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8, "I": 9}
ATOM_IDX_INV = {0: "E", 1: "C", 2: "N", 3: "O", 4: "F", 5: "P", 6: "S", 7: "Cl", 8: "Br", 9: "I"}
PERIODIC_TABLE = {"C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53, }

# bond idx
BOND_IDX = {"ZERO": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}


def read_smilesset(path):
    smiles_list = []
    with open(path) as f:
        for smiles in f:
            smiles_list.append(smiles.rstrip())

    return smiles_list


def mat2graph(A: torch.tensor, X: torch.tensor):
    A = torch.argmax(A, dim=2)
    A = A.detach().cpu().numpy().copy()
    X = X.detach().cpu().numpy().copy()

    atoms = []
    delete_list = []
    for i in range(X.shape[0]):
        ind = np.argmax(X[i])
        if ind == 0:
            delete_list.append(i)
        else:
            atoms.append(PERIODIC_TABLE[ATOM_IDX_INV[ind]])

    delete_list.reverse()
    for i in delete_list:
        A = np.delete(A, i, 0)
        A = np.delete(A, i, 1)

    mol = Chem.RWMol()

    node_to_idx = {}
    for i in range(len(atoms)):
        a = Chem.Atom(atoms[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    for ix, row in enumerate(A):
        for iy, bond in enumerate(row):

            if iy <= ix:
                continue

            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    mol = mol.GetMol()

    return mol


def reward_mol(A: torch.tensor, X: torch.tensor):
    mols = []
    for i in range(A.size()[0]):
        mols.append(mat2graph(A[i, ], X[i, ]))

    rewards = []
    for mol in mols:
        if mol is not None:
            try:
                rewards.append(QED.qed(mol))
            except:
                rewards.append(-1)
        else:
            rewards.append(-1)

    return torch.tensor(rewards, device=device, requires_grad=True, dtype=dtype)
