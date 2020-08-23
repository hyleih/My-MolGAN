import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


if __name__ == "__main__":
    smiles_list = read_smilesset("data/zinc_under15.smi")

    N = 14
    sub_smiles = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        atoms = mol.GetAtoms()

        if len(atoms) < N:
            sub_smiles.append(smiles)

    print(len(smiles_list), len(sub_smiles))
    with open(f"data/zinc_under{N}.smi", mode="w") as f:
        for smiles in sub_smiles:
            f.write(smiles+"\n")


