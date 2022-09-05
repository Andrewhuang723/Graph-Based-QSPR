from dgl import DGLGraph
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from typing import List, Union
import networkx as nx

device = torch.device("cuda")


class Featurization_parameters:
    """
    According to Chemprop from MIT.
    """
    def __init__(self) -> None:
        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            "atomic_num": list(range(self.MAX_ATOMIC_NUM)),
            # "atomic_num": ["C", "H", "O", "N", "S"],
            "degree": [0, 1, 2, 3, 4, 5],
            "formal_charge": [-1, -2, 1, 2, 0],
            "chiral_tag": [0, 1, 2, 3],
            "num_Hs": [0, 1, 2, 3, 4],
            "hybridization": [
                 Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2
            ]
        }
        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass; + 3 for atom positions
        self.ATOM_FDIM = sum(len(choices) for choices in self.ATOM_FEATURES.values()) + 2
        self.BOND_FDIM = 13 + 10 # + 10 for bond length
        self.EXPLICIT_H = False

PARAMS = Featurization_parameters()
# PARAMS.EXPLICIT_H = True

def atom_features(atom: Chem.rdchem.Atom):
    if atom is None:
        features = [0] * (PARAMS.ATOM_FDIM) # if add 3D positions, then (PARAMS.ATOM_FDIM - 3)
    else:
        features = onek_encoding_unk(atom.GetAtomicNum(), PARAMS.ATOM_FEATURES["atomic_num"]) + \
                onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES["degree"]) + \
                onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES["formal_charge"]) + \
                onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
                onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES["num_Hs"]) + \
                onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES["hybridization"]) + \
                [1 if atom.GetIsAromatic() else 0] + \
                [atom.GetMass() * 0.01]
    return features

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 11) # if bond length, then (PARAMS.BOND_FDIM - 2)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def get_mol(smiles, is_explict_H: bool=False):
    if is_explict_H:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
        else:
            return None
    else:
        mol = Chem.MolFromSmiles(smiles)

    Chem.Kekulize(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d) # 3D coordinate positions
    return mol


def get_atom_positions(Mol: Chem.Mol, is_explicit_H: bool=False):
    MolBlock = Chem.MolToMolBlock(Mol)
    if is_explicit_H:
        Mol = Chem.MolFromMolBlock(MolBlock, removeHs=False)
    else:
        Mol = Chem.MolFromMolBlock(MolBlock)
    return Mol.GetConformers()[0].GetPositions()


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def onek_encoding_bondlength(x):
    distance_bins = [(0, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 4.5), (4.5, 5), (5, 5.5), (5.5, 6), (6, np.Inf)]
    return [x < s[1] and x >= s[0] for s in distance_bins]


class Mol2Graph:
    def __init__(self, Mol: Chem.Mol):
        self.Mol = Mol
        self.Graph = DGLGraph()
        self.bond_src = [] #edge source node
        self.bond_dst = [] #edge destination node
        self.is_explicit_H = False

        self.f_atoms = [atom_features(atom) for atom in Mol.GetAtoms()]
        # self.addAtomPositions() # Add atom 3D positions

        self.n_atoms = len(self.f_atoms)

        self.n_bonds = self.Mol.GetNumBonds()
        self.f_bonds = []
        self.is_all_edges = False

    @property
    def atom_positions(self):
        if self.is_explicit_H:
            return get_atom_positions(self.Mol, self.is_explicit_H)
        else:
            return get_atom_positions(self.Mol)

    def addAtomPositions(self):
        for f, p in zip(self.f_atoms, self.atom_positions):
            f.extend(p)

    def BondLength(self, u: int, v: int):
        u_pos = self.atom_positions[u]
        v_pos = self.atom_positions[v]
        distance = np.sqrt(np.sum(np.square(u_pos - v_pos)))
        return onek_encoding_bondlength(distance)

    def addNodes(self):
        """
        Add nodes into graph
        """
        self.Graph.add_nodes(self.n_atoms)
        self.Graph.ndata["h"] = torch.Tensor(self.f_atoms)


    def addEdges(self):
        """
        Add edges into graph
        """
        for bond_idx in range(self.n_bonds):
            bond = self.Mol.GetBondWithIdx(bond_idx)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bl = self.BondLength(u, v) # calculated bond length
            if self.is_shuffle:
                u = self.shuffle_idx.index(u)
                v = self.shuffle_idx.index(v)
            self.bond_src.extend([u, v])
            self.bond_dst.extend([v, u])

            bond_f = bond_features(bond)
            bond_f.extend(bl) # append bond length
            self.f_bonds.append(bond_f) # direct
            self.f_bonds.append(bond_f) # reversed direct

        self.Graph.add_edges(self.bond_src, self.bond_dst)
        tb = torch.Tensor(self.f_bonds).squeeze(0)
        self.Graph.edata["h"] = tb
