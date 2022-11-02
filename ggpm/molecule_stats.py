import psi4
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from molmass import Formula
from typing import List

psi4.core.set_output_file("output.dat", True)


class MoleculeStats(object):
    def __init__(self, memory_limit=4, num_threads=4):
        psi4.set_memory('{} GB'.format(memory_limit))
        psi4.set_num_threads(num_threads)

    @staticmethod
    def get_mass(smiles_list):
        assert isinstance(smiles_list, List[str])
        masses = [Formula(smiles).mass for smiles in smiles_list]
        return masses

    @staticmethod
    def mol2xyz(mol):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        AllChem.UFFOptimizeMolecule(mol)
        atoms = mol.GetAtoms()
        string = "\n"
        for i, atom in enumerate(atoms):
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
        string += "units angstrom\n"

        return string, mol

    @staticmethod
    def _calculate_homo_lumo(smiles):
        mol = Chem.MolFromSmiles(smiles)
        xyz, mol = MoleculeStats.mol2xyz(mol)

        mol_geom = psi4.geometry(xyz)
        scf_e, scf_wfn = psi4.energy("B3LYP/cc-pVDZ", return_wfn=True)

        HOMO = scf_wfn.epsilon_a_subset('AO', 'ALL').np[scf_wfn.nalpha()]
        LUMO = scf_wfn.epsilon_a_subset('AO', 'ALL').np[scf_wfn.nalpha() + 1]

        return HOMO, LUMO

    @staticmethod
    def get_homo_lumo(smiles_list):
        # calculation of HOMO/LUMO inspired from https://iwatobipen.wordpress.com/2018/08/24/calculate-homo-and-lumo-with-psi4-rdkit-psi4/
        assert isinstance(smiles_list, List[str])

        homos, lumos = [], []
        for smiles in smiles_list:
            h, l = MoleculeStats._calculate_homo_lumo(smiles)
            homos.append(h)
            lumos.append(l)

        return homos, lumos
