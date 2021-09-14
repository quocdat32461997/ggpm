from torch.utils.data import Dataset
from rdkit import Chem

from ggpm.utils import *

class MoleculeDataset(Dataset):
    def __init__(self, data, vocab):
        super(MoleculeDataset, self).__init__()

        self.vocab = vocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return None