""" dataset.py: Contains a custom dataset class for the Axelrod spin-flip dataset. """

__author__  = "Rohit Dilip"
__email__   = "rdilip@caltech.edu"

import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import from_smiles
torch.set_default_dtype(torch.float64)

import os
import tarfile
import warnings
import msgpack
from tqdm import tqdm

from typing import Callable, List

def default_pre_transform(data: dict) -> Data:
    """ Default pre-transform function for the dataset. Encodes smiles string into molecular graph,
    and sets y to be the gap between the first excited state energy and the SCF ground state energy.
    """
    transformed = from_smiles(data['species']['smiles'])
    transformed.y = torch.Tensor([[data['props']['excitedstates'][0]['energy']]])
                                   #   data['props']['totalenergy']
                                   #k]])
    transformed.y /= 627.5 # convert to kcal / mol
    transformed.x = transformed.x.float()
    pts = torch.Tensor(data['xyz'])[:, 1:]
    distance = torch.cdist(pts, pts)

    ei = transformed.edge_index
    transformed.edge_weight = distance[ei[0], ei[1]]

    return transformed

def schnet_transform(data: dict) -> Data:
    """ Transforms to input for SchNet """
    transformed = from_smiles(data['species']['smiles'])
    transformed.y = torch.Tensor([[data['props']['excitedstates'][0]['energy'] -\
                                      data['props']['totalenergy']
                                   ]])
    coords = torch.Tensor(data['xyz'])[:, 1:]
    charges = torch.Tensor(data['xyz']).long()[:, 0]
    transformed.z = charges
    transformed.pos = coords
    return transformed

def GeometricTransform(method: str) -> Callable[dict, Data]:
    assert method in ["normalize", "log", "gap", "raw", "loggap"]
    def geometric_transform(data: dict) -> Data:
        transformed = from_smiles(data['species']['smiles'])
        transformed.y = torch.Tensor([[data['props']['excitedstates'][0]['energy']]])
        transformed.ground_state = torch.Tensor([[data['props']['totalenergy']]])
        coords = torch.Tensor(data['xyz'])[:, 1:]
        ixs = torch.Tensor(data['xyz']).long()[:, 0]
        # Hardcoded carbon and nitrogen values.
        ixs[ixs == 6] = 0
        ixs[ixs == 7] = 2
        ixs = F.one_hot(ixs).float()
        transformed.y /= 627.5 # convert to kcal / mol
        transformed.ground_state /= 627.5 # convert to kcal / mol

        if method == "gap":
            transformed.y -= data['props']['totalenergy'] / 627.5
        elif method == "log":
            transformed.y = torch.log(-transformed.y)
        elif method == "normalize":
            coords = (coords - coords.mean(dim=0)) / coords.std(dim=0)
        elif method == "loggap":
            transformed.y -= data['props']['totalenergy'] / 627.5
            assert transformed.y.min() > 0
            transformed.y = torch.log(transformed.y)

        transformed.x = torch.hstack((ixs, coords))
        return transformed
    return geometric_transform


def geometric_transform(data: dict) -> Data:
    """ Geometric transformation for dataset. One hot encodes atomic numbers and concatenates 
        with coordinate information. No edge information is included.
    """
    transformed = from_smiles(data['species']['smiles'])
    transformed.y = torch.Tensor([[data['props']['excitedstates'][0]['energy']]])
            #]])#data['props']['totalenergy']
    transformed.y /= 627.5 # convert to kcal / mol
    coords = torch.Tensor(data['xyz'])[:, 1:]
    ixs = torch.Tensor(data['xyz']).long()[:, 0]

    # Hardcoded carbon and nitrogen values.
    ixs[ixs == 6] = 0
    ixs[ixs == 7] = 2
    ixs = F.one_hot(ixs).float()
    transformed.x = torch.hstack((ixs, coords))

    return transformed

def MolecularFilter(formula: str='C12H10N2') -> Callable:
    """ Filters by chemical formula. Default is azobenzene. """
    def filter(data: dict) -> bool:
        matches_formula = data['stoichiometry']['formula'] == formula
        ordered = data['props']['totalenergy'] < data['props']['excitedstates'][0]['energy']
        return matches_formula and ordered
        
    return filter

class ExcitedDataset(InMemoryDataset):
    """ Excited state dataset of azobenzene molecules. """
    def __init__(self, root: str,
                       transform: Callable=None, 
                       pre_transform: Callable=default_pre_transform, 
                       pre_filter: Callable=MolecularFilter()):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filenames = [f"{i:04}.msgpack" for i in range(12)]
        return filenames

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        fnames = self.raw_file_names
        download_url_base = "https://data.materialsdatafacility.org/mdf_open/"\
                                "axelrod_azoflip_spinflip_derivatives_v1.3/separate_files/"
        # Cases: unzipped file exists, don't download
        #       zipped file exists, unzip
        #       unzipped file doesn't exist, download and unzip
        m = 0
        for fname in fnames:
            if os.path.exists(f"{self.raw_dir}/{fname}"):
                continue
            elif os.path.exists(f"{self.raw_dir}/{fname}.tar.gz"):
                self._extract_tar(fname)
            else:
                if m == 0:
                    warnings.warn("Downloading files...this may take a while.")
                if m > 3:
                    warnings.warn("Seriously, this is going to take some time. "\
                                        "You may want to take a nap or something.")
                url = f"{download_url_base}{fname}.tar.gz"
                download_url(url, self.raw_dir)
                self._extract_tar(fname)

                m += 1

    def process(self):
        data_list = []
        for raw_path in tqdm(self.raw_paths, desc="Loading data..."):
            with open(raw_path, "rb") as f:
                unpacker = msgpack.Unpacker(f, strict_map_key=False)
                data = next(unpacker)
                for k in data.keys():
                    data_list.append(data[k])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data\
                    in tqdm(data_list, desc="Processing data...")]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def _extract_tar(self, fname: str):
        tar = tarfile.open(f"{self.raw_dir}/{fname}.tar.gz")
        for member in tar.getmembers():
            tar.extract(member, self.raw_dir)

