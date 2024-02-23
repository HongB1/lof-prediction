from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
class LOFDataset(Dataset):
    """Face Landmarks dataset."""
    

    def __init__(self, data_name, pkl_file=None, root_dir=None, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_name = data_name
        path = '/data_hdd4/hb/lof/data'
        
        if pkl_file:
            self.dataset_frame = pd.read_pickle(pkl_file)
        else:
            self.dataset_frame = pd.read_pickle(os.path.join(path, f'{data_name}_dataset_info.pkl'))
        self.dataset_frame = self.dataset_frame.sample(frac=1, random_state=42).reset_index(drop=True)

        if not root_dir:
            self.root_dir = os.path.join(path, f'{data_name}_input_data_pickle_new_label')
        else:
            self.root_dir = root_dir

        sample_id = self.dataset_frame.sample_id[0]
        data = pd.read_pickle(os.path.join(self.root_dir, sample_id))
        # return data
        # print(data)
        input_feature_matrix = data.x
        self.node_num = data.num_nodes
        edge_index = data.edge_index
        input_feature_matrix = torch.from_numpy(input_feature_matrix)
        self.unique_label = self.dataset_frame.label.unique()

    def __len__(self):
        return len(self.dataset_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_id = self.dataset_frame.sample_id[idx]
        data = pd.read_pickle(os.path.join(self.root_dir, sample_id))
        # return data
        # print(data)
        input_feature_matrix = data.x
        self.node_num = data.num_nodes
        edge_index = data.edge_index
        input_feature_matrix = torch.from_numpy(input_feature_matrix)
        label = self.dataset_frame.label[idx]
        self.unique_label = self.dataset_frame.label.unique()
        sample = {'id': sample_id.split('.')[:-1][0], 'matrix': input_feature_matrix, 'label': label}
        return data
    
    def __repr__(self):
        return f'LOFDataset: {self.data_name}, Node Num: {self.node_num}, Perturb Num: {len(self.unique_label)}, Total Data Num: {len(self.dataset_frame)}'