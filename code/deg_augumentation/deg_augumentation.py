import scanpy
import pandas as pd
import anndata
import os
import hdf5plugin
import os
import pickle as pkl

import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import pybiomart
import mygene
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = "/data_hdd4/hb/lof/data/raw_data_after_preprocessing/norman_for_deg_processing_gsym_dup_remove.h5ad"
adata = anndata.read_h5ad(path)

metadata = adata.obs
counts_df = adata.to_df()
metadata_ctrl_total = metadata.loc[metadata.condition == 'control'].index.tolist()
for condition in list(set(metadata.condition.unique()) - set(['control'])):
    """condition 별 폴더 생성"""
    save_dir = f'/data_hdd4/hb/lof/data/deg_augmentation/{condition}'
    os.makedirs(save_dir, exist_ok=True)
    print(f'{condition} 데이터 생성 시작!')

    """perturb와 control 데이터셋 생성"""
    metadata_perturb_total = metadata.loc[metadata.condition == condition].index.tolist()

    perturb_sampling_index_sets = {}
    ctrl_sampling_index_sets = {}
    random_int = np.random.randint(1, 100000, size=len(metadata_perturb_total))

    for seed_value in random_int:
        np.random.seed(seed_value)

        perturb_sampling_index_sets[seed_value] = np.random.choice(metadata_perturb_total, size=30, replace=False)
        ctrl_sampling_index_sets[seed_value] = np.random.choice(metadata_ctrl_total, size=30, replace=False)

    for seed_value, perturb_sampling_index, ctrl_sampling_index in zip(perturb_sampling_index_sets.keys(), perturb_sampling_index_sets.values(), ctrl_sampling_index_sets.values()):
        
        metadata_ctrl = metadata.loc[ctrl_sampling_index]
        metadata_pertb = metadata.loc[perturb_sampling_index]

        metadata_total = pd.concat([metadata_pertb, metadata_ctrl]).sort_index()
        counts_df_total = counts_df[counts_df.index.isin(metadata_total.index)].sort_index()

        inference = DefaultInference(n_cpus=72)
        dds = DeseqDataSet(
            counts=counts_df_total,
            metadata=metadata_total,
            design_factors="condition",
            refit_cooks=True,
            inference=inference,
            ref_level = ['condition', 'control']
        )
        dds.deseq2()
        lfc = dds.varm['LFC']

        column = [x for x in lfc.columns if 'condition' in x]
        lfc = lfc.dropna(axis=0)
        lfc.sort_values(by=column, ascending=False, inplace=True, ignore_index=False)
        lfc.to_csv(f"{save_dir}/{condition}_{seed_value}.csv")