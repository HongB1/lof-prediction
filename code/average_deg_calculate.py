"""모수의 평균 DEG를 구하는 코드
표본 평균 DEG를 구하고 싶을 땐, 아래에 sampling하고 싶을 때를 적용하면 됨. """



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
# adata = anndata.read_h5ad("/home/hb/python/lof/deg/norman/norman_for_deg_processing.h5ad")
# adata

# dataset = pybiomart.Dataset(name='hsapiens_gene_ensembl',
#  host='http://www.ensembl.org')
# gene_name_dict = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'],)
# gene_name_dict = gene_name_dict.set_index('Gene stable ID').to_dict()['Gene name']
# gene_name_dict




adata = anndata.read_h5ad("/home/hb/python/lof/deg/norman/norman_for_deg_processing.h5ad")

mg = mygene.MyGeneInfo()
ens = adata.var.index.tolist()
ginfo = mg.querymany(ens, scopes='ensembl.gene', fields='symbol', species='human')
gene_name_dict = {}
for g in ginfo:
    if 'symbol' in g.keys():
        _query = g['query']
        _symbol = g['symbol']
        gene_name_dict[_query]  = _symbol


metadata = adata.obs
counts_df = adata.to_df()
already_calculate = [x.split('.')[0] for x in os.listdir("/home/hb/python/lof/deg/norman") if '.csv' in x]
already_calculate.append('control')

for condition in [x for x in metadata['condition'].unique() if x not in already_calculate]:
    print(f'{condition} DEG analysis start!')
    metadata_toy = metadata[metadata['condition'].isin([condition, 'control'])]

    """sampling하고 싶을 때 코드
    metadata_toy_ctrl = metadata_toy[metadata_toy['condition']=='control'].sample(n=10)
    metadata_toy_perturb = metadata_toy[metadata_toy['condition']==condition].sample(n=10)
    metadata_toy = pd.concat([metadata_toy_perturb, metadata_toy_ctrl]).sort_index()
    counts_df_toy = counts_df[counts_df.index.isin(metadata_toy.index)].sort_index()
    """
    
    counts_df_toy = counts_df[counts_df.index.isin(metadata_toy.index)]
    inference = DefaultInference(n_cpus=72)
    dds = DeseqDataSet(
        counts=counts_df_toy,
        metadata=metadata_toy,
        design_factors="condition",
        refit_cooks=True,
        inference=inference,
        ref_level = ['condition', 'control']
    )
    dds.deseq2()
    lfc = dds.varm['LFC']
    lfc['gene_name'] = [gene_name_dict[x] if x in gene_name_dict.keys() else float('nan') for x in lfc.index]
    lfc = lfc[lfc['gene_name'].notna()]
    column = lfc.columns[1]
    lfc = lfc.dropna(axis=0)
    lfc['gene_id'] = lfc.index
    lfc.set_index('gene_name', append=False, inplace=True)
    lfc.sort_values(by=column, ascending=False, inplace=True, ignore_index=False)
    lfc.to_csv(f"/home/hb/python/lof/deg/norman/{condition}.csv")
    print(f'{condition}.csv saved!')
