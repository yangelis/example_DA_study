# %%
"""
Example of a chronjob
"""

# %%
import tree_maker
from tree_maker import NodeJob
import pandas as pd
import awkward as ak
from joblib import Parallel, delayed

#from dask import dataframe as dd
# %%
# Load the tree from a yaml
try:
    root=tree_maker.tree_from_json(
    f'./study_000/tree.json')
except Exception as e:
    print(e)
    print('Probably you forgot to edit the address of you json file...')

my_list=[]
if root.has_been('completed'):
    print('All descendants of root are completed!')
    for node in root.generation(2):
	#os.sytem(f'bsub cd {node.path} &&  {node.path_template} ')
        #my_list.append(pd.read_parquet(f'{node.path}/test.parquet', columns=['x']).iloc[-1].x)
        my_list.append(node.has_been('completed'))
	#my_list.append(ak.from_parquet(f'{node.path}/test.parquet', columns=['x'])[-1,'x'])
    #Parallel(n_jobs=16)(delayed(node.has_been)('completed') for node in root.generation(2))
    #print(my_list)
else:
    print('Complete first all jobs')

