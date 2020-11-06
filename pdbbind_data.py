import numpy as np
import pandas as pd
import pybel
from utils import *
import os

if not os.path.isdir("data"):
    os.system("mkdir data")

affinity_data = pd.read_csv("affinity_data.csv", comment='#')
with open("missing.csv", 'r') as f:
    missing = f.read().splitlines()

affinity_data = affinity_data[~np.in1d(affinity_data['pdbid'], missing)]

print(affinity_data['-logKd/Ki'].isnull().any())

with open("core_pdbbind2016.ids", 'r') as f:
    core_set = f.read().splitlines()
core_set = set(core_set)

with open("refined_pdbbind2016.ids", 'r') as f:
    refined_set = f.read().splitlines()
refined_set = set(refined_set)

general_set = set(affinity_data['pdbid'])

assert core_set & refined_set == core_set
assert refined_set & general_set == refined_set

print(len(general_set), len(refined_set), len(core_set))

with open("core_pdbbind2013.ids", 'r') as f:
    core2013 = f.read().splitlines()
core2013 = set(core2013)

affinity_data['include'] = True
affinity_data.loc[np.in1d(affinity_data['pdbid'], list(core2013 & (general_set - core_set))), 'include'] = False

affinity_data.loc[np.in1d(affinity_data['pdbid'], list(general_set)), 'set'] = 'general'
affinity_data.loc[np.in1d(affinity_data['pdbid'], list(refined_set)), 'set'] = 'refined'
affinity_data.loc[np.in1d(affinity_data['pdbid'], list(core_set)), 'set'] = 'core'
print(affinity_data.head())
print(affinity_data[affinity_data['include']].groupby('set').apply(len).loc[['general', 'refined', 'core']])

dataset_path = {'general': 'general-set-except-refined', 'refined': 'refined-set', 'core': 'refined-set'}

path = "database/"
max_len = 200

# features for grpah
adjs_float_2013 = [[]]
adjs_1_2013 = [[]]
adjs_2_2013 = [[] for num in range(2)]
adjs_4_2013 = [[] for num in range(4)]
adjs_8_2013 = [[] for num in range(8)]
feats_2013 = []
labels_2013 = []

skipped = []

j = 0
for dataset_name, data in affinity_data.groupby('set'):

    print(dataset_name, 'set')
    i = 0
    ds_path = dataset_path[dataset_name]

    adjs_float = [[]]
    adjs_1 = [[]]
    adjs_2 = [[] for num in range(2)]
    adjs_4 = [[] for num in range(4)]
    adjs_8 = [[] for num in range(8)]
    feats = []
    labels = []

    for _, row in data.iterrows():
        name = row['pdbid']
        affinity = row['-logKd/Ki']
        ligand = next(pybel.readfile('mol2', '%s/%s/%s/%s_ligand.mol2' %(path, ds_path, name, name)))
        pocket = next(pybel.readfile('mol2', '%s/%s/%s/%s_pocket.mol2' %(path, ds_path, name, name)))
        # moltype : pocket 1, ligand 0
        complex_atoms = get_atoms(ligand, pocket)
        if len(complex_atoms) > max_len:
            print("more atoms then %d (%s set)" %(max_len, dataset_name))
            skipped.append(name)
            continue
            
        feat, adj_float, adj_1, adj_2, adj_4, adj_8 = graph_featurizer(complex_atoms, max_len)
        if row['include']:
            feats.append(feat)
            labels.append([name, affinity])
            adjs_float[0].append(adj_float[0])
            adjs_1[0].append(adj_1[0])
            for num in range(2): adjs_2[num].append(adj_2[num])
            for num in range(4): adjs_4[num].append(adj_4[num])
            for num in range(8): adjs_8[num].append(adj_8[num])
            i += 1
        else:
            feats_2013.append(feat)
            labels_2013.append([name, affinity])
            adjs_float_2013[0].append(adj_float[0])
            adjs_1_2013[0].append(adj_1[0])
            for num in range(2): adjs_2_2013[num].append(adj_2[num])
            for num in range(4): adjs_4_2013[num].append(adj_4[num])
            for num in range(8): adjs_8_2013[num].append(adj_8[num])
            j += 1
        if (dataset_name == 'core') & (name in core2013):
            feats_2013.append(feat)
            labels_2013.append([name, affinity])
            adjs_float_2013[0].append(adj_float[0])
            adjs_1_2013[0].append(adj_1[0])
            for num in range(2): adjs_2_2013[num].append(adj_2[num])
            for num in range(4): adjs_4_2013[num].append(adj_4[num])
            for num in range(8): adjs_8_2013[num].append(adj_8[num])
    np.save("data/%s_feat.npy" %dataset_name, np.array(feats))
    np.save("data/%s_label.npy" %dataset_name, np.array(labels))
    np.save("data/%s_adjfloat.npy" %dataset_name, np.array(adjs_float))
    np.save("data/%s_adj1.npy" %dataset_name, np.array(adjs_1))
    np.save("data/%s_adj2.npy" %dataset_name, adjs_2)
    np.save("data/%s_adj4.npy" %dataset_name, adjs_4)
    np.save("data/%s_adj8.npy" %dataset_name, adjs_8)

    print('prepared', i, 'complexes')
print('excluded', j, 'complexes')

np.save("data/core2013_feat.npy", np.array(feats_2013))
np.save("data/core2013_label.npy", np.array(labels_2013))
np.save("data/core2013_adjfloat.npy", np.array(adjs_float_2013))
np.save("data/core2013_adj1.npy", np.array(adjs_1_2013))
np.save("data/core2013_adj2.npy", adjs_2_2013)
np.save("data/core2013_adj4.npy", adjs_4_2013)
np.save("data/core2013_adj8.npy", adjs_8_2013)

with open('skipped.txt', 'w') as f:
    for x in skipped:
        f.write("%s\n" %x)
