import os
import pickle
import numpy as np
import argparse
import pybel
from utils import *

max_len = 200
dock_num = 3

parser = argparse.ArgumentParser(description="Split dataset into training, validation.")
parser.add_argument('--input_path', '-i', default='data', help='directory with pdbbind dataset')
parser.add_argument('--output_path', '-o', default='data', help='directory to store output files')
parser.add_argument('--size_val', '-s', type=int, default=369, help='number of samples in the validation set')
args = parser.parse_args()

with open("docking_dict.pickle", 'rb') as f:
    docking_dict = pickle.load(f)

refined_feat = np.load("%s/refined_feat.npy" %args.input_path)
refined_label = np.load("%s/refined_label.npy" %args.input_path)
refined_adjfloat = np.load("%s/refined_adjfloat.npy" %args.input_path)
refined_adj1 = np.load("%s/refined_adj1.npy" %args.input_path)
refined_adj2 = np.load("%s/refined_adj2.npy" %args.input_path)
refined_adj4 = np.load("%s/refined_adj4.npy" %args.input_path)
refined_adj8 = np.load("%s/refined_adj8.npy" %args.input_path)
idx = np.arange(0, refined_label.shape[0])
np.random.shuffle(idx)
adjs_float = [[]]
adjs_1 = [[]]
adjs_2 = [[] for num in range(2)]
adjs_4 = [[] for num in range(4)]
adjs_8 = [[] for num in range(8)]
feats = []
labels = []
val_count = 0
for i in idx[:args.size_val]:
    feats.append(refined_feat[i])
    labels.append(refined_label[i])
    adjs_float[0].append(refined_adjfloat[0][i])
    adjs_1[0].append(refined_adj1[0][i])
    for num in range(2): adjs_2[num].append(refined_adj2[num][i])
    for num in range(4): adjs_4[num].append(refined_adj4[num][i])
    for num in range(8): adjs_8[num].append(refined_adj8[num][i])
    val_count += 1
    name = refined_label[i][0]
    affinity = refined_label[i][1]
    if not name in docking_dict.keys():
        continue
    cnt = 0
    for y in docking_dict[name]:
        ligand = next(pybel.readfile('pdbqt', "data/docking/%s/%s" %(name, y)))
        pocket = next(pybel.readfile('mol2', "database/refined-set/%s/%s_pocket.mol2" %(name, name)))
        complex_atoms = get_atoms(ligand, pocket)
        if len(complex_atoms) > max_len:
            print("more atoms then %d (refined set)" %max_len)
            continue
        feat, adj_float, adj_1, adj_2, adj_4, adj_8 = graph_featurizer(complex_atoms, max_len)
        feats.append(feat)
        labels.append([name, affinity])
        adjs_float[0].append(adj_float[0])
        adjs_1[0].append(adj_1[0])
        for num in range(2): adjs_2[num].append(adj_2[num])
        for num in range(4): adjs_4[num].append(adj_4[num])
        for num in range(8): adjs_8[num].append(adj_8[num])
        val_count += 1
        cnt += 1
        if cnt > dock_num:
            break

print('prepared', val_count, 'complexes for validation')

np.save("%s/validation_feat.npy" %args.output_path, np.array(feats))
np.save("%s/validation_label.npy" %args.output_path, np.array(labels))
np.save("%s/validation_adjfloat.npy" %args.output_path, np.array(adjs_float))
np.save("%s/validation_adj1.npy" %args.output_path, np.array(adjs_1))
np.save("%s/validation_adj2.npy" %args.output_path, np.array(adjs_2))
np.save("%s/validation_adj4.npy" %args.output_path, np.array(adjs_4))
np.save("%s/validation_adj8.npy" %args.output_path, np.array(adjs_8))
adjs_float = [[]]
adjs_1 = [[]]
adjs_2 = [[] for num in range(2)]
adjs_4 = [[] for num in range(4)]
adjs_8 = [[] for num in range(8)]
feats = []
labels = []
train_cnt = 0
for i in idx[args.size_val:]:
    feats.append(refined_feat[i])
    labels.append(refined_label[i])
    adjs_float[0].append(refined_adjfloat[0][i])
    adjs_1[0].append(refined_adj1[0][i])
    for num in range(2): adjs_2[num].append(refined_adj2[num][i])
    for num in range(4): adjs_4[num].append(refined_adj4[num][i])
    for num in range(8): adjs_8[num].append(refined_adj8[num][i])
    train_cnt += 1
    name = refined_label[i][0]
    affinity = refined_label[i][1]
    if not name in docking_dict.keys():
        continue
    cnt = 0
    for y in docking_dict[name]:
        ligand = next(pybel.readfile('pdbqt', "data/docking/%s/%s" %(name, y)))
        pocket = next(pybel.readfile('mol2', "database/refined-set/%s/%s_pocket.mol2" %(name, name)))
        complex_atoms = get_atoms(ligand, pocket)
        if len(complex_atoms) > max_len:
            print("more atoms then %d (refined set)" %max_len)
            continue
        feat, adj_float, adj_1, adj_2, adj_4, adj_8 = graph_featurizer(complex_atoms, max_len)
        feats.append(feat)
        labels.append([name, affinity])
        adjs_float[0].append(adj_float[0])
        adjs_1[0].append(adj_1[0])
        for num in range(2): adjs_2[num].append(adj_2[num])
        for num in range(4): adjs_4[num].append(adj_4[num])
        for num in range(8): adjs_8[num].append(adj_8[num])
        train_cnt += 1
        cnt += 1
        if cnt > dock_num:
            break

print('prepared', train_cnt, 'complexes for training')

np.save("%s/training_feat.npy" %args.output_path, np.array(feats))
np.save("%s/training_label.npy" %args.output_path, np.array(labels))
np.save("%s/training_adjfloat.npy" %args.output_path, np.array(adjs_float))
np.save("%s/training_adj1.npy" %args.output_path, np.array(adjs_1))
np.save("%s/training_adj2.npy" %args.output_path, adjs_2)
np.save("%s/training_adj4.npy" %args.output_path, adjs_4)
np.save("%s/training_adj8.npy" %args.output_path, adjs_8)
