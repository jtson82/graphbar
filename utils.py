import numpy as np
import math

def graph_featurizer(c_atoms, max_len):
    atomic = {5:0, 6:1, 7:2, 8:3, 15:4, 16:5, 34:6, 9:7, 17:7, 35:7, 35:7, 53:7, 85:7}
    adj_float = [np.zeros((max_len, max_len), dtype=np.float16)]
    adj_1 = [np.zeros((max_len, max_len), dtype=np.int16)]
    adj_2 = [np.zeros((max_len, max_len), dtype=np.int16) for j in range(2)]
    adj_4 = [np.zeros((max_len, max_len), dtype=np.int16) for j in range(4)]
    adj_8 = [np.zeros((max_len, max_len), dtype=np.int16) for j in range(8)]
    adj_float[0][0:len(c_atoms), 0:len(c_atoms)] = np.eye(len(c_atoms))
    adj_1[0][0:len(c_atoms), 0:len(c_atoms)] = np.eye(len(c_atoms))
    for i in range(2): adj_2[i][0:len(c_atoms), 0:len(c_atoms)] = np.eye(len(c_atoms))
    for i in range(4): adj_4[i][0:len(c_atoms), 0:len(c_atoms)] = np.eye(len(c_atoms))
    for i in range(8): adj_8[i][0:len(c_atoms), 0:len(c_atoms)] = np.eye(len(c_atoms))
    feat = np.zeros((max_len, 13), dtype=np.float16)
    for i in range(len(c_atoms)):
        if c_atoms[i][0] in atomic.keys(): feat[i, atomic[c_atoms[i][0]]] = 1
        else: feat[i, 8] = 1
        feat[i, 9:13] = c_atoms[i][2:6]
        for j in range(len(c_atoms)):
            if (i == j): continue
            distance = math.sqrt((c_atoms[i][1][0]-c_atoms[j][1][0])**2 + (c_atoms[i][1][1]-c_atoms[j][1][1])**2 + (c_atoms[i][1][2]-c_atoms[j][1][2])**2)
            # build adjacency matrices
            if (distance>=0) and (distance<=0.5):
                adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                adj_2[0][i, j], adj_2[0][j, i] = 1, 1
                adj_4[0][i, j], adj_4[0][j, i] = 1, 1
                adj_8[0][i, j], adj_8[0][j, i] = 1, 1
            if (distance>0.5) and (distance<=1.0):
                adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                adj_2[0][i, j], adj_2[0][j, i] = 1, 1
                adj_4[0][i, j], adj_4[0][j, i] = 1, 1
                adj_8[1][i, j], adj_8[1][j, i] = 1, 1
            if (distance>1.0) and (distance<=1.5):
                adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                adj_2[0][i, j], adj_2[0][j, i] = 1, 1
                adj_4[1][i, j], adj_4[1][j, i] = 1, 1
                adj_8[2][i, j], adj_8[2][j, i] = 1, 1
            if (distance>1.5) and (distance<=2.0):
                adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                adj_2[0][i, j], adj_2[0][j, i] = 1, 1
                adj_4[1][i, j], adj_4[1][j, i] = 1, 1
                adj_8[3][i, j], adj_8[3][j, i] = 1, 1
            if c_atoms[i][6] != c_atoms[j][6]:
                if (distance>2.0) and (distance<=2.5):
                    adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                    adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                    adj_2[1][i, j], adj_2[1][j, i] = 1, 1
                    adj_4[2][i, j], adj_4[2][j, i] = 1, 1
                    adj_8[4][i, j], adj_8[4][j, i] = 1, 1
                if (distance>2.5) and (distance<=3.0):
                    adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                    adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                    adj_2[1][i, j], adj_2[1][j, i] = 1, 1
                    adj_4[2][i, j], adj_4[2][j, i] = 1, 1
                    adj_8[5][i, j], adj_8[5][j, i] = 1, 1
                if (distance>3.0) and (distance<=3.5):
                    adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                    adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                    adj_2[1][i, j], adj_2[1][j, i] = 1, 1
                    adj_4[3][i, j], adj_4[3][j, i] = 1, 1
                    adj_8[6][i, j], adj_8[6][j, i] = 1, 1
                if (distance>3.5) and (distance<=4.0):
                    adj_float[0][i, j], adj_float[0][j, i] = pow((4-distance),3)/64, pow((4-distance),3)/64
                    adj_1[0][i, j], adj_1[0][j, i] = 1, 1
                    adj_2[1][i, j], adj_2[1][j, i] = 1, 1
                    adj_4[3][i, j], adj_4[3][j, i] = 1, 1
                    adj_8[7][i, j], adj_8[7][j, i] = 1, 1
    return feat, adj_float, adj_1, adj_2, adj_4, adj_8

def get_atoms(ligand, pocket):
    p_atoms = [[atom.atomicnum, atom.coords, atom.hyb, atom.heavyvalence, atom.heterovalence, atom.partialcharge, 1] for atom in pocket if not atom.OBAtom.GetAtomicNum()==1]
    l_atoms = [[atom.atomicnum, atom.coords, atom.hyb, atom.heavyvalence, atom.heterovalence, atom.partialcharge, 0] for atom in ligand if not atom.OBAtom.GetAtomicNum()==1]
    c_atoms = [l_atom for l_atom in l_atoms]
    for p_atom in p_atoms:
        for l_atom in l_atoms:
            distance = math.sqrt((p_atom[1][0]-l_atom[1][0])**2 + (p_atom[1][1]-l_atom[1][1])**2 + (p_atom[1][2]-l_atom[1][2])**2)
            if distance <= 4:
                # pocket atoms within 4A from ligand
                c_atoms.append(p_atom)
                break
    return c_atoms

def load_data(set_num, data_type, adj_name):
    adj = np.load("%s/%s_adj%s.npy" %(set_num, data_type, adj_name))
    feat = np.load("%s/%s_feat.npy" %(set_num, data_type))
    label = np.load("%s/%s_label.npy" %(set_num, data_type))
    return adj, feat, label

def load_test(data_type, adj_name):
    adj = np.load("data/%s_adj%s.npy" %(data_type, adj_name))
    feat = np.load("data/%s_feat.npy" %(data_type))
    label = np.load("data/%s_label.npy" %(data_type))
    return adj, feat, label

def nor_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

def next_batch(idx, adj, feat, labels):
    adj_batch = [] 
    for i in range(len(adj)):
        adj_batch.append([])
        for j in idx:
            adj_batch[i].append(nor_adj(adj[i][j]))
    feat_batch = np.asarray([feat[i] for i in idx])
    labels_batch = np.asarray([labels[i] for i in idx])
    value_labels_batch = labels_batch[:,1].astype(np.float)
    pdbid_labels_batch = labels_batch[:,0]
    return np.asarray(adj_batch), feat_batch, value_labels_batch, pdbid_labels_batch
