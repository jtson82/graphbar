import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys
import time
import resource
import argparse

from layers.graph import *
from utils import *

# model parameter
parser = argparse.ArgumentParser(description="Training graphBAR model")
parser.add_argument('--dataset', '-s', help="training dataset path ")
parser.add_argument('--adj_type', '-at', default="4", help="type of adjacency matrix")
parser.add_argument('--gpu', '-gpu', default='0', help="gpu")
parser.add_argument('--output', '-o', help="directory for saving output files")
parser.add_argument('--testset', '-t', default='core', help="name of testset (core, core2013)")
args = parser.parse_args()
gpu_environ = args.gpu
adj_num = {"float": 1, "1": 1, "2": 2, "4": 4, "8": 8}
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_environ

if not os.path.isdir(args.output):
    os.system("mkdir " + args.output)

# parameter 
ite_epochs = 1000
l_sizes = [128, 128, 128, 32]
learning_rate = 0.001
batch_size = 32
patience = 10
max_len = 200

class gcn_block():
    def __init__(self, l_sizes, dropout_rate):
        self.l_sizes = l_sizes
        self.dropout_rate = dropout_rate

    def gcn_model(self, num, A, X):
        gc = GraphConvLayer(l_sizes[0], l_sizes[1], name="gc%d_0"%num, activation=tf.nn.relu)(adj_norm = A, x = X)
        gc = tf.layers.dense(gc, units=l_sizes[1], activation=tf.nn.relu)
        gc = tf.layers.dropout(gc, rate=self.dropout_rate)
        gc = GraphConvLayer(l_sizes[1], l_sizes[2], name="gc%d_1"%num, activation=tf.nn.relu)(adj_norm = A, x = gc)
        gc = tf.layers.dense(gc, units=l_sizes[2], activation=tf.nn.relu)
        gc = tf.layers.dropout(gc, rate=self.dropout_rate)
        gc = GraphConvLayer(l_sizes[2], l_sizes[3], name="gc%d_2"%num, activation=tf.nn.relu)(adj_norm = A, x = gc)
        gc = GraphGather()(x = gc)
        return gc

# build the model
ph = {
        'x': tf.placeholder(tf.float32, shape=[None, max_len, 13], name='x'),
        "labels": tf.placeholder(tf.float32, name="labels"),
        "dropout_rate": tf.placeholder(tf.float32, name="dropout_rate")}
adj_input = [tf.placeholder(tf.float32, shape=[None, max_len, max_len]) for i in range(adj_num[args.adj_type])]

pds1 = tf.layers.dense(inputs=ph["x"], units=l_sizes[0], activation=tf.nn.relu)
pdr1 = tf.layers.dropout(inputs=pds1, rate=ph["dropout_rate"])

gcn = gcn_block(l_sizes, ph["dropout_rate"])

gcl = [gcn.gcn_model(i, adj_input[i], pdr1) for i in range(adj_num[args.adj_type])]

con = tf.concat(gcl, 1)

ds1 = tf.layers.dense(inputs=con, units=l_sizes[3]*adj_num[args.adj_type]/2, activation=tf.nn.relu)
dr1 = tf.layers.dropout(inputs=ds1, rate=ph["dropout_rate"])

model = tf.layers.dense(inputs=dr1, units=1, activation=None)

sess = tf.Session()
saver = tf.train.Saver()

cost = tf.reduce_mean(tf.square(tf.subtract(tf.reshape(model, (-1,)), tf.reshape(ph['labels'], (-1,)))))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess.run(tf.global_variables_initializer())

#early stopper parameter
min_loss = float('inf')
step = 0

# load data 
adj_train, feat_train, labels_train = load_data(args.dataset, "training", args.adj_type)
adj_val, feat_val, labels_val = load_data(args.dataset, "validation", args.adj_type)
adj_test, feat_test, labels_test = load_test(args.testset, args.adj_type)

rmse_training = []
rmse_validation = []
cal_time = []
cal_memory = []

for epoch in range(ite_epochs):
    time_start = time.perf_counter()
    # train data
    total_cost = 0
    total_num = 0
    idx = np.arange(0, len(labels_train))
    np.random.shuffle(idx)
    total_batch = int((len(labels_train)) / batch_size)
    if ((len(labels_train))%batch_size) != 0:
        total_batch = total_batch + 1
    for i in range(total_batch):
        adj_batch, feat_batch, labels_batch, _ = next_batch(idx[i*batch_size:(i+1)*batch_size], adj_train, feat_train, labels_train)
        feed_dict = {ph['x']: feat_batch, ph["labels"]: labels_batch, ph["dropout_rate"]: 0.5}
        for i in range(adj_num[args.adj_type]):
            feed_dict[adj_input[i]] = adj_batch[i]
        _, cost_val = sess.run([optimizer, cost], feed_dict=feed_dict)
        total_cost += (cost_val * (len(labels_batch)))
        total_num += len(labels_batch)
    print('Epoch:', '%04d' % (epoch+1), 'Avg. cost = ', '{:.3f}'.format(total_cost/total_num))
    rmse_training.append((total_cost/total_num)**0.5)

    # validation data
    total_cost = 0
    total_num = 0
    idx = np.arange(0, len(labels_val))
    np.random.shuffle(idx)
    total_batch = int((len(labels_val)) / batch_size)
    if ((len(labels_val))%batch_size) != 0:
        total_batch = total_batch + 1
    for i in range(total_batch):
        adj_batch, feat_batch, labels_batch, _ = next_batch(idx[i*batch_size:(i+1)*batch_size], adj_val, feat_val, labels_val)
        feed_dict = {ph['x']: feat_batch, ph["labels"]: labels_batch, ph["dropout_rate"]: 0.0}
        for i in range(adj_num[args.adj_type]):
            feed_dict[adj_input[i]] = adj_batch[i]
        cost_val = sess.run(cost, feed_dict=feed_dict)
        total_cost += (cost_val * (len(labels_batch)))
        total_num += len(labels_batch)
    print("Validation cost:", '{:.3f}'.format(total_cost/total_num))
    rmse_validation.append((total_cost/total_num)**0.5)

    # check time
    time_elapsed = (time.perf_counter() - time_start)
    cal_time.append(time_elapsed)

    #early stooping
    compared = (total_cost/total_batch)>min_loss
    if compared:
        step += 1
        if step > patience:
            print("early stopping")
            break
    else:
        step = 0
        min_loss = (total_cost/total_batch)
        checkpoint = saver.save(sess, "%s/%s-%s-%s"%(args.output, 'model', args.adj_type, args.testset))

rmse_file = "%s/%s-%s-%s-rmse.csv" %(args.output, 'model', args.adj_type, args.testset)
rmses = pd.DataFrame(data={'training': rmse_training[:], 'validation': rmse_validation[:]})
rmses.to_csv(rmse_file, index=False)
cals = pd.DataFrame(data={'time': cal_time[:]})
cals.to_csv("%s/%s-%s-%s-time.csv" %(args.output, 'model', args.adj_type, args.testset), index=False)

predictions = []
rmse = {}

sess = tf.Session()
saver.restore(sess, os.path.abspath(checkpoint))
save_name = "%s/%s-%s-%s-best" %(args.output, 'model', args.adj_type, args.testset)
saver.save(sess, save_name)

def make_prediction(total_batch, adj, feat, labels):
    idx = np.arange(0, len(labels))
    pred, real, pdb_id = [], [], []
    for i in range(total_batch):
        adj_batch, feat_batch, labels_batch, pdbid_batch = next_batch(idx[i*batch_size:(i+1)*batch_size], adj, feat, labels)
        feed_dict = {ph['x']: feat_batch, ph["labels"]: labels_batch, ph["dropout_rate"]: 0.0}
        for i in range(adj_num[args.adj_type]):
            feed_dict[adj_input[i]] = adj_batch[i]
        pred.extend(sess.run(model, feed_dict=feed_dict).flatten().tolist())
        real.extend(labels_batch.tolist())
        pdb_id.extend(pdbid_batch.tolist())
    return pred, real, pdb_id

predictions = []
# prediction of training data
total_batch = int((len(labels_train)) / batch_size)
if ((len(labels_train))%batch_size) != 0:
    total_batch = total_batch + 1
pred, real, pdb_id = make_prediction(total_batch, adj_train, feat_train, labels_train)
predictions.append(pd.DataFrame(data={'pdbid': np.array(pdb_id)[:], 'real': np.array(real)[:], 'predicted': np.array(pred)[:], 'set': "training"}))

# prediction of validation data
total_batch = int((len(labels_val)) / batch_size)
if ((len(labels_val))%batch_size) != 0:
    total_batch = total_batch + 1
pred, real, pdb_id = make_prediction(total_batch, adj_val, feat_val, labels_val)
predictions.append(pd.DataFrame(data={'pdbid': np.array(pdb_id)[:], 'real': np.array(real)[:], 'predicted': np.array(pred)[:], 'set': "validation"}))

# prediction of test data
total_batch = int((len(labels_test)) / batch_size)
if ((len(labels_test))%batch_size) != 0:
    total_batch = total_batch + 1
pred, real, pdb_id = make_prediction(total_batch, adj_test, feat_test, labels_test)
predictions.append(pd.DataFrame(data={'pdbid': np.array(pdb_id)[:], 'real': np.array(real)[:], 'predicted': np.array(pred)[:], 'set': "test"}))

predictions = pd.concat(predictions, ignore_index=True)
predictions.to_csv("%s/%s-%s-%s-predictions.csv" %(args.output, 'model', args.adj_type, args.testset), index=False)
