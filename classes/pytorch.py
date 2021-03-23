import os
import glob
import shutil
import random
import torch
import pathlib
import itertools

import numpy             as np
import pandas            as pd
import pytorch_lightning as pl

from tqdm.notebook    import tqdm
from torch.nn         import Conv2d
from torch.nn         import Linear
from torch.nn         import Sigmoid
from torch.nn         import functional as F
from torch.nn.init    import xavier_normal_
from torch.optim      import Adam

from pytorch_lightning.metrics.functional.classification import precision_recall_curve, auc




class Dataset(torch.utils.data.Dataset):
    """Dataset for generating/loading batches"""

    def __init__(self, root_dir, rel_path, batchSize=None,
                 overwrite=False, wShuffle=0.02, minCells=40):

        self.root_dir = root_dir
        self.rel_path = rel_path
        self.batch_size = batchSize
        self.overwrite = overwrite
        self.pt_shuffle = wShuffle
        self.min_cells = minCells

        self.sce_fnames = sorted(pathlib.Path(self.root_dir).glob(self.rel_path))

        self.X_fnames = [None] * len(self.sce_fnames)
        self.y_fnames = [None] * len(self.sce_fnames)

        print('Loading batches from path...')
        for sce_fname in tqdm(self.sce_fnames):
            self.generate_batches(str(sce_fname))



    def __len__(self):
        """Total number of batches"""
        return len(self.X_fnames)



    def __getitem__(self, idx):
        """Load a given batch"""
        new_batch, idx_ = True, idx
        np.random.seed(self.seed_from_string(self.X_fnames[idx]))
        while new_batch:
            X = np.load(self.X_fnames[idx_], allow_pickle=True)
            y = np.load(self.y_fnames[idx_], allow_pickle=True)
            if X.shape[3] > self.min_cells: new_batch = False
            idx_ = np.random.choice(np.arange(len(self.X_fnames)))
        return X, y



    def seed_from_string(self, s):
        n = int.from_bytes(s.encode(), 'little')
        return sum([int(x) for x in str(n)])



    def shuffle_pt(self, pt, seed):
        """Kernelized swapper"""
        np.random.seed(seed)
        pt = pt.copy().values
        for i in np.arange(pt.size):
            j = np.random.normal(loc=0, scale=self.pt_shuffle*pt.size)
            i_ = int(round(np.clip(i+j, 0, pt.size-1)))
            pt[[i,i_]] = pt[[i_,i]]
        return pt



    def generate_batches(self, sce_fname):
        """Generate batch(es) as .npy file(s) from sce"""

        def grouper(iterable, m, fillvalue=None):
            args = [iter(iterable)] * m
            return itertools.zip_longest(*args, fillvalue=fillvalue)

        # generate batches (>=1) for each trajectory
        sce_folder = '/'.join(sce_fname.split('/')[:-1])
        pt = pd.read_csv(f"{sce_folder}/PseudoTime.csv", index_col=0)

        n_clusters, sce, ref = pt.shape[1], None, None

        # outer loop: trajectories
        for k in range(n_clusters):
            traj_folder = f"{sce_folder}/traj{k+1}/"

            # load previously generated batches for given trajectory
            if os.path.isdir(traj_folder) and self.overwrite==False:
                # save batch filenames for __len__ and __getitem__
                for file in sorted(glob.glob(f'{traj_folder}/*.npy')):
                    if file.split('/')[-1][0]=='X':
                        idx = np.where(np.array(self.X_fnames) == None)[0]
                        if idx.size > 0:
                            self.X_fnames[idx[0]] = file
                        else:
                            self.X_fnames.extend([file])
                    elif file.split('/')[-1][0]=='y':
                        idx = np.where(np.array(self.y_fnames) == None)[0]
                        if idx.size > 0:
                            self.y_fnames[idx[0]] = file
                        else:
                            self.y_fnames.extend([file])
            else:
                if os.path.isdir(traj_folder):
                    shutil.rmtree(traj_folder)
                os.mkdir(traj_folder)

                if sce is not None: pass
                else:
                    sce = pd.read_csv(sce_fname, index_col=0).T

                    # sort expression in experiment using slingshot pseudotime
                    sce = sce.loc[pt.sum(axis=1).sort_values().index,:].copy()

                    # shuffle pseudotime (if synthetic experiment)
                    if sce_folder.split('/')[-4]!='experimental':
                        seed = self.seed_from_string(traj_folder)
                        sce = sce.loc[self.shuffle_pt(sce.index, seed),:].copy()

                    # generate list of tuples containing all possible gene pairs
                    gpairs = [g for g in itertools.product(sce.columns, repeat=2)]
                    seed = self.seed_from_string(traj_folder)
                    random.seed(seed); random.shuffle(gpairs)

                    n, n_cells = len(gpairs), sce.shape[0]

                if ref is not None: pass
                else:
                    ref = pd.read_csv(f"{sce_folder}/refNetwork.csv").values
                    ref_1d = np.array(["%s %s" % x for x in list(zip(ref[:,0], ref[:,1]))])

                if self.batch_size is not None:
                    gpairs_batched = [list(x) for x in grouper(gpairs, self.batch_size)]
                    gpairs_batched = [list(filter(None, x)) for x in gpairs_batched]
                else: gpairs_batched = [gpairs]

                if sce_folder.split('/')[-4]=='experimental':
                    print(f"Generating batches for {'/'.join(sce_folder.split('/')[-2:])}")

                # inner loop: batches of gene pairs
                for j in range(len(gpairs_batched)):
                    X_fname = f"{traj_folder}/X_batch{j}_size{len(gpairs_batched[j])}.npy"
                    y_fname = f"{traj_folder}/y_batch{j}_size{len(gpairs_batched[j])}.npy"

                    gpairs_list = list(itertools.chain(*gpairs_batched[j]))

                    if self.batch_size is None or j==len(gpairs_batched)-1:
                        sce_list = np.array_split(sce[gpairs_list].values, len(gpairs_batched[j]), axis=1)
                    else:
                        sce_list = np.array_split(sce[gpairs_list].values, self.batch_size, axis=1)

                    sce_list = [g_sce.reshape(1,1,2,n_cells) for g_sce in sce_list]
                    X_batch = np.concatenate(sce_list, axis=0).astype(np.float32)

                    gpairs_batched_1d = np.array(["%s %s" % x for x in gpairs_batched[j]])
                    y_batch = np.in1d(gpairs_batched_1d, ref_1d).reshape(X_batch.shape[0],1)

                    traj_idx = np.where(~pt.iloc[:,k].isnull())[0]
                    np.save(X_fname, X_batch[...,traj_idx], allow_pickle=True)
                    np.save(y_fname, y_batch.astype(np.float32), allow_pickle=True)

                    # save batch filenames for __len__ and __getitem__
                    idx = np.where(np.array(self.X_fnames) == None)[0]
                    if idx.size > 0:
                        self.X_fnames[idx[0]] = X_fname
                        self.y_fnames[idx[0]] = y_fname
                    else:
                        self.X_fnames.extend([X_fname])
                        self.y_fnames.extend([y_fname])





class Classifier(pl.LightningModule):
    """CNN for binary classification of gene trajectory pairs"""

    # ISSUES TO DEAL WITH:
    # * Multiple validation sets?
    # * Need more data normalization?
    # * How to deal with imbalanced data?
    # * ... and different batch sizes?
    # * Implement learning rate scheduler
    # * Need to use GPU... setup?


    def __init__(self, lr):
        super().__init__()

        self.lr = lr

        # convolutional layers
        self.conv1 = Conv2d(1, 32, padding=(1,0), kernel_size=(3,8))
        self.conv2 = Conv2d(32, 32, padding=(1,0), kernel_size=(3,8))
        self.conv3 = Conv2d(32, 32, padding=(1,0), kernel_size=(3,8))
        self.conv4 = Conv2d(32, 32, kernel_size=(2,8))

        # dense layers
        self.dense1  = Linear(32, 32)
        self.dense2  = Linear(32, 32)
        self.dense3  = Linear(32, 32)
        self.dense4  = Linear(32, 32)

        # output layer
        self.output = Linear(32, 1)

        # weight initializations
        xavier_normal_(self.conv1.weight)
        xavier_normal_(self.conv2.weight)
        xavier_normal_(self.conv3.weight)
        xavier_normal_(self.conv4.weight)
        xavier_normal_(self.dense1.weight)
        xavier_normal_(self.dense2.weight)
        xavier_normal_(self.dense3.weight)
        xavier_normal_(self.dense4.weight)



    def forward(self, x):

        # convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.squeeze(F.max_pool1d(torch.squeeze(x),
                            kernel_size=x.size()[-1]))

        # dense layers
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))

        # output, activation
        x = Sigmoid(self.output(x))

        return x



    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer



    def binary_cross_entropy_loss(self, pred, labels):
        return F.binary_cross_entropy(pred, labels)



    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        pred = self.forward(X)
        loss = self.binary_cross_entropy_loss(pred, y)
        precision, recall, _ = precision_recall_curve(pred, y)
        self.log('train_auprc', auc(recall, precision))
        self.log('train_loss', loss)
        return loss



    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        pred = self.forward(X)
        loss = self.binary_cross_entropy_loss(pred, y)
        precision, recall, _ = precision_recall_curve(pred, y)
        self.log('val_auprc', auc(recall, precision))
        self.log('val_loss', loss)





#
