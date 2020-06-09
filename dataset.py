# custom class for storing and processing single cell datasets

import os, scprep, magic, tqdm
import pandas as pd
import numpy  as np
import scipy  as sp

from scipy.sparse            import csr_matrix
from scprep.filter           import filter_rare_genes, filter_library_size
from scprep.normalize        import library_size_normalize
from scprep.transform        import log
from magic                   import MAGIC
from tqdm                    import tqdm
from sklearn.metrics.cluster import adjusted_mutual_info_score              as ami

# custom function from file bblocks.py
from bblocks             import bayesian_blocks


class dataset:

    def __init__(self, name):

        self.name                  = name
        self.raw_counts            = None
        self.normalized            = None
        self.imputed               = None
        self.binned                = None
        self.pseudotimes           = None
        self.clusters              = None
        self.gene_similarities     = None
#        self.gene_clusters         = None

#         self.pca_embedding = None
#         self.umap_embedding = None
#         self.tsne_embedding = None


    def raw_counts_from_sparse_matrix(
        self, cell_names, gene_names, data,
        indices, indptr, shape, dtype):

        self.raw_counts = pd.DataFrame(
                            data=csr_matrix(
                                (data,indices,indptr),
                                shape=shape,
                                dtype=dtype).toarray(),
                            index=cell_names,
                            columns=gene_names)


    def scprep_preprocessing(self, cutoff):
        """
        preprocess raw counts with scprep:

        * filter rare genes
        * filter cells with low counts
        * normalize library sizes
        * log scale

        """

        self.normalized = filter_rare_genes(
                                self.raw_counts)
        self.normalized = filter_library_size(
                                self.normalized,
                                cutoff=cutoff)
        self.normalized = library_size_normalize(
                                self.normalized)
        self.normalized = log(self.normalized)



    def magic_imputation(self, genes):
        """
        impute missing expression from
        normalized data with MAGIC

        """

        magic_op = MAGIC(t='auto',
                    verbose=False,
                    random_state=0)

        self.imputed = magic_op.fit_transform(
                                self.normalized,
                                genes=genes)


    def binning(self, data, genes, in_pt, pt_bin, normalize):
        """
        perform binning of indicated data and genes

        * bin in pseudotime if in_pt == True
        * bin in expression using bayesian blocks

        """

        if data == 'raw_counts':
            X = self.raw_counts
        elif data == 'normalized':
            X = self.normalized
        elif data == 'imputed':
            X = self.imputed
        else:
            X = self.raw_counts

        X = X[[g for g in genes if g in X.columns]]


        if in_pt:

            # in pseudotime
            self.binned = pd.DataFrame(columns=X.columns)

            bins = np.vstack((np.arange(self.pseudotimes.min(), self.pseudotimes.max()-pt_bin, pt_bin),
                              np.arange(pt_bin, self.pseudotimes.max(), pt_bin))).T

            for i in bins:

                idxs = self.pseudotimes[(self.pseudotimes > i[0]) & (self.pseudotimes < i[1])].index

                if idxs.shape[0] > 0:

                    self.binned.loc[np.mean(i)] = X.loc[idxs].mean(axis=0)

        else:
            self.binned = X


        # in expression
        for i in self.binned.columns:

            self.binned[i] = pd.cut(self.binned[i],
                                    bayesian_blocks(
                                    self.binned[i].astype('float64')),
                                    include_lowest=True,
                                    labels=False)

        self.binned = self.binned.loc[:,self.binned.nunique(axis=0)>1]

        if normalize:
            self.binned /= self.binned.max(axis=0)


    def find_gene_similarities(self):
        """
        ADD DESCRIPTION
        """

        ### find gene pairwise affinities (similarities) ###
        self.gene_similarities = np.ones((self.binned.shape[1],
                                          self.binned.shape[1]))

        idxs = np.vstack(np.triu_indices(self.binned.shape[1])).T

        for i in tqdm(idxs):

            self.gene_similarities[[[i[0]],[i[1]]],
                                   [[i[1]],[i[0]]]] = ami(self.binned.iloc[:,i[0]],
                                                          self.binned.iloc[:,i[1]])

        self.gene_similarities = np.clip(self.gene_similarities,0,1)
