import os, sklearn, scprep, magic, tqdm
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import scipy             as sp

from scipy.sparse            import csr_matrix
from scipy.stats             import pearsonr
from scprep.filter           import filter_rare_genes, filter_library_size
from scprep.normalize        import library_size_normalize
from scprep.transform        import log
from magic                   import MAGIC
from tqdm                    import tqdm
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from sklearn.cluster         import SpectralClustering
from sklearn.metrics         import silhouette_score

# from bblocks.py
from bblocks                 import bayesian_blocks


class dataset:

    def __init__(self, name):

        self.name                  = name
        self.raw_counts            = None
        self.normalized            = None
        self.imputed               = None
        self.binned                = None
        self.correlated            = None
        self.pseudotimes           = None
        self.clusters              = None
        self.gene_similarities     = None
        self.clustered_gene_names  = None
        self.silhouette_scores     = None
        self.gene_cluster_n_vals   = None
        self.gene_module_labels    = None
        self.n_gene_modules        = None
        self.correlations          = None

#         self.pca_embedding = None
#         self.umap_embedding = None
#         self.tsne_embedding = None


    def raw_counts_from_sparse_matrix(
        self, cell_names, gene_names, data, indices, indptr, shape, dtype):
        """
        parameters:
        * cell_names: 1D array-like of cell barcodes or identifiers
        * gene_names: 1D array-like of gene names or identifiers
        * data, indices, indptr: column indices for row i are stored in
          indices[indptr[i]:indptr[i+1]] and their corresponding values are
          stored in data[indptr[i]:indptr[i+1]]
        * shape: tuple, shape of full counts matrix
        * dtype: python data type of sparse matrix

        attributes:
        * dataset.raw_counts: pd.DataFrame, cells x genes raw counts matrix

        """

        self.raw_counts = pd.DataFrame(
                            data=csr_matrix(
                                (data,indices,indptr),
                                shape=shape,
                                dtype=dtype).toarray(),
                            index=cell_names,
                            columns=gene_names)


    def preprocess_raw_counts(self, library_size_cutoff=0):
        """
        preprocess the raw counts matrix by:
        1. filtering rare genes
        2. filtering cells with low total counts
        3. normalizing library sizes
        4. log scaling the normalized data
        (all performed via the scprep library)

        parameters:
        * library_size_cutoff: float or tuple of floats, smallest (and largest,
          if tuple) allowable library size to retain a cell, optional (default 0)

        attributes:
        * dataset.normalized: pd.DataFrame, filtered, normalized and scaled
          cells x genes matrix

        """

        self.normalized = filter_rare_genes(
                                self.raw_counts)
        self.normalized = filter_library_size(self.normalized,
                                    cutoff=library_size_cutoff)
        self.normalized = library_size_normalize(
                                self.normalized)
        self.normalized = log(self.normalized)



    def impute_from_normalized(self, genes='all_genes'):
        """
        impute missing expression from normalized matrix
        (performed via MAGIC data diffusion)

        parameters:
        * genes: 1D array-like of gene names or identifiers to impute
          expression for, optional (default all)

        attributes:
        * dataset.imputed: pd.DataFrame, filtered cells x imputed genes matrix

        """

        magic_op = MAGIC(t='auto',
                    verbose=False,
                    random_state=0)

        self.imputed = magic_op.fit_transform(
                                self.normalized,
                                genes=genes)


    def bin_data(self, in_pt, pt_bin, normalize=True, data='raw_counts', genes=[]):
        """
        perform binning of indicated data and genes:
        * bin in pseudotime if in_pt == True
        * bin in expression using bayesian blocks

        parameters:
        * data: one of ['raw_counts' (default),'normalized','imputed'], optional
        * genes: 1D array-like of gene names or identifiers to bin, optional
          (default all)
        * in_pt: True/False, whether or not to bin data in pseudotime
        * pt_bin: float, bin width in pseudotime if in_pt == True
        * normalize: True/False, whether or not to scale expression to [0,1],
          optional (default True)

        attributes:
        * dataset.binned: pd.DataFrame, cells (binned in pt) x genes (binned in
          expression) matrix

        """

        if data == 'raw_counts':
            X = self.raw_counts
        elif data == 'normalized':
            X = self.normalized
        elif data == 'imputed':
            X = self.imputed

        if len(list(genes)) > 0:
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


    def find_gene_similarities(self, n_runs=5, save=False, fname=''):
        """
        this function compiles a pairwise gene similarities matrix
        from the binned data, where similarity is defined as the
        adjusted mutual information score (mean of n_runs).

        parameters:
        * n_runs: int, number of independent runs performed to find mean ami
          for each pairwise comparison, optional, (default 5)
        * save: True/False, whether or not to save the similarities matrix
          to a .npy file, optional (default False)
        * fname: string, path/filename for saved array if save == True,
          optional

        attributes:
        * dataset.gene_similarities: np array, binned genes x binned genes
          (mean of n runs)

        """

        # find pairwise similarities
        self.gene_similarities = np.ones((n_runs, self.binned.shape[1],
                                          self.binned.shape[1]))

        idxs = np.vstack(np.triu_indices(self.binned.shape[1])).T

        for i in tqdm(idxs):

            for j in np.arange(n_runs):

                self.gene_similarities[[[j],[j]],[[i[0]],[i[1]]],[[i[1]],[i[0]]]] = \
                    ami(self.binned.iloc[:,i[0]],self.binned.iloc[:,i[1]])

        self.gene_similarities = np.mean(self.gene_similarities,axis=0)
        self.gene_similarities = np.clip(self.gene_similarities,0,1)

        if save:
            with open(fname, 'wb') as f:
                np.save(f, self.gene_similarities)


    def cluster_genes(self, min_clusters=2, max_clusters=10, n_components=10):
        """
        this function performs spectral clustering over a range of
        n_clusters to assign genes to a gene module, choosing the
        optimal number of modules by maximizing the silhouette score.

        parameters:
        * min_clusters: int, minimum number of clusters
        * max_clusters: int, maximum number of clusters
        * n_components: int, number of components used in kmeans clustering
          of decomposed affinity matrix during spectral clustering

        attributes:
        * dataset.gene_module_labels: 1D array of labels assigning genes
          to a module
        * dataset.n_gene_modules: int, optimal number of modules from
          maximizing the silhouette score
        * dataset.silhouette_scores: 1D array containing silhouette scores
          from clustering at each n_clusters
        * dataset.gene_cluster_n_vals: 1D array of each n_clusters used
          for spectral clustering
        """

        max_clusters += 1

        self.clustered_gene_names = self.binned.columns.values
        self.gene_cluster_n_vals = np.arange(min_clusters,max_clusters)
        self.silhouette_scores = np.zeros(max_clusters-min_clusters)
        self.gene_module_labels = np.zeros((self.gene_similarities.shape[0],
                                            self.silhouette_scores.shape[0]))

        for n in np.arange(min_clusters,max_clusters):

            labels = SpectralClustering(affinity='precomputed',
                                        n_components=n_components,
                                        assign_labels='kmeans',
                                        n_init=500, n_clusters=n,
                                        ).fit_predict(self.gene_similarities)


            # distances = 1 - ami
            score = silhouette_score(X = 1 - self.gene_similarities,
                                    metric='precomputed', labels=labels)

            self.gene_module_labels[:,n-min_clusters] = labels
            self.silhouette_scores[n-min_clusters] = score

        self.n_gene_modules = np.argmax(self.silhouette_scores)
        self.gene_module_labels = self.gene_module_labels[:,self.n_gene_modules]
        self.n_gene_modules += 2


    def plot_silhouette_scores(self):
        """
        generate a plot to compare silhouette scores across n_clusters

        """

        plt.figure()
        plt.plot(self.gene_cluster_n_vals,
                 self.silhouette_scores,
                 c='tab:blue', marker='o')

        ax = plt.gca()
        l,r = ax.get_xlim()
        b,t = ax.get_ylim()

        plt.vlines(self.n_gene_modules, b, t,
                   color='k', linestyle='dotted')

        ax.set_xlabel('Clusters')
        ax.set_ylabel('Silhouette Score')
        plt.xticks(self.gene_cluster_n_vals[::2])
        ax.set_ylim(b,t)  # reset after vline
        ax.set_aspect(abs((r-l)/(b-t))*0.33)
        plt.tight_layout()


    def correlate_genes_in_modules(self):
        """
        orient (flip) trajectories so that all genes within a module
        positively correlate with each other (assumes normalization
        of expression to between [0,1] during binning)

        attributes:
        * dataset.correlated: pd.DataFrame, same structure as dataset.binned
        * dataset.correlations: 1D binary array corresponding to binned genes,
          indicating whether gene is positively or negatively correlated with
          module

        """

        self.correlated = self.binned.copy()
        self.correlations = np.ones(self.gene_module_labels.shape)

        for i in self.correlated.columns:

            gene_module = self.gene_module_labels[np.where(
                                self.clustered_gene_names==i)][0]

            ref_gene = self.clustered_gene_names[np.where(
                                self.gene_module_labels==gene_module)][0]

            corr = pearsonr(self.correlated[i],self.binned[ref_gene])

            if corr[0] < 0:

                self.correlated[i] = -self.correlated[i] + 1

                self.correlations[np.where(self.clustered_gene_names==i)] = -1
