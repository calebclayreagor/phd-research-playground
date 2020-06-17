import os, sklearn, scprep, magic, tqdm
import matplotlib        as mpl
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import scipy             as sp
import gseapy            as gp

from scipy.sparse            import csr_matrix
from scipy.interpolate       import UnivariateSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scprep.filter           import filter_rare_genes, filter_library_size
from scprep.normalize        import library_size_normalize
from scprep.transform        import log
from magic                   import MAGIC
from tqdm                    import tqdm
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from sklearn.cluster         import SpectralClustering
from sklearn.metrics         import silhouette_score
from gseapy.parser           import Biomart

# from bblocks.py
from bblocks                 import bayesian_blocks


class dataset:

    def __init__(self, name):

        self.name                  = name
        self.raw_counts            = None
        self.normalized            = None
        self.imputed               = None
        self.binned                = None
        self.plot_data             = None
        self.pseudotimes           = None
        self.clusters              = None
        self.gene_similarities     = None
        self.clustered_gene_names  = None
        self.silhouette_scores     = None
        self.gene_cluster_n_vals   = None
        self.gene_module_labels    = None
        self.n_gene_modules        = None
        self.module_axes           = None
        self.genes_1d              = None
        self.pathway_ea            = None
#         self.pca_embedding = None
#         self.umap_embedding = None
#         self.tsne_embedding = None



    def raw_counts_from_sparse_matrix(
        self, cell_names, gene_names, data, indices, indptr, shape, dtype):
        """
        parameters:
        * cell_names: 1D array-like of cell barcodes or identifiers
        * gene_names: 1D array-like of ensembl gene identifiers
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



    def bin_data(self, in_pt, pt_bin, data='raw_counts', genes=[]):
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

            # in pseudotime (more pythonic way?)
            self.binned = pd.DataFrame(columns=X.columns)

            bins = np.vstack((np.arange(self.pseudotimes.min(),
                              self.pseudotimes.max()-pt_bin, pt_bin),
                              np.arange(pt_bin, self.pseudotimes.max(),
                              pt_bin))).T

            for i in bins:

                idxs = self.pseudotimes[(self.pseudotimes > i[0])&
                                        (self.pseudotimes < i[1])].index

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



    def find_gene_similarities(self, n_runs=5):
        """
        this function compiles a pairwise gene similarities matrix
        from the binned data, where similarity is defined as the
        adjusted mutual information score (mean of n_runs).

        parameters:
        * n_runs: int, number of independent runs performed to find mean ami
          for each pairwise comparison, optional (default 5)

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



    def cluster_genes(
        self, min_clusters=2, max_clusters=20, n_components=10, plot_silhouette=True):
        """
        this function performs spectral clustering over a range of n_clusters
        to assign genes to a gene module, choosing the optimal number of
        modules by maximizing the silhouette score.

        parameters:
        * min_clusters: int, minimum number of clusters, optional (default 2)
        * max_clusters: int, maximum number of clusters, optional (default 10)
        * n_components: int, number of components used in kmeans clustering
          of decomposed affinity matrix during spectral clustering, optional
          (default 20)
        * plot_silhouette: True/False, plot the silhouette score vs n_clusters,
          optional (default True)

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

        if plot_silhouette:

            plt.figure()
            plt.plot(self.gene_cluster_n_vals,
                    self.silhouette_scores,
                    c='tab:blue', marker='.')

            ax = plt.gca()
            l,r = ax.get_xlim()
            b,t = ax.get_ylim()

            plt.vlines(self.n_gene_modules, b, t,
                       color='k', linestyle='dotted')

            ax.set_xlabel('clusters')
            ax.set_ylabel('silhouette score')
            plt.xticks(self.gene_cluster_n_vals[::2])
            ax.set_ylim(b,t)  # reset after vline
            ax.set_aspect(abs((r-l)/(b-t))*0.33)
            plt.tight_layout()



    def plot_modules(self, smoothing=0.05, data='normalized'):
        """
        prepare the gene module data and plot the results:
        1. orient (flip) trajectories so that all genes within a
           module positively correlate with each other
        2. smooth trajectories using a sliding window
        3. scale expression values to between 0 and 1
        4. take the average trajectory of genes in a module and
           fit a univariate spline to the average
        5. find the standard deviation at several points along
           the average trajectory
        6. plot the fitted spline and errors to visualize the
           trajectory of the gene module

        parameters:
        * data: one of ['raw_counts','normalized' (default),
          'imputed'], optional
        * smoothing: float, the width (in pseudotime units) of
          the sliding window used for smoothing gene trajectories,
          optional (default 0.05)

        attributes:
        * dataset.plot_data: pd.DataFrame, cells x binned genes,
          containing smoothed/scaled expression values
        * dataset.module_axes: list, contains plots' axes objects

        """

        if data == 'raw_counts':
            X = self.raw_counts
        elif data == 'normalized':
            X = self.normalized
        elif data == 'imputed':
            X = self.imputed

        self.plot_data = X[self.binned.columns].copy()

        for i in sorted(np.unique(self.gene_module_labels)):

            genes = np.where(self.gene_module_labels==i)[0]
            cells = np.arange(self.plot_data.shape[0])

            # correlate trajectories
            pcorr = np.corrcoef(self.plot_data.iloc[:,genes].T)
            for j in np.arange(genes.shape[0]):
                if pcorr[0,j] < 0:
                    self.plot_data.iloc[:,genes[j]] *= -1

            # smooth trajectories
            for cell in cells:
                w = [x for x in cells if abs(self.pseudotimes.iloc[x] -
                     self.pseudotimes.iloc[cell]) < smoothing/2]
                self.plot_data.iloc[cell,genes] = self.plot_data.iloc[w,genes].mean(axis=0)

            # scale trajectories
            self.plot_data.iloc[:,genes] = (self.plot_data.iloc[:,genes] -           \
                                            self.plot_data.iloc[:,genes].min())/     \
                                           (self.plot_data.iloc[:,genes].max() -     \
                                            self.plot_data.iloc[:,genes].min())

            # fit spline
            x = self.pseudotimes
            y = self.plot_data.iloc[:,genes].mean(axis=1)
            x_spline = np.linspace(x.min(), x.max(), 200)
            spl = UnivariateSpline(x, y)
            y_spline = spl(x_spline)

            # find std at several points along trajectory
            stds = self.plot_data.iloc[:,genes].std(axis=1)
            x_std = np.linspace(self.pseudotimes.min(), self.pseudotimes.max(), 5)
            y_ = np.array([y_spline[np.argmin(abs(x_spline-x))] for x in x_std])
            y_std = np.array([stds.loc[abs(self.pseudotimes-x).idxmin()] for x in x_std])

            # plot
            plt.figure()

            plt.plot(x_spline, y_spline, c='k')
            plt.scatter(x_std, y_+y_std, c='r', marker='.')
            plt.scatter(x_std, y_+y_std, c='r', marker='_')
            plt.scatter(x_std, y_-y_std, c='r', marker='.')
            plt.scatter(x_std, y_-y_std, c='r', marker='_')

            ax = plt.gca()
            l,r = ax.get_xlim()
            b,t = ax.get_ylim()

            ax.set_xlabel('pseudotime')
            ax.set_ylabel('expression (AU)')
            plt.title('module'+str(int(i)))
            ax.set_aspect(abs((r-l)/(b-t))*0.25)
            plt.tight_layout()

            if i == 0:
                self.module_axes = [ax]
            else:
                self.module_axes.append(ax)



    def order_genes_pt(self, method='max'):
        """
        this function finds a 1d projection for each gene trajectory
        using predefined criteria (described below).

        parameters:
        * method: string, method to use for projecting genes to 1d,
          optional:
          * 'max': pseudotime of trajectory maximum expression (default)
          * 'median': pseudotime of trajectory median expression

        attributes:
        * dataset.genes_1d: 1d array of gene 1d projections

        """

        if method == 'max':
            self.genes_1d = self.pseudotimes[self.plot_data.idxmax()]

        elif method == 'median':
            self.genes_1d = self.pseudotimes[abs(self.plot_data-0.5).idxmin()]



    def pathway_ea_in_pt(self, pathways, pt_bin=0.1, plot=True):
        """
        ...

        """

        bins = np.vstack((np.arange(self.pseudotimes.min(),
                          self.pseudotimes.max()-pt_bin, pt_bin),
                          np.arange(pt_bin, self.pseudotimes.max(),
                          pt_bin))).T

        self.pathway_ea = pd.DataFrame(0, index=pathways, columns=
                                       [np.mean(x) for x in bins])

        bm = Biomart()
        background = bm.query(dataset='drerio_gene_ensembl',
                              attributes=['external_gene_name'],
                              filters={'ensembl_gene_id':
                              list(self.clustered_gene_names)})

        for i in tqdm(bins):
            gene_ids = self.clustered_gene_names[np.where((self.genes_1d>i[0])&
                                                          (self.genes_1d<i[1]))]
            if gene_ids.shape[0]>0:
                gene_names = bm.query(dataset='drerio_gene_ensembl',
                                      attributes=['external_gene_name'],
                                      filters={'ensembl_gene_id':list(gene_ids)})
            else:
                gene_names = pd.DataFrame()

            # enrichment analysis
            if gene_names.shape[0]>0:

                enr = gp.enrichr(gene_list=list(gene_names['external_gene_name']),
                                 background=list(background['external_gene_name']),
                                 organism='Fish', gene_sets='KEGG_2019',
                                 outdir=None, cutoff=1, no_plot=True)
                # update results
                if enr.res2d['Term'].isin(pathways).any():
                    res = enr.res2d.loc[np.where(enr.res2d['Term'].isin(pathways))[0],
                                                            ['Term','Combined Score']]
                    res.set_index('Term', inplace=True)
                    self.pathway_ea.loc[res.index,np.mean(i)] = res['Combined Score']

        self.pathway_ea.clip(lower=0, inplace=True)

        if plot:
            plt.figure()
            im = plt.imshow(self.pathway_ea, norm=mpl.colors.LogNorm(),
                            cmap='binary', aspect='equal', origin='lower')
            ax = plt.gca()
            ax.yaxis.tick_right()
            ax.grid(True, color='k')
            ax.set_xlabel('pseudotime')
            ax.set_xticks(np.arange(self.pathway_ea.shape[1]+1)-0.5)
            ax.set_yticks(np.arange(self.pathway_ea.shape[0])-0.5)
            ax.set_xticklabels([round(x,2) for x in
                                np.arange(self.pseudotimes.min(),
                                self.pseudotimes.max(), pt_bin)])
            ax.set_yticklabels(self.pathway_ea.index, va='bottom')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('left', size='5%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('combined score')
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')

            plt.tight_layout()



### ADD COMMENTS FOR PATHWAY ENRICHMENT FUNCTION
