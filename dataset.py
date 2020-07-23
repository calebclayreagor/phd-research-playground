import os
import math
import tqdm
import sklearn
import scprep
import magic
import matplotlib        as mpl
import pandas            as pd
import numpy             as np
import scipy             as sp
import gseapy            as gp

from matplotlib              import pyplot as plt
from matplotlib.ticker       import MaxNLocator
from scipy.sparse            import csr_matrix
from scipy.interpolate       import UnivariateSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scprep.filter           import filter_rare_genes
from scprep.filter           import filter_library_size
from scprep.normalize        import library_size_normalize
from scprep.transform        import log
from magic                   import MAGIC
from tqdm                    import tqdm
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from sklearn.cluster         import SpectralClustering
from sklearn.metrics         import silhouette_score
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE
from gseapy.parser           import Biomart
from cycler                  import cycler

# from bblocks.py
from bblocks                 import bayesian_blocks

# R interface via rpy2
from rpy2 import robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects import Formula
robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
base = importr("base")
dollar = base.__dict__["$"]
deseq = importr('DESeq2')


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
        self.enrichment_axes       = None
        self.pca_embedding         = None
#        self.umap_embedding       = None
        self.tsne_embedding        = None
        self.embedding_axes        = None
        self.deg                   = None



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
                            dtype=np.int64,
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
        impute missing expression from normalized data matrix
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
        using the previously binned data, where similarity is defined
        as the adjusted mutual information score (mean of n_runs).

        parameters:
        * n_runs: int, number of independent runs performed to find mean ami
          for each pairwise comparison, optional (default 5)

        attributes:
        * dataset.gene_similarities: pd.DataFrame, binned genes x binned genes
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
        self.gene_similarities = pd.DataFrame(self.gene_similarities,
                                              index=self.binned.columns,
                                              columns=self.binned.columns)



    def cluster_genes(
        self, min_clusters=2, max_clusters=20, n_components=10, plot_silhouette=True):
        """
        this function performs spectral clustering (using the previously computed
        similarity matrix) over a range of n_clusters to assign genes to a gene
        module, choosing the optimal number of modules by maximizing the silhouette score.

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
        (in plot_data) using predefined criteria.

        parameters:
        * method: string, method to use for projecting genes to 1d,
          optional:
          * 'max': pseudotime of trajectory maximum expression (default)
          * 'median': pseudotime of trajectory median expression

        attributes:
        * dataset.genes_1d: 1d array of gene 1d projections

        """

        self.genes_1d = pd.DataFrame(index=self.plot_data.columns, columns=['pt'])

        if method == 'max':
            self.genes_1d['pt'] = self.pseudotimes[self.plot_data.idxmax()].values

        elif method == 'median':
            self.genes_1d['pt'] = self.pseudotimes[abs(self.plot_data-0.5).idxmin()].values



    def pathway_ea_in_pt(self, pathways, pt_bin=0.1, plot=True):
        """
        this function uses the previously computed gene trajectory pseudotime
        projections to perform GO enrichment analysis on KEGG pathways using
        bins of specified width.

        paramters:
        * pathways: list of strings, names of KEGG pathways for analysis
        * pt_bin: float, bin width in pseudotime, optional (default 0.1)
        * plot: True/False, generate plot for results, optional (default True)

        attributes:
        * dataset.pathway_ea: pd.DataFrame, pathways x bins matrix of combined
          scores
        * dataset.enrichment_axes: plot axis

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
            gene_ids = self.clustered_gene_names[np.where((self.genes_1d['pt']>i[0])&
                                                          (self.genes_1d['pt']<i[1]))]
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

            self.enrichment_axes = ax



    def embed_pca(self, data='raw_counts', genes=[], n_components=20):
        """
        dimensionsionality reduction using principal components analysis

        parameters:
        * data: one of ['raw_counts' (default),'normalized','imputed'], optional
        * genes: 1D array-like of gene names or identifiers to bin, optional
          (default all)
        * n_components: int, number of components to keep, optional (default 20)

        attributes:
        * dataset.pca_embedding: pd.DataFrame, cells x components embedding

        """

        if data == 'raw_counts':
            X = self.raw_counts
        elif data == 'normalized':
            X = self.normalized
        elif data == 'imputed':
            X = self.imputed

        if len(list(genes)) > 0:
            X = X[[g for g in genes if g in X.columns]]

        self.pca_embedding = pd.DataFrame(PCA(n_components=n_components).fit_transform(X), index=X.index)



    def embed_tsne(self, data='pca', genes=[]):
        """
        dimensionsionality reduction using t-stochastic neighbor embedding

        parameters:
        * data: one of ['pca' (default),'raw_counts','normalized','imputed'],
          optional
        * genes: 1D array-like of gene names or identifiers to bin, optional
          (default all)

        attributes:
        * dataset.tsne_embedding: pd.DataFrame, cells x components embedding

        """

        if data == 'pca':
            X = self.pca_embedding
        elif data == 'raw_counts':
            X = self.raw_counts
        elif data == 'normalized':
            X = self.normalized
        elif data == 'imputed':
            X = self.imputed

        if len(list(genes)) > 0:
            X = X[[g for g in genes if g in X.columns]]

        self.tsne_embedding = pd.DataFrame(TSNE().fit_transform(X), index=X.index)



    def plot_embedding(self, data, plot_clusters=[], labels=[], ar=1, legend=True):
        """
        plot designated clusters in reduced dimensions

        parameters:
        * data: one of ['pca','tsne'], which embedding to plot
        * plot_clusters: list of clusters to plot, optional (default all)
        * labels: list of labels for plotted clusters, optional (one label
          for each cluster in plot_clusters, else one label for every cluster)
        * ar: aspect ratio, optional (default 1)
        * legend: True/False, show legend, optional (default True)

        attributes:
        * dataset.plot_axes: plot axis

        """

        if data == 'pca':
            X = self.pca_embedding
            ax_labels = ['PC1','PC2']
        elif data == 'tsne':
            X = self.tsne_embedding
            ax_labels = ['tSNE1','tSNE2']

        if len(list(plot_clusters)) == 0:
            plot_clusters = self.clusters.unique()

        if len(list(labels)) == 0:
            labels = plot_clusters

        plt.figure()
        cmap = plt.cm.jet
        N = len(plot_clusters)
        c = cycler('color', cmap(np.linspace(0,1,N)))
        plt.rcParams["axes.prop_cycle"] = c

        for c in plot_clusters:
            x = np.where(self.clusters==c)[0]
            plt.scatter(X.iloc[x,0], X.iloc[x,1], edgecolor='k')

        ax = plt.gca()
        l,r = ax.get_xlim()
        b,t = ax.get_ylim()
        ax.axis('off')

        ax.set_aspect(abs((r-l)/(b-t))*ar)

        if legend:
            ax.legend(labels, frameon=False, markerscale=1.25)

        if ar > 0.5:
            ax_ar = 0.5
        else:
            ax_ar = 1

        plt.annotate(s='', xy=(0,ax_ar*0.4), xytext=(ax_ar*0.4*ar,0),
                     xycoords='axes fraction', arrowprops=dict(arrowstyle='<->',
                     connectionstyle='angle, rad=0, angleA=0, angleB=-90',
                     color='k'))

        plt.text(s=ax_labels[0], x=ax_ar*0.18*ar, y=-0.025/ar,
                 va='center', ha='center', color='k', transform= ax.transAxes)

        plt.text(s=ax_labels[1], x=-0.025, y=ax_ar*0.18,
                 rotation=90, va='center', ha='center',
                 color='k', transform=ax.transAxes)

        self.embedding_axes = ax



    def diff_exp2(self, method='deseq2', clusters=[]):
        """
        two group differential expression analysis

        ...

        """

        if method == 'deseq2':

            x = self.clusters.isin(clusters)

            dds = deseq.DESeqDataSetFromMatrix(design=Formula('~ polarity'),
                    countData=self.raw_counts.loc[x,:].values.T, colData=
                    robjects.DataFrame({'polarity': robjects.StrVector(
                    self.clusters[x].astype(str).values).factor()}))

            dds = deseq.DESeq(dds)
            ds_res = deseq.results(dds)

            self.deg = pd.DataFrame(index=self.raw_counts.columns)
            self.deg['P.Value'] = dollar(ds_res,'pvalue')
            self.deg['adj.P.Val'] = dollar(ds_res,'padj')
            self.deg['logFC'] = dollar(ds_res,'log2FoldChange')
            self.deg.sort_values('logFC', ascending=False, inplace=True)
            self.deg.dropna(how='all', inplace=True)



    def plot_violin(
            self, clusters=[], cluster_labels=[], gene=[], gene_label=[], ar=1):
        """
        ...

        """

        plt.figure(dpi=100)
        ax = plt.gca()
        l,r = ax.get_xlim()
        b,t = ax.get_ylim()

        data = [self.raw_counts.loc[self.clusters==clusters[0], gene],
                self.raw_counts.loc[self.clusters==clusters[1], gene]]

        v = ax.violinplot(data, showmeans=False,
                          showmedians=False,
                          showextrema=False)

        ax.scatter(np.random.normal(1, 0.1, data[0].shape[0]),
                   data[0] + np.random.uniform(-0.05*(t-b), 0.05*(t-b), data[0].shape[0]), c='k')

        ax.scatter(np.random.normal(2, 0.1, data[1].shape[0]),
                   data[1] + np.random.uniform(-0.05*(t-b), 0.05*(t-b), data[1].shape[0]), c='k')

        yl,yh = ax.get_ylim()
        ax.plot([1,2], [yh,yh], c='k')
        ax.text(1.5, yh+0.05*(yh-yl), ha='center', va='center',
                s='p='+str(round(self.deg.loc[gene,'P.Value'],5)))
        ax.set_ylim(yl,yh+0.1*(yh-yl))
        ax.title.set_text(gene_label)
        plt.xticks([1,2])
        ax.set_xticklabels(cluster_labels)
        ax.set_aspect(abs((r-l)/(b-t))*ar)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('transcripts')






#    def gaussian_mixture


#
