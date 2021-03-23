import os
import tqdm
import shutil
import warnings
import sklearn
import palettable
import magic

import matplotlib           as mpl
import numpy                as np
import scipy                as sp
import pandas               as pd
import networkx             as nx
import seaborn              as sns
import matplotlib.pyplot    as plt

from numpy.random                          import choice, gamma, normal
from matplotlib.colors                     import rgb_to_hsv
from matplotlib.cm                         import ScalarMappable
from mpl_toolkits.axes_grid1               import ImageGrid
from matplotlib.colorbar                   import Colorbar
from palettable.colorbrewer.diverging      import Spectral_10
from palettable.scientific.sequential      import Hawaii_5
from sklearn.manifold                      import TSNE
from magic                                 import MAGIC
from tqdm.notebook                         import tqdm
from IPython.display                       import Image
from IPython.display                       import Markdown

# custom script for bayesian Lasso with scaled weights on priors
from lasso                                 import inference


class synthdata:

    def __init__(self, name, directory, boolODE_directory, beeline_directory):

        self.name        = name
        self.directory   = f'{directory}/{name}'
        self.boolODE_dir = boolODE_directory
        self.beeline_dir = beeline_directory
        self.grn         = None
        self.grn_full    = None
        self.expression  = None
        self.t           = None
        self.imputed     = None
        self.tfbs        = None
        self.bee_auc     = None
        self.lasso_auc   = None

        # make results directory, if necessary
        if os.path.exists(self.directory): pass
        else: os.mkdir(self.directory)


    def generate_model(self, n_tfs, n_targets, selfs=0.5, frac=0.5,
                 strengths_min=0.8, strengths_max=1, k_strengths=2,
                 mindist=0.5, dpi=1000, load_prev=False, verbose=True):
        """
        generate a gene regulatory network that mimics HC development

        parameters:
        * n_tfs: int, number of transcription factors in the core tf cascade
        * n_targets: int, number of additional genes downstream of core tfs
        * selfs: float [0.0, 1.0], adjusts the number of self-loops after the
          toggle switch (optional)
        * frac: float [0.0, 1.0], percent of downstream genes inhibited by
          core tfs (optional)
        * k_strengths: float, parameter controling distribution of connection
          weights, with higher values giving higher weights (optional)
        * strengths_min and strengths_max: float, minimum and maximum values
          (respectively) for connection weights (optional, max=20)
        * mindist: float [0.0, 1.0], minimum separation between nodes in plot
        * load_prev: bool, whether or not to load existing model with name
        * verbose: bool, whether or not to print the parameters from prev model

        attributes:
        * synthdata.grn[_full]: tf [gene] x gene array, where tf_i regulates
          gene_j with strength i,j
        """

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        if load_prev:
            try:
                self.grn = pd.read_csv(f'{self.directory}/model.csv', index_col=0)

                self.grn_full = pd.DataFrame(np.pad(self.grn, ((0, self.grn.shape[1]-self.grn.shape[0]), (0,0))),
                                                         index = [f'g{i}' for i in np.arange(self.grn.shape[1])],
                                                         columns=[f'g{i}' for i in np.arange(self.grn.shape[1])])

                n_tfs, n_targets = self.grn.shape[0], self.grn.shape[1]-self.grn.shape[0]
                if verbose:
                    with open(f"{self.directory}/log.txt") as log:
                        params = [next(log) for x in range(7)]
                    print('Succesfully loaded previous model with the following parameters: ' + \
                                                             ', '.join(params).replace('\n',''))
            except:
                print("Couldn't find model file to load previous results.")



        else:
            # PART 1: SPECIFY GRN STRUCTURE
            adj_mat = np.zeros((n_tfs, n_tfs))

            # Simple regulation (SR): all tfs
            adj_mat[np.arange(n_tfs-1),
                    np.arange(1,n_tfs)] = 1.0

            # Toggle switch ("DNL"):
            #   from early -> late HC genes
            id = round((n_tfs/2)-1)
            id = np.clip(id, a_max=n_tfs, a_min=0)
            adj_mat[id-1, id] = -1.0
            adj_mat[id, id+1] = -1.0
            adj_mat[id+1, id] = -1.0

            # Toggle_0 -> all early HC tfs
            for i in range(id):
                adj_mat[id,i] = 1.0

            # Positive autoregulation (PAR):
            #   gene0 + random tfs after toggle
            k = round(selfs * ((n_tfs/2)-1))
            parid = choice(np.arange(id+2, n_tfs),
                           k, replace=False)
            parid = np.pad(parid, (0,1), 'constant')
            adj_mat[parid, parid] = 1.0

            # Single-input modules (SIMs): from all tfs
            targets = np.zeros((n_tfs, n_targets))
            simid = choice(np.arange(n_tfs), n_targets)
            targets[simid, np.arange(n_targets)] = 1.0
            sign = choice([1,-1], size=targets.shape, p=[1-frac, frac])
            targets = np.multiply(targets, sign)
            targets[targets==-0.0] = 0.0

            # represent the full network in a combined matrix
            self.grn = np.concatenate((adj_mat, targets), axis=1)

            # PART 2: SPECIFY INTERACTION STRENGTHS
            # genes have connection weights ~ gamma distribution
            self.grn *= gamma(shape=k_strengths,size=self.grn.shape)
            grnvals = abs(self.grn[self.grn.nonzero()])
            difference = grnvals.max()-grnvals.min()
            self.grn[self.grn > 0] -= grnvals.min()-10e-5
            self.grn[self.grn < 0] += grnvals.min()-10e-5
            self.grn[self.grn.nonzero()] /= difference+10e-5
            self.grn[self.grn.nonzero()] *= (strengths_max - \
                                                strengths_min)
            self.grn[self.grn > 0] += strengths_min
            self.grn[self.grn < 0] -= strengths_min

            # Early HC cacade strongly inactivates toggle_0
            grnvals = abs(self.grn[self.grn.nonzero()])
            self.grn[id-1, id] = -np.quantile(grnvals, 0.99)
            self.grn[id-1, id] *= normal(2.0, 0.01)

            # Asymmetric DNL strengths, favoring toggle_1
            self.grn[id+1, id] = -np.quantile(grnvals, 0.99)
            self.grn[id+1, id] *= normal(2.0, 0.01)
            self.grn[id, id+1] = -np.quantile(grnvals, 0.99)
            self.grn[id, id+1] *= normal(1.0, 0.01)

            # Toggle_0 strongly activates early cascade
            self.grn[id, 0] = np.quantile(grnvals, 0.95)
            self.grn[id, 0] *= normal(1.0, 0.01)

            # Strong PAR at selected genes throughout
            self.grn[parid, parid] = np.quantile(grnvals, 0.95)
            for i in parid: self.grn[i, i] *= normal(1.0, 0.01)

            self.grn = pd.DataFrame(self.grn, index=[f'g{i}' for i in np.arange(self.grn.shape[0])],
                                             columns=[f'g{i}' for i in np.arange(self.grn.shape[1])])

            self.grn_full = pd.DataFrame(np.pad(self.grn, ((0, self.grn.shape[1]-self.grn.shape[0]), (0,0))),
                                                     index = [f'g{i}' for i in np.arange(self.grn.shape[1])],
                                                     columns=[f'g{i}' for i in np.arange(self.grn.shape[1])])

            self.grn.to_csv(f'{self.directory}/model.csv')

            # write parameters to log file
            log = open(f'{self.directory}/log.txt', 'w')
            log.write(

f"""n_tfs = {n_tfs}
n_targets = {n_targets}
selfs = {selfs}
frac = {frac}
strengths_min = {strengths_min}
strengths_max = {strengths_max}
k_strengths = {k_strengths}\n"""

                        )

            log.close()

        # PART 3: VISUALIZE THE GENE REGULATORY NETWORK
        cmin, cmax = 0, 0   # custom function for returning colors from colormap
        def cmap_(val): return rgb_to_hsv(Spectral_10.mpl_colormap(norm(val))[:-1])
        with open(f"{self.directory}/log.txt") as log:
            for line in log.readlines()[4:6]:
                p, _, val = line.partition(' = ')
                if p=='strengths_min': cmin = float(val)
                elif p=='strengths_max': cmax = float(val)
        norm = mpl.colors.Normalize(vmin=-(cmax-cmin), vmax=(cmax-cmin))

        # get location of DNL for pretty plotting
        id = self.grn.values[:,0].nonzero()[0][1]

        # plot the core tf cascade using a networkx graph as base
        G = nx.from_pandas_adjacency(self.grn.iloc[:n_tfs,:n_tfs],
                                         create_using=nx.DiGraph)

        # update edge arrows
        for i,j in G.edges:
            c = cmap_((abs(self.grn.loc[i,j])-cmin)*\
                            np.sign(self.grn.loc[i,j]))
            G[i][j]['fillcolor'] = f'{c[0]} {c[1]} {c[2]}'
            if self.grn.loc[i,j] < 0:
                G[i][j]['arrowhead'] = 'box'
            elif i=='g0' and j=='g0':
                G[i][j]['tailport'] = 'n'
                G[i][j]['headport'] = 'c'
            elif i==j:
                G[i][j]['tailport'] = 'sw'
                G[i][j]['headport'] = 'c'

        # AND gates for early genes
        for i in np.arange(n_tfs):
            if i>0 and i<id:
                n = G.number_of_nodes()
                G.add_node(n+1)
                G.nodes[n+1]['label'] = 'AND'
                G.nodes[n+1]['fontsize'] = '8'
                G.nodes[n+1]['shape'] = 'house'
                G.nodes[n+1]['width'] = '0.3'
                G.nodes[n+1]['height'] = '0.3'
                predecessors = [x for x in G.predecessors(f'g{i}') if x!=i]
                for j in predecessors:
                    G.add_edge(j,n+1)
                    G[j][n+1]['fillcolor'] = G[j][f'g{i}']['fillcolor']
                    G[j][n+1]['weight'] = 100.0
                    G.remove_edge(j, f'g{i}')
                G.add_edge(n+1,f'g{i}')
                G[n+1][f'g{i}']['arrowsize'] = '0.5'

        G.graph['node'] = {'shape' : 'circle',
                           'fixedsize' : 'True',
                           'fontsize' : '20'}

        G.graph['edge'] = {'arrowsize' : '1.0'}

        A = nx.nx_agraph.to_agraph(G)
        A.graph_attr['dpi'] = dpi
        A.layout('circo', args=f'-Gmindist={mindist} -Groot=g{id+1}')
        return Image(A.draw(format='png'))




    def simulate_model(self, sim_time, n_cells, drop_cutoff, drop_prob, par_std=0.5,
                                        n_datasets=1, load_prev=False, verbose=True):
        """
        generate synthetic single cell dataset(s) from the grn (BoolODE)
        note: for more details, see Pratapa et al., 2020, Nature Methods

        parameters:
        * sim_time: int, specifies the total number of time units for each
          stochastic simulation (1 simulation -> 1 cell)
        * n_cells: int, number of cells to generate for the synthetic dataset,
          each sampled from a single timepoint in a single stochastic simulation
        * drop_cutoff and drop_prob: each float in [0.0, 1.0], the bottom q% of
          cells expressing a gene (as specified by drop_cutoff) will have their
          expression set to zero with q% chance (as specified by drop_prob)
        * par_std: float, standard deviation of normal distribution used to
          sample kinetic rate parameters for stochastic differential equations
        * n_datasets: int, number of datasets to sample from the simulations
        * load_prev: bool, whether or not to load existing simulation results
        * verbose: bool, whether or not to print parameters from prev simulations


        attributes:
        * synthdata.expression: list of pd.DataFrames (len=n_datasets), cells x
          gene expression values matrices
        * synthdata.t: list of pd.DataFrames (len=n_datasets), simulation times
          for cells in expression matrices
        """

        if load_prev:
            try:
                # load sampled expression, times
                self.expression, self.t = [], []
                for i in range(1, n_datasets+1):
                    self.expression.append(pd.read_csv(
                    f'{self.directory}/{self.name}-{n_cells}-{i}-{int(100*drop_cutoff)}-{drop_prob}/ExpressionData.csv', index_col=0).T)
                    self.t.append(pd.read_csv(f'{self.directory}/PseudoTime.csv', index_col=0))
                    self.t[i-1] = self.t[i-1].loc[self.expression[i-1].index, 'Time'].sort_values()
                    self.expression[i-1] = self.expression[i-1].loc[self.t[i-1].index, :]
                if verbose:
                    with open(f"{self.directory}/log.txt") as log:
                        params = []
                        for line in (log.readlines()[7:13]):
                            params.append(str(line))
                    print('Succesfully loaded previous simulations with the following parameters: ' + \
                                                                    ', '.join(params).replace('\n',''))
                else:
                    print('Finished loading expression matrices from previous simulations.')
            except:
                print("Couldn't find simulation files to load previous results.")



        else:
            # delete previous simulations & results
            for f in os.scandir(f'{self.directory}/'):
                if f.is_dir(): shutil.rmtree(f.path)

            print('Writing required configuration files for BoolODE...')

            # get location of DNL for initial conditions
            id = self.grn.values[:,0].nonzero()[0][1]

            # write model, strengths, and initial conditions
            model = open(f'{self.directory}/model.txt', 'w')
            strengths = open(f'{self.directory}/strengths.txt', 'w')
            ics = open(f'{self.directory}/ics.txt', 'w')
            model.write("Gene\tRule\n")
            strengths.write("Gene1\tGene2\tStrength\n")
            ics.write("Genes\tValues\n")
            ics_list, genes_list = [], []
            for g in np.arange(self.grn.shape[1]):

                # translate grn matrix to boolean strings
                a = np.where(self.grn.values[:,g] > 0)[0]
                r = np.where(self.grn.values[:,g] < 0)[0]

                # * activation: AND / OR
                a_and, a_or = [], []
                if g>0 and g<id:
                    a_and = [f'g{g-1}', f'g{id}']
                    a_and = '('+' and '.join(a_and)+')'
                    a_or = [f'g{str(x)}' for x in a
                            if x!=(g-1) and x!=id]
                    a_or.append(a_and)
                else:
                    a_or = [f'g{str(x)}' for x in a]

                # -> concatenate
                if len(a_or)>1:
                    a = '('+' or '.join(a_or)+')'
                elif len(a_or)==1:
                    a = a_or[0]
                else: a = ''

                # * repression: OR only
                r = [f'g{str(x)}' for x in r]

                # -> concatenate
                if len(r)>0:
                    r = 'not ('+' or '.join(r)+')'
                else: r = ''

                # write rules to file
                gene = f'g{str(g)}'
                if len(a)>0 and len(r)>0:
                    rule = f'{a} and {r}'
                else: rule = f'{a}{r}'
                model.write(f'{gene}\t{rule}\n')

                # write interaction strengths: target, tf
                for h in self.grn.values[:,g].nonzero()[0]:
                    strengths.write(f'g{g}\tg{h}\t{abs(self.grn.values[h,g])}\n')

                # write initial conditions to file
                genes_list.append(f'g{g}')
                # Toggle_0 (-> early genes) starts ON
                if g==id: ics_list.append(2)
                # Toggle_1 (-> late genes) and others start OFF
                elif g in np.arange(self.grn.shape[0]):
                    ics_list.append(0.001)
                elif self.grn.values[:,g].sum()>0:
                    ics_list.append(0.001)
                else: ics_list.append(2)
            ics.write(f"{genes_list}\t{ics_list}")
            model.close()
            strengths.close()
            ics.close()



            # write boolODE configuration file (.yaml)
            config = open(f'{self.directory}/boolconfig.yaml', 'w')
            config.write(

f"""global_settings:
    model_dir: 'inputs'
    output_dir: 'inputs'
    do_simulations: True
    do_post_processing: True
    modeltype: 'hill'

jobs:
    - name: '{self.name}'
      model_definition: '{self.name}/model.txt'
      interaction_strengths: '{self.name}/strengths.txt'
      model_initial_conditions: '{self.name}/ics.txt'
      simulation_time: {sim_time}
      num_cells: {n_cells}
      do_parallel: True
      sample_pars: True
      sample_std: {par_std}
      identical_pars: False

post_processing:
    GenSamples:
        - sample_size: {n_cells}
          nDatasets: {n_datasets}

    Dropouts:
        - dropout: True
          drop_cutoff: {drop_cutoff}
          drop_prob: {drop_prob}"""

                            )

            config.close()



            print('Simulating the gene regulatory network in shell...')
            os.system(f'python {self.boolODE_dir}/boolode.py --config {self.directory}/boolconfig.yaml')
            print('Finished simulating synthetic gene regulatory network and sampling cells.')



            # load sampled expression, times
            self.expression, self.t = [], []
            for i in range(1, n_datasets+1):
                self.expression.append(pd.read_csv(
                f'{self.directory}/{self.name}-{n_cells}-{i}-{int(100*drop_cutoff)}-{drop_prob}/ExpressionData.csv', index_col=0).T)
                self.t.append(pd.read_csv(f'{self.directory}/PseudoTime.csv', index_col=0))
                self.t[i-1] = self.t[i-1].loc[self.expression[i-1].index, 'Time'].sort_values()
                self.expression[i-1] = self.expression[i-1].loc[self.t[i-1].index, :]



            # update log file
            with open(f"{self.directory}/log.txt") as log:
                lines = log.readlines()
            with open(f"{self.directory}/log.txt", 'w') as log:
                log.writelines(lines[:7])
                log.write(

f"""sim_time = {sim_time}
n_cells = {n_cells}
drop_cutoff = {drop_cutoff}
drop_prob = {drop_prob}
par_std = {par_std}
n_datasets = {n_datasets}\n"""

                            )




    def impute_expression(self, genes='all_genes', verbose=True):
        """
        impute missing expression from dropouts (MAGIC)
        note: for more details, see van Dijk et al., 2018, Cell

        parameters:
        * genes: 1D array of genes to impute, optional (default all)

        attributes:
        * synthdata.imputed: list of pd.DataFrames (len=n_datasets),
          cells x imputed expression matrices
        """

        self.imputed = []
        for i in range(len(self.expression)):
            if verbose: print(f'Imputing dataset #{i+1}...')
            magic_op = MAGIC(t='auto', verbose=verbose, random_state=0)
            self.imputed.append(magic_op.fit_transform(
                                        self.expression[i], genes=genes))
            self.imputed[i] = self.imputed[i].loc[self.t[i].index, :]

        if verbose==False:
            print('Finished imputing expression via data diffusion.')




    def plot_tsne(self, data, perplexity=30, s=1, fs=6, datasets='all'):
        """
        plot tsne of simulated single cell experiment(s), colored by t
        """

        if data=='expression': X = self.expression
        elif data=='imputed': X = self.imputed

        if datasets=='all': datasets = range(len(X))

        fig, t_max = plt.figure(), 0
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(datasets)),
                 axes_pad=0.05, cbar_location='right', cbar_mode='single',
                 cbar_size='10%', cbar_pad=0.05)

        for i in range(len(datasets)):
            ax = grid[i]
            x = TSNE(perplexity=perplexity).fit_transform(X[datasets[i]])
            ax.scatter(x[:,0], x[:,1], c=self.t[datasets[i]],
                       s=s, cmap=Hawaii_5.mpl_colormap)
            ax.tick_params(axis='both', length=0,
                    labelbottom=False, labelleft=False)
            ax.set_xlabel('tSNE1', fontsize=fs)
            l,r = ax.get_xlim()
            b,t = ax.get_ylim()
            ax.set_aspect(abs((r-l)/(b-t)))
            ax.set_box_aspect(1)

            for loc in ['top','bottom','left','right']:
                ax.spines[loc].set_linewidth(0.5)

        with open(f"{self.directory}/log.txt") as log:
            _,_,t_max = log.readlines()[-6].partition(' = ')

        norm = mpl.colors.Normalize(vmin=self.t[datasets[i]].min(),
                                    vmax=self.t[datasets[i]].max())
        ax.cax.cla()
        cbar = Colorbar(mappable=ScalarMappable(norm=norm,
                        cmap=Hawaii_5.mpl_colormap), ax=ax.cax,
                        ticks=[self.t[datasets[i]].min(),
                               self.t[datasets[i]].max()])
        cbar.ax.set_ylabel('Time', rotation=-90, va='top', fontsize=fs)
        cbar.ax.set_yticklabels([0, int(t_max)], fontsize=fs)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(length=0)

        grid[0].set_ylabel('tSNE2', fontsize=fs)
        plt.show()




    def plot_expression(self, data, dpi=1000, ar=1, fs=6, datasets='all', **kwargs):
        """
        plot gene expression heatmaps (tfs only)
        """

        print('Genes, from top-to-bottom: g6, g0-g5, g7-g14')

        dpi_prev = mpl.rcParams['figure.dpi']
        mpl.rcParams['figure.dpi'] = dpi

        if data=='expression': X = self.expression
        elif data=='imputed': X = self.imputed

        id = self.grn.values[:,0].nonzero()[0][1]
        genes = [f'g{id}']+[f'g{i}' for i in range(self.grn.shape[0]) if i!=id]
        if datasets=='all': datasets = range(len(X))
        else: datasets -= 1

        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(datasets)), axes_pad=0.05)

        for i in range(len(datasets)):
            ax = grid[i]
            sns.heatmap(X[datasets[i]][genes].T, ax=ax, cbar=False,
                        xticklabels=False, yticklabels=False, **kwargs)
            ax.set_title(f'Dataset{datasets[i]+1}', fontsize=fs*0.9)
            ax.set_xlabel('Time', fontsize=fs)
            l,r = ax.get_xlim()
            b,t = ax.get_ylim()
            ax.set_aspect(abs((r-l)/(b-t))*ar)

        grid[0].set_ylabel('Genes', fontsize=fs)
        plt.show(); mpl.rcParams['figure.dpi'] = dpi_prev




    def generate_tfbs(self, bindsite_prob, pmf_true_bindsites, pmf_others,
                      exclude_selfs=True, load_prev=False, verbose=True,
                      plot=True, dpi=1000, fs=6, datasets='all', **kwargs):

        """
        generate binding sites matrices for each dataset given the regulatory network

        parameters:
        * bindsite_prob: float [0.0,1.0], probability that at least one binding site
          exist in the promoter of a target for any given TF
        * pmf_true_bindsite: list of floats [with sum(list)=1.0], where each element
          is the probability that n=list.index(element) binding sites exist in the
          promoter of a target gene for a true regulator
        * pmf_others: see above; each element is the probability that n binding sites
          exist for a non-regulator
        * exclude_selfs: bool, whether or not to include binding sites for a TF in its
          own promoter (i.e. include self-regulation in model of grn inference) [optional,
          default True]
        * load_prev: bool, whether or not to load existing tfbs matrix for model
        * verbose: bool, whether or not to print the parameters from prev matrix and
          average proportions (if plot=True)

        attributes:
        * synthdata.tfbs: list of genes x genes arrays of len(n_datasets), where element
          i,j is the number of binding sites for TF_i in promoter of TF_j for the dataset
        """

        if load_prev:
            try:
                self.tfbs = [ None ] * len(self.expression)
                for i in range(len(self.expression)):
                    self.tfbs[i] = pd.read_csv(f'{self.directory}/bindsites/tfbs-{i}.csv', index_col=0)

                if verbose:
                    with open(f"{self.directory}/log.txt") as log:
                        params = []
                        for line in (log.readlines()[13:17]):
                            params.append(str(line))
                    print('Succesfully loaded previous binding sites matrices with the following parameters: ' + \
                                                                    ', '.join(params).replace('\n',''))
            except:
                print("Couldn't find binding sites files to load previous results.")
        else:
            # check for directory containing binding sites files
            if os.path.exists(f"{self.directory}/bindsites/"): pass
            else: os.mkdir(f"{self.directory}/bindsites/")

            # delete previous binding sites files if they exist
            for f in os.scandir(f"{self.directory}/bindsites/"):
                os.remove(f.path)

            self.tfbs = [ None ] * len(self.expression)
            for i in range(len(self.expression)):

                # binding sites given ground truth network
                self.tfbs[i] = (self.grn_full!=0).astype(bool)

                # binding sites for additional random pairs
                self.tfbs[i] = np.logical_or(self.tfbs[i], choice([True, False],
                                             p=[bindsite_prob, 1-bindsite_prob],
                                             size=self.tfbs[i].shape))

                self.tfbs[i] = self.tfbs[i].astype(int)

                # number of binding sites for true regulators and others using probability mass functions
                self.tfbs[i].values[(self.grn_full!=0).values] *= choice(np.arange(1,len(list(pmf_true_bindsites))+1,dtype=int),
                                                                         p=pmf_true_bindsites, size=(self.grn_full!=0).values.sum(axis=None))
                self.tfbs[i].values[(self.grn_full==0).values] *= choice(np.arange(1,len(list(pmf_others))+1,dtype=int),
                                                                         p=pmf_others, size=(self.grn_full==0).values.sum(axis=None))
                if exclude_selfs:
                    self.tfbs[i].values[np.arange(self.tfbs[i].shape[0]), np.arange(self.tfbs[i].shape[0])] = 0

                self.tfbs[i].to_csv(f'{self.directory}/bindsites/tfbs-{i}.csv')

            # update log file
            with open(f"{self.directory}/log.txt") as log:
                lines = log.readlines()
            with open(f"{self.directory}/log.txt", 'w') as log:
                log.writelines(lines[:13])
                log.write(

f"""bindsite_prob = {bindsite_prob}
pmf_true_bindsites = {pmf_true_bindsites}
pmf_others = {pmf_others}
exclude_selfs = {exclude_selfs}"""

                            )


        if plot:
            print('\nGenes, from top-to-bottom and left-to-right: g6, g0-g5, g7-g14')

            # plot matrices (tfs only)
            dpi_prev = mpl.rcParams['figure.dpi']
            mpl.rcParams['figure.dpi'] = dpi

            id = self.grn.values[:,0].nonzero()[0][1]
            genes = [f'g{id}']+[f'g{i}' for i in range(self.grn.shape[0]) if i!=id]
            if datasets=='all': datasets = range(len(self.tfbs))
            else: datasets -= 1

            fig = plt.figure()
            grid = ImageGrid(fig, 111, nrows_ncols=(1, len(datasets)), axes_pad=0.05)

            for i in range(len(datasets)):
                ax = grid[i]
                sns.heatmap(self.tfbs[i].loc[genes, genes], ax=ax, cbar=False, square=True,
                            vmax=self.tfbs[i].max().max(), xticklabels=False, yticklabels=False, **kwargs)
                ax.set_title(f'BindMat{i+1}', fontsize=fs*0.9)
                ax.set_xlabel('Genes', fontsize=fs)

            grid[0].set_ylabel('Genes', fontsize=fs)
            plt.show(); mpl.rcParams['figure.dpi'] = dpi_prev

            if verbose:
                proportions = np.zeros((len(self.tfbs), self.tfbs[i].max().max()))
                for i in np.arange(len(self.tfbs)):
                    for j in np.arange(self.tfbs[i].max().max()):
                        proportions[i,j] = self.tfbs[i].values[self.tfbs[i]==j+1].size/self.tfbs[i].size

                print('Proportion n bindsites:')
                for j in np.arange(self.tfbs[i].max().max()):
                    print(f'{j+1}: {proportions[:,j].mean()}')




    def infer_network(self, mode, pars='default', load_prev=False, verbose=True):
        """
        NEED DESCRIPTION
        """

        n_cells, drop_cutoff, drop_prob = 0, 0.0, 0.0
        with open(f"{self.directory}/log.txt") as log:
            for line in log.readlines()[8:11]:
                p, _, val = line.partition(' = ')
                if p=='n_cells': n_cells = int(val)
                elif p=='drop_cutoff': drop_cutoff = float(val)
                elif p=='drop_prob': drop_prob = float(val)

        if mode=='beeline':
            if load_prev:
                try:
                    self.bee_auc = pd.read_csv(f'outputs/{self.name}/{self.name}-AUPRC.csv', index_col=0)

                    if verbose:
                        print('Succesfully loaded previous results from the following algorithms: ' + \
                                                                    ', '.join(list(self.bee_auc.index)))
                except:
                    print("Couldn't find results from previous Beeline run.")

            else:
                # delete previous beeline results
                for f in os.scandir(f'outputs/{self.name}/'):
                    if f.is_dir(): shutil.rmtree(f.path)

                if pars=='default':
                    pars = {'algorithms' : ['SINCERITIES',
                                            'PPCOR',
                                            'GRNBOOST2',
                                            'GENIE3',
                                            'PIDC'],

                            'SINCERITIES' : {'nBins' : 10},
                            'PPCOR'       : {'pVal' : 0.01}}

                # write BEELINE configuration file (.yaml)
                config = open(f'{self.directory}/beeconfig.yaml', 'w')
                config.write(

f"""input_settings:
    input_dir: 'inputs'
    dataset_dir: '{self.name}'
    datasets:\n"""

                            )

                for i in range(len(self.expression)):
                    config.write(

f"""        - name: '{self.name}-{n_cells}-{i+1}-{int(100*drop_cutoff)}-{drop_prob}'
          exprData: 'ExpressionData.csv'
          cellData: 'PseudoTime.csv'
          trueEdges: 'refNetwork.csv'\n"""

                                    )

                config.write(

f"""    algorithms:
        - name: 'SINCERITIES'
          params:
              should_run: [{'SINCERITIES' in pars['algorithms']}]
              nBins: [{pars['SINCERITIES']['nBins']}]
        - name: 'PPCOR'
          params:
              should_run: [{'PPCOR' in pars['algorithms']}]
              pVal: [{pars['PPCOR']['pVal']}]
        - name: 'GRNBOOST2'
          params:
              should_run: [{'GRNBOOST2' in pars['algorithms']}]
        - name: 'GENIE3'
          params:
              should_run: [{'GENIE3' in pars['algorithms']}]
        - name: 'PIDC'
          params:
              should_run: [{'PIDC' in pars['algorithms']}]\n"""

                            )

                config.write(

f"""output_settings:
    output_dir: 'outputs'
    output_prefix: '{self.name}' """

                                )

                config.close()

                print('Running Beeline in the shell...')
                os.system(f'python {self.beeline_dir}/BLRunner.py --config {self.directory}/beeconfig.yaml')
                os.system(f'python {self.beeline_dir}/BLEvaluator.py --config {self.directory}/beeconfig.yaml --auc')
                self.bee_auc = pd.read_csv(f'outputs/{self.name}/{self.name}-AUPRC.csv', index_col=0)
                print('Finished inferring gene regulatory network.')


        if mode=='lasso':
            ############### UPDATE THIS TO LOAD GRID, HYPERPARAMETERS ###################
            if load_prev:
                try:
                    self.lasso_auc = pd.read_csv(f'outputs/{self.name}/lasso-AUPRC.csv', index_col=0)

                    if verbose:
                        print('Succesfully loaded previous grid search-optimized Lasso results.')

                except:
                    print("Couldn't find Lasso results from previous run.")
            #############################################################################
            else:
                if pars=='default':
                    pars = {'lambda' : np.logspace(-5, 0, 6),
                            'sigma' : np.linspace(0, 10, 6),
                            'split' : 0.7, 'exclude_selfs' : True}

                n_train = int(pars['split'] * len(self.expression))
                print(f'Performing Lasso with n={n_train} training datasets...')

                # grid search over lambda, sigma hyperparameters
                ll, ss = np.meshgrid(pars['lambda'], pars['sigma'])
                ll_idx, ss_idx = np.indices(ll.shape)
                ll_idx, ss_idx = ll_idx.ravel(), ss_idx.ravel()
                lasso_grid = np.zeros((n_train, ll.shape[0], ll.shape[1]))   # directed, unsigned auprc

                for i in tqdm(np.arange(lasso_grid.shape[0])):

                    print(f'Optimizing over grid for dataset{i+1}:')
                    for j in tqdm(np.arange(ll_idx.size)):

                        sce = self.expression[i][self.tfbs[i].columns].astype(np.float64)
                        tfbs, y_true = self.tfbs[i].astype(np.float64), (self.grn_full!=0)

                        lasso_grid[i,ll_idx[j],ss_idx[j]], _ = inference(sce=sce, tfbs=tfbs, y_true=y_true,
                                                                         Λ=ll[ll_idx[j], ss_idx[j]],
                                                                         σ=ss[ll_idx[j], ss_idx[j]],
                                                                         exclude_selfs=pars['exclude_selfs'])

                lasso_grid = lasso_grid.swapaxes(1,2)
                idx = lasso_grid.mean(axis=0).ravel().argmax()
                lambda_best = ll.T.ravel()[idx]
                sigma_best = ss.T.ravel()[idx]

                # save grid search results for use later
                np.save(f'outputs/{self.name}/grid', lasso_grid)
                np.save(f'outputs/{self.name}/lambda', ll.T)
                np.save(f'outputs/{self.name}/sigma', ss.T)

                # inference results
                self.lasso_auc = pd.DataFrame(columns=[f'{self.name}-{n_cells}-{i+1}-{int(100*drop_cutoff)}-{drop_prob}'
                                                       for i in range(len(self.expression))], index=['LASSO', 'split'])
                for i in np.arange(n_train):
                    self.lasso_auc.iloc[:, i] = lasso_grid[i,...].ravel()[idx], 'train'

                print(f'Testing performance on n={len(self.expression)-n_train} development datasets...')
                for i in tqdm(np.arange(n_train, len(self.expression))):

                    sce = self.expression[i][self.tfbs[i].columns].astype(np.float64)
                    tfbs, y_true = self.tfbs[i].astype(np.float64), (self.grn_full!=0)

                    self.lasso_auc.iloc[0, i], _ = inference(sce=sce, tfbs=tfbs, y_true=y_true,
                                                             Λ=lambda_best, σ=sigma_best,
                                                             exclude_selfs=pars['exclude_selfs'])
                    self.lasso_auc.iloc[1, i] = 'dev'

                self.lasso_auc.to_csv(f'outputs/{self.name}/lasso-AUPRC.csv')
                print('Successfully completed Lasso training.')





#      NEED PLOTTING FUNCTIONS FOR: 1) GRID SEARCH RESULTS; 2) AUPRC RESULTS; 3) INFERRED NETWORK

#                dpi_prev = mpl.rcParams['figure.dpi']
#                mpl.rcParams['figure.dpi'] = 100
#
#                plt.figure(figsize=(6,6))
#                sns.heatmap(grid, annot=True, linewidths=0.5, square=True, cbar=False)
#                plt.show(); mpl.rcParams['figure.dpi'] = dpi_prev
