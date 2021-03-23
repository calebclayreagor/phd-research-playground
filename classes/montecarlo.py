import tifffile
import cellpose.models
import matplotlib              as mpl
import matplotlib.pyplot       as plt
import numpy                   as np
import pandas                  as pd
import matplotlib.font_manager as fm

from tqdm.notebook                            import tqdm
from numpy.random                             import uniform
from scipy.spatial.distance                   import cdist
from scipy.stats                              import norm
from skimage.segmentation                     import find_boundaries
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from palettable.cartocolors.qualitative       import Bold_10

class branching_model:

    def __init__(self, stack_path, xy_res, z_res):
        self.stack_path = stack_path
        self.aspect = z_res / xy_res
        self.xy_res = xy_res
        self.z_res = z_res
        self.stack_arr = None
        self.xz_maxProj = None
        self.xz_maxNorm = None
        self.xz_maxMask = None
        self.membrane = None
        self.position = None
        self.start_pos = None
        self.pos_mask = None
        self.preprocess_stack()




    def preprocess_stack(self):
        # load tiff stack, generate max projection in xz
        self.stack_arr = tifffile.imread(self.stack_path)
        self.xz_maxProj = np.moveaxis(self.stack_arr,1,3)
        self.xz_maxProj = self.xz_maxProj.max(axis=1)
        self.xz_maxProj = np.flip(self.xz_maxProj, axis=0)

        # generate normalized max projection in xz
        self.xz_maxNorm = self.xz_maxProj.copy()
        q = np.quantile(self.xz_maxNorm, [0.99,0.01], axis=(0,1))
        self.xz_maxNorm[...,0] = np.clip(self.xz_maxNorm[...,0], q[1,0], q[0,0])
        self.xz_maxNorm[...,1] = np.clip(self.xz_maxNorm[...,1], q[1,1], q[0,1])
        self.xz_maxNorm[...,0] = self.xz_maxNorm[...,0] - q[1,0]
        self.xz_maxNorm[...,1] = self.xz_maxNorm[...,1] - q[1,1]
        self.xz_maxNorm[...,0] /= self.xz_maxNorm[...,0].max()
        self.xz_maxNorm[...,1] /= self.xz_maxNorm[...,1].max()
        self.xz_maxNorm = np.pad(self.xz_maxNorm, ((0,0),(0,0),(0,1)))

        print('Loaded stack from path.')
        print(f'xy resolution: {self.xy_res} nm')
        print(f'z resolution: {self.z_res} nm')




    def segment_stack(self, diameter):
        # segment hair cells in max projection using cellpose
        print(f'Running Cellpose with diameter = {diameter} pixels...\n')
        model = cellpose.models.Cellpose(gpu=False, model_type='cyto')
        self.xz_maxMask, _,_,_ = model.eval(self.xz_maxProj[...,0],
                                            channels=[0, 0],
                                            diameter=diameter)

        # find points along boundary of segmented cells -> membrane boundary
        y, x = np.where(find_boundaries(self.xz_maxMask>0, mode='inner'))
        self.membrane = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)

        # rescale membrane to account for aspect ratio
        self.membrane = self.membrane.astype(np.float64)
        self.membrane[:,1] = self.membrane[:,1] * self.aspect




    def show_segmentation(self, separate_channels=False, annotate=None, **kwargs):
        # show xz max projection
        img = self.xz_maxNorm.copy()
        if separate_channels==True:
            # remove axon channel from hair cell segments
            y, x = np.where(self.xz_maxMask>0); img[y,x,1] = 0.
            plt.imshow(img, aspect=self.aspect, origin='lower')
        else:
            plt.imshow(img, aspect=self.aspect, origin='lower')

        # plot hair cell boundaries (i.e. membrane) on top of projection
        plt.scatter(self.membrane[:,0], self.membrane[:,1]/self.aspect,
                                    s=2, linewidth=0, marker='s', c='w')
        # optional annotation
        if annotate is not None:
            if len(annotate[0])==1:
                x = [annotate[0], annotate[0]]
            else: x = annotate[0]

            if len(annotate[1])==1:
                y = [annotate[1], annotate[1]]
            else: y = annotate[1]

            plt.plot(x, y, color='w', **kwargs)

        ax = plt.gca()
        ax.axis('off')

        # add scalebar
        scalebar = AnchoredSizeBar(ax.transData, 5000/self.xy_res, '5 μm', 'lower left',
                                   pad=1, color='white', frameon=False, size_vertical=1,
                                   fontproperties=fm.FontProperties(size=6))
        ax.add_artist(scalebar)
        plt.tight_layout()




    def simulate_metropolisMC(self, start, nsteps, nseeds, d, σ, U, s, U2, β=1, rng_seed=1234):
        # UPDATE VARIABLES TO REFLECT SUMMATION OF POTENTIALS
        print(f'minimum potential: -{U} (AU)')
        print(f'signal length scale: {σ} μm')
        print(f'maximum step size: {d} μm')

        # x-values of simulation starting points
        if len(start[0])==1: x = start[0]
        else: x = np.linspace(start[0][0], start[0][1], nseeds)

        # y-values of simulation starting points
        if len(start[1])==1: y = start[1]
        else: y = np.linspace(start[1][0], start[1][1], nseeds)

        self.position = np.zeros((nseeds, 2, nsteps), dtype=np.float64)
        self.pos_mask = np.zeros(self.xz_maxMask.shape)

        # set starting points
        self.start_pos = start
        self.position[:,0,0] = x
        self.position[:,1,0] = y
        self.position[:,1,0] *= self.aspect

        np.random.seed(rng_seed)

        # length scales: μm -> xy pixels
        σ = σ / self.xy_res * 1e3
        d = d / self.xy_res * 1e3

        # monte carlo random walks (distances in xy pixels)
        for idx in tqdm(range(1, self.position.shape[2])):

            # xi -> current positions
            xi = self.position[...,idx-1].copy()

            # choose distance, direction of random steps
            dist = uniform(0, d, self.position.shape[0])
            theta = uniform(0, 2, self.position.shape[0])

            # xi+1 -> proposed new positions
            xi_1 = self.position[...,idx-1].copy()
            xi_1[:,0] += dist * np.cos(theta * np.pi)
            xi_1[:,1] += dist * np.sin(theta * np.pi)

            # find dist to every point on membrane
            xi_memdist = cdist(xi, self.membrane)
            xi_1_memdist = cdist(xi_1, self.membrane)

            # calculate U_xi (AU) at current positions xi (sum across membrane U's)
            U_xi = self.negative_gaussian_potential(xi_memdist, σ, U).sum(axis=1)

            # calculate U_xi+1 (AU) at proposed new positions xi+1 (sum across membrane U's)
            U_xi_1 = self.negative_gaussian_potential(xi_1_memdist, σ, U).sum(axis=1)

#            # NEED TO FIX: HIGH POTENTIAL FOR BEING NEAR PREVIOUS POSITIONS (VECTORIZE)
#            if idx>1:
#                for j in range(self.position.shape[0]):
#                    print(xi_1[j,:].reshape(1,-1).shape)
#                    print(self.position[j,:,:(idx-1)].reshape(1,-1).shape)
#
#                    xi_1_j_selfdist = cdist(xi_1[j,:].reshape(1,-1), self.position[j,:,:(idx-1)].reshape(1,-1))
#                    U_xi_1[j] -= self.negative_gaussian_potential(xi_1_j_selfdist, s, U2).sum()

            # check if new positions xi+1 are on-grid
            x = np.round(xi_1[:,0]).astype(np.int64)
            y = np.round(xi_1[:,1]/self.aspect).astype(np.int64)
            in_domain = (x>=0) & (x<self.xz_maxMask.shape[1])
            in_range = (y>=0) & (y<self.xz_maxMask.shape[0])
            on_grid = np.where(in_domain & in_range)[0]

            # check if on-grid positions xi+1 are extracellular
            extracellular = ~self.xz_maxMask[y[on_grid],x[on_grid]].astype(bool)
            self.pos_mask[y[on_grid][extracellular],x[on_grid][extracellular]] = 1.

            # accept or reject new positions xi+1 at heat β
            r_move = np.minimum(np.exp(-β * (U_xi_1 - U_xi)), 1)
            r_rand = uniform(0, 1, self.position.shape[0])
            accept = np.greater(r_move, r_rand)
            accept[on_grid] *= extracellular

            # update positions based on acceptance
            self.position[accept,:,idx] = xi_1[accept,:]
            self.position[~accept,:,idx] = xi[~accept,:]




    def negative_gaussian_potential(self, x, σ, U):
        return -U * np.sqrt(2*np.pi) * norm.pdf(x, loc=0, scale=σ)




    def plot_trajectories(self, xtickinterval=5, ytickinterval=5, **kwargs):
        fig, ax = plt.subplots(1,1)

        # plot metropolis MC random walk trajectories
        ax.set_prop_cycle('color', Bold_10.mpl_colors)
        for idx in range(self.position.shape[0]):
            x = self.position[idx,0,:]
            y = self.position[idx,1,:]/self.aspect
            ax.plot(x, y, linewidth=0.5)

        # hair cell boundary
        ax.scatter(self.membrane[:,0], self.membrane[:,1]/self.aspect,
                      s=2, linewidth=0, marker='s', c='k', zorder=1000)

        # line through starting points
        if len(self.start_pos[0])==1:
            x = [self.start_pos[0], self.start_pos[0]]
        else: x = self.start_pos[0]
        if len(self.start_pos[1])==1:
            y = [self.start_pos[1], self.start_pos[1]]
        else: y = self.start_pos[1]
        plt.plot(x, y, color='k', **kwargs)

        y_upper, x_upper, _ = self.xz_maxNorm.shape

        ax.set_xlim([0, x_upper])
        ax.set_ylim([0, y_upper])
        ax.set_aspect(self.aspect)

        xticklabels = np.arange(0, x_upper / 1e3 * self.xy_res, xtickinterval)
        yticklabels = np.arange(0, y_upper / 1e3 * self.z_res, ytickinterval)
        xticklocs = xticklabels * 1e3 / self.xy_res
        yticklocs = yticklabels * 1e3 / self.z_res

        ax.set_xticks(xticklocs)
        ax.set_yticks(yticklocs)
        ax.set_xticklabels(xticklabels.astype(np.int64))
        ax.set_yticklabels(yticklabels.astype(np.int64))

        ax.set_xlabel('μm')
        ax.set_ylabel('μm')
        ax.grid(True)

        plt.tight_layout()




#    def



#
