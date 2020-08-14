import matplotlib        as mpl
import numpy             as np
import pandas            as pd
import trackpy           as tp

import os
import tqdm
import pims
import skimage
import h5py
import PIL
import cellpose

from matplotlib          import pyplot as plt
from matplotlib.colors   import hsv_to_rgb
from matplotlib.colors   import rgb_to_hsv
from cellpose.models     import Cellpose
from skimage.measure     import label
from skimage.measure     import regionprops
from skimage.color       import rgb2gray
from skimage.color       import gray2rgb
from PIL                 import Image
from PIL                 import ImageDraw
from PIL                 import ImageFont


class idataset:

    def __init__(self, frames_directory, date, exper_id,
                    times_directory, masks_directory, results_directory):

        self.directory           = frames_directory
        self.date                = date
        self.exper_id            = exper_id
        self.tdirectory          = times_directory
        self.mdirectory          = masks_directory
        self.rdirectory          = results_directory
        self.frames              = None
        self.n_frames            = None
        self.max_intensities     = None
        self.timepoints          = None
        self.masks_cellpose      = None
        self.tracking            = None
        self.n_particles         = None
        self.masks_linked        = None
        self.colors              = None
        self.avg_intensities     = None

        self.get_frames()
        self.get_timepoints()



    def get_frames(self):
        """ compile a list of all tif stacks for given timelapse

        attributes:
        * idataset.frames: list of pims stacks
        * idataset.n_frames: int, number of frames
        * idataset.max_intensities: 1D np array, max pixel
          intensity at each depth through the timelapse """

        try:
            folders, self.frames = os.listdir(self.directory), []
            for folder in folders:
                if self.date in folder and self.exper_id in folder:
                    wd, cont, t = self.directory + folder, True, 1
                    while cont:
                        filename = wd + '/' + folder + '_w1_t' + str(t) + '.tif'
                        if os.path.isfile(filename):
                            self.frames.append(pims.open(filename)); t += 1
                        else:
                            cont = False

            # find max intensity at each depth
            self.n_frames = len(self.frames)
            self.max_intensities = np.zeros((len(self.frames[0]),))
            for i in range(self.n_frames):
                frameMaxIntensities = np.array(self.frames[i]).max(axis=1)
                frameMaxIntensities = frameMaxIntensities.max(axis=1)
                idx = (frameMaxIntensities > self.max_intensities).nonzero()
                self.max_intensities[idx] = frameMaxIntensities[idx]

            print("Loaded timelapse:")
            print(f"t = {self.n_frames}")
            print(f"z = {len(self.frames[0])}")
            print(f"y = {self.frames[0].frame_shape[0]}")
            print(f"x = {self.frames[0].frame_shape[1]}")

        except:
            print('Could not load the frames. Check frames_directory, date and exper_id.')



    def get_timepoints(self):
        """ get timepoints (in minutes) for each frame in timelapse

        attributes:
        * idataset.timepoints: np array, timepoints in minutes """

        try:
            files = os.listdir(self.tdirectory)
            for file in files:
                if self.date in file and self.exper_id in file:
                    timepoints_df = pd.read_csv(self.tdirectory + file)
                    self.timepoints = timepoints_df['Timepoint [min]']
                    self.timepoints = self.timepoints.values[0:self.n_frames]
        except:
            print('Could not load timepoints. Check times_directory, date and exper_id.')



    def text_stamp(self, text, size, maxIntensity):
        """ generate a text stamp as grayscale array """

        pil_font = ImageFont.truetype("Arial.ttf", encoding="unic",
                                                size=size // len(text))
        text_width, text_height = pil_font.getsize(text)
        canvas = Image.new('RGB', [size, size], (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        offset = ((size - text_width) // 2, (size - text_height) // 2)
        draw.text(offset, text, font=pil_font, fill="#000000")
        canvas = 255 - np.asarray(canvas)
        canvas = rgb2gray(canvas) * maxIntensity; mask = canvas > 0
        canvas = canvas[np.ix_(mask.any(1), mask.any(0))]
        return canvas.astype(int)



    def show_frames_at_depth(self, z):
        """ return scrollable frames at given depth """

        frames_list, size, border, intensity = [], 200, 20, self.max_intensities[z]
        for i in range(self.n_frames):
            frame = self.frames[i][z]  # generate the timestamp as a square array (size x size)
            timestamp = self.text_stamp("{:03d}m".format(self.timepoints[i]), size, intensity)

            # select the indices where text exists -> frame corner
            idxs = np.stack(list(timestamp.nonzero()), axis=0).T; idxs_frame = idxs.copy()
            idxs_frame[:,0] += frame.shape[0] - timestamp.shape[0] - border
            idxs_frame[:,1] += frame.shape[1] - timestamp.shape[1] - border
            frame[idxs_frame[:,0], idxs_frame[:,1]] = timestamp[idxs[:,0], idxs[:,1]]

            frames_list.append(frame)
        return pims.Frame(frames_list)



    def crop_z(self, zmin, zmax):
        """ crop frames along z axis (depth) through timelapse """

        for i in range(self.n_frames):
            self.frames[i] = self.frames[i][zmin:zmax+1]
        self.max_intensities = self.max_intensities[zmin:zmax+1]

        print("Cropped timelapse:")
        print(f"t = {self.n_frames}")
        print(f"z = {len(self.frames[0])}")
        print(f"y = {self.frames[0].frame_shape[0]}")
        print(f"x = {self.frames[0].frame_shape[1]}")



    def segment_frames(self, diameter=90, load_prev=False):
        """ segment cells at each depth (individually) for stacks in timelapse

        parameters:
        * diameter: diameter of cells, in pixels (optional, default 90)
        * load_prev: bool, load segmentation from previous run (optional,
          default False)

        attributes:
        * idataset.masks_cellpose: list of masks (corresponding to frames)
        * idataset.mdirectory: str, mask directory for saving/loading """

        if load_prev:
            h5f = h5py.File(f"{self.mdirectory}{self.date}_{self.exper_id}.h5",'r')
            masks_4darr = h5f['cellpose_masks'][:]; h5f.close()

            self.masks_cellpose = []
            for i in range(self.n_frames):
                m = np.split(masks_4darr[i,...], masks_4darr.shape[1], axis=0)
                m = [np.squeeze(x) for x in m]
                self.masks_cellpose.append(m)
            print('Successfully loaded segmentation from previous run.')

        else:
            model = Cellpose(gpu=False, model_type='cyto')
            self.masks_cellpose, masks_4darr = [], None

            for frame in self.frames:
                m, _, _, _ = model.eval([frame[i] for i in range(len(frame))],
                                         channels=[0, 0], diameter=diameter)
                self.masks_cellpose.append(m)

                if masks_4darr is None:
                    masks_4darr = np.rollaxis(np.dstack(m), -1)
                    masks_4darr = np.expand_dims(masks_4darr, 0)
                else:
                    masks = np.rollaxis(np.dstack(m), -1)
                    masks = np.expand_dims(masks, 0)
                    masks_4darr = np.append(masks_4darr, masks, 0)

            h5f = h5py.File(f"{self.mdirectory}{self.date}_{self.exper_id}.h5", 'w')
            h5f.create_dataset('cellpose_masks', data=masks_4darr); h5f.close()



    def link_masks(self, searchRange_t, threshold_t, memory_t=3,
                   searchRange_z=20, threshold_z=4, memory_z=1):
        """ link masks within stacks and then across stacks in timelapse

        parameters:
        * searchRange_t[_z]: int, the maximum distance features can move
          between frames [in pixels]
        * threshold_t[_z]: int, minimum number of points to survive
        * memory_t[_z]: int, the maximum number of frames during which a
          feature can vanish, then reappear nearby, and be considered the
          same particle

        attributes:
        * idataset.tracking: pd Dataframe, features contained in particle
          trajectories; mask label = particle number + 1
        * idataset.n_particles: int, number of particles after filtering
          spurious trajectories (stubs)
        * idataset.masks_linked: list of masks (corresponding to frames) """

        # Part 1: link masks within stacks
        masks_link_z = [] # link masks in z
        for mask in self.masks_cellpose:
            # save mask regions as features
            features = pd.DataFrame()
            for num, img in enumerate(mask):
                for region in regionprops(img):
                    y, x = region.centroid[0], region.centroid[1]
                    feature = {'y' : y, 'x' : x, 'frame' : num}
                    feature['mask_cp'] = img[int(y+0.5), int(x+0.5)]
                    features = features.append([feature,])

            # link features via Trackpy
            tracking = tp.link(features, memory=memory_z,
                                search_range=searchRange_z)
            tracking = tp.filter_stubs(tracking,
                            threshold=threshold_z)
            tracking.reset_index(drop=True, inplace=True)

            # link masks using particles identified by Trackpy
            p = tracking['particle'].unique(); m = np.argsort(p) + 1
            particle_to_mask = {p[i] : m[i] for i in range(m.shape[0])}
            tracking['mask'] = tracking['particle'].map(particle_to_mask)
            masks_z = [np.zeros(mask[0].shape) for i in range(len(mask))]
            for idx in tracking.index:
                f = tracking.loc[idx,'frame']
                m_cp = tracking.loc[idx,'mask_cp']
                m_z = tracking.loc[idx,'mask']
                masks_z[f][mask[f]==m_cp] = m_z
            masks_link_z.append(masks_z)


        # Part 2: link masks in 3D across timelapse
        features = pd.DataFrame() # link masks in t
        for num, img in enumerate(masks_link_z):
            img = np.dstack(img).astype(int)
            img = np.rollaxis(img, -1)
            for region in regionprops(img):
                z, y, x = region.centroid[0], region.centroid[1], region.centroid[2]
                feature = {'z' : z, 'y' : y, 'x' : x, 'frame' : num}
                feature['mask_cp'] = img[region.bbox[0], int(y+0.5), int(x+0.5)]
                features = features.append([feature,])

        # link features via Trackpy
        self.tracking = tp.link(features, memory=memory_t,
                                    search_range=searchRange_t)
        self.tracking = tp.filter_stubs(self.tracking,
                                        threshold=threshold_t)
        self.tracking.reset_index(drop=True, inplace=True)

        # link masks using particles identified by Trackpy
        p = self.tracking['particle'].unique(); m = np.argsort(p) + 1
        self.n_particles = self.tracking['particle'].nunique()
        particle_to_mask = {p[i] : m[i] for i in range(m.shape[0])}
        self.tracking['mask'] = self.tracking['particle'].map(particle_to_mask)
        self.masks_linked = [[np.zeros(img.shape[1:]) for _ in range(img.shape[0])]
                                                      for _ in range(self.n_frames)]
        for idx in self.tracking.index:
            f = self.tracking.loc[idx,'frame']
            m_cp = self.tracking.loc[idx,'mask_cp']
            m_z = self.tracking.loc[idx,'mask']
            for z in range(img.shape[0]):
                self.masks_linked[f][z][masks_link_z[f][z]==m_cp] = m_z

        # set colors for showing segmentation
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.n_particles))
        self.colors = rgb_to_hsv(self.colors[:,0:3]); np.random.shuffle(self.colors)



    def show_segmentation_at_depth(self, z):
        """ return scrollable frames with segmentation results at given depth """

        frames_list, size, border, spacing = [], 200, 20, 25
        for i in range(self.n_frames):
            frame = self.frames[i][z]
            mask = self.masks_linked[i][z]

            # add masks from segmentation over top of cells
            HSV = np.zeros((frame.shape[0], frame.shape[1], 3))
            HSV[:,:,2] = frame/self.max_intensities[z]  # [0,1]
            for n in range(int(self.tracking['mask'].max())):
                ipix = (mask==n+1).nonzero()
                HSV[ipix[0],ipix[1],0] = self.colors[n,0]
                HSV[ipix[0],ipix[1],1] = 1.0

                # add mask labels along the bottom
                label = self.text_stamp(f"{n+1}", 50, 1.0)

                # select the indices where text exists in label -> bottom
                idxs = np.stack(list(label.nonzero()), axis=0).T
                idxs_frame = idxs.copy()
                idxs_frame[:,0] += frame.shape[0] - label.shape[0] - border
                idxs_frame[:,1] += border*(n+1) + spacing*n
                HSV[idxs_frame[:,0], idxs_frame[:,1], 0] = self.colors[n,0]
                HSV[idxs_frame[:,0], idxs_frame[:,1], 1] = 1.0
                HSV[idxs_frame[:,0], idxs_frame[:,1], 2] = 0.8

            frame = (hsv_to_rgb(HSV) * self.max_intensities[z]).astype(int)

            # add timestamp: select the indices where text exists in label -> add to frame lower right corner
            timestamp = self.text_stamp("{:03d}m".format(self.timepoints[i]), size, self.max_intensities[z])
            idxs = np.stack(list(timestamp.nonzero()), axis=0).T
            idxs_frame = idxs.copy()
            idxs_frame[:,0] += frame.shape[0] - timestamp.shape[0] - border
            idxs_frame[:,1] += frame.shape[1] - timestamp.shape[1] - border
            frame[idxs_frame[:,0], idxs_frame[:,1], :] = gray2rgb(timestamp[idxs[:,0], idxs[:,1]])

            frames_list.append(frame)
        return pims.Frame(frames_list)



    def cell_per_pixel_intensities(self):
        """ apply masks to frames to compute cells' per pixel intensities

        attributes:
        * idataset.avg_intensities: np array, n_frames x n_particles """

        self.avg_intensities = np.empty((self.n_frames, self.n_particles))
        self.avg_intensities[:] = np.nan
        for i in range(self.n_frames):
            frame = np.array(self.frames[i])
            mask = self.masks_linked[i]
            mask = np.dstack(mask).astype(int)
            mask = np.rollaxis(mask, -1)

            cells = np.arange(self.n_particles)
            cells = cells[np.isin(cells+1, mask)]

            sums = np.bincount(mask.ravel(), frame.ravel())
            counts = np.bincount(mask.ravel())
            idx = cells[np.isin(cells+1, mask)]+1
            avg = sums[idx]/counts[idx]
            self.avg_intensities[i,cells] = avg
        print('Calculated average intensities per pixel.')



    def plot_intensities(self, cells, vline=None):
        """ return a scatter plot of cells' per pixel intensities """

        cells  = [x-1 for x in cells]
        plt.figure()
        for i in cells:
            plt.scatter(self.timepoints, self.avg_intensities[:,i],
                        color=hsv_to_rgb(self.colors[i,:]), marker='.')
        ax = plt.gca()
        l,r = ax.get_xlim()
        b,t = ax.get_ylim()

        if vline != None:
            plt.vlines(vline, b, t, color='k',
                        linestyles='dashed')
            ax.set_ylim(b,t)

        ax.set_xlabel('time (m)')
        ax.set_ylabel('fluor./pix. (AU)')
        ax.set_aspect(abs((r-l)/(b-t))*0.25)
        plt.tight_layout()



    def save_intensities_to_csv(self, precursor, daughter, cutoff=None):
        """ save precursor and daughter cells' per pixel intensities to csv """

        precursor -= 1; daughter -= 1; results = pd.DataFrame()
        results['Timepoints (min)'] = self.timepoints
        results['Precursor'] = self.avg_intensities[:, precursor]
        results['Daughter'] = self.avg_intensities[:, daughter]
        results.set_index('Timepoints (min)', drop=True, inplace=True)

        if cutoff:
            results = results[results.index < cutoff]

        results.to_csv(f"{self.rdirectory}{self.date}_{self.exper_id}.csv")
        print('Successfully saved intensities to csv file in results_directory.')
