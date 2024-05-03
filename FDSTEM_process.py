import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as pch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tifffile
import tkinter.filedialog as tkf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

try:
    import cv2
except:
    print('The package "OpenCV" is not installed.')
    print('Thus, symmetry STEM imaging cannot be implemented')

class fourd_viewer:
    def __init__(self, fig, ax, fdata):
        self.fig = fig
        self.ax = ax
        self.fdata = fdata
        self.pick_flag = 0
        self.roi_flag = 0
        self.dif_flag = 0
        self.ind = np.zeros(2).astype(np.int16)
        
        self.sy, self.sx, self.dsy, self.dsx = fdata.shape
        self.log_scale = -1
        self.log_scale_message = "False"
        self.ax[0].set_title("[x, y]=[%d, %d]"%(self.ind[1], self.ind[0]))
        
        self.int_img = np.sum(fdata, axis=(2, 3))
        self.mask = np.zeros((self.sy, self.sx))
        self.mask[self.ind[0], self.ind[1]] = 1

        self.box_ = RectangleSelector(self.ax[0], self.roi_onselect)
        self.ax[0].imshow(self.int_img, cmap="gray")
        self.ax[0].imshow(self.mask, cmap="Reds", alpha=self.mask)
        self.ax[0].axis("off")
        
        
        self.by, self.bx, self.height, self.width = int(self.dsy/2), int(self.dsx/2), int(self.dsy/10), int(self.dsx/10)
        self.df_img = np.sum(self.fdata[:, :, self.by:self.by+self.height, self.bx:self.bx+self.width], axis=(2,3))
        self.ax[1].imshow(self.df_img, cmap="gray")
        self.ax[1].axis("off")
        
        self.box = RectangleSelector(self.ax[2], self.dif_onselect)
        self.ax[2].set_title("log scale: %s"%(self.log_scale_message))
        if self.log_scale == -1:
            self.ax[2].imshow(self.fdata[self.ind[0], self.ind[1]], cmap="gray")
        else:
            self.ax[2].imshow(np.log(self.fdata[self.ind[0], self.ind[1]]), cmap="gray")
        self.df_mask = np.zeros((self.dsy, self.dsx))
        self.df_mask[self.by:self.by+self.height, self.bx:self.bx+self.width] = 0.5
        self.ax[2].imshow(self.df_mask, cmap="Reds", alpha=self.df_mask)
        self.ax[2].axis("off")
        

    def on_press(self, event):
        if event.key == "up":
            if self.ind[0] != 0:
                self.ind[0] -= 1
        elif event.key == "down":
            if self.ind[0] != self.sy:
                self.ind[0] += 1
        elif event.key == "right":
            if self.ind[1] != self.sx:
                self.ind[1] += 1
        elif event.key == "left":
            if self.ind[1] != 0:
                self.ind[1] -= 1
        elif event.key == "l":
            self.log_scale *= -1

        self.update()
        
    def on_pick(self, event):
        if event.inaxes == self.ax[0]:
            my, mx = int(event.ydata), int(event.xdata)
            self.ind[0] = my
            self.ind[1] = mx
            
            self.pick_flag = 1
            self.roi_flag = 0
            self.dif_flag = 0
            self.update()
            
    def roi_onselect(self, eclick, erelease):
        self.by_, self.bx_  = int(eclick.ydata), int(eclick.xdata)
        self.height_, self.width_ = int(erelease.ydata)-int(eclick.ydata), int(erelease.xdata)-int(eclick.xdata)
        
        self.pick_flag = 0
        self.roi_flag = 1
        self.dif_flag = 0
        self.update()    
        
    def dif_onselect(self, eclick, erelease):
        self.by, self.bx  = int(eclick.ydata), int(eclick.xdata)
        self.height, self.width = int(erelease.ydata)-int(eclick.ydata), int(erelease.xdata)-int(eclick.xdata)

        self.dif_flag = 1   
        self.update()
        
    def update(self):
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[2].cla()

        #self.ax[1].set_title("%d, %d, %d"%(self.pick_flag, self.roi_flag, self.dif_flag))

        if self.pick_flag:
            self.mask = np.zeros((self.sy, self.sx))
            self.ax[0].set_title("[x, y]=[%d, %d]"%(self.ind[1], self.ind[0]))
            self.mask[self.ind[0], self.ind[1]] = 0.5
            self.ax[0].imshow(self.int_img, cmap="gray")
            self.ax[0].imshow(self.mask, cmap="Reds", alpha=self.mask)
            self.ax[1].imshow(self.df_img, cmap="gray")
            self.ax[1].imshow(self.mask, cmap="Reds", alpha=self.mask)
    
            if self.log_scale == -1:
                self.log_scale_message = "False"
                self.ax[2].imshow(self.fdata[self.ind[0], self.ind[1]], cmap="gray")
                self.ax[2].set_title("log scale: %s"%(self.log_scale_message))
            else:
                self.log_scale_message = "True"
                self.ax[2].imshow(np.log(self.fdata[self.ind[0], self.ind[1]]), cmap="gray")
                self.ax[2].set_title("log scale: %s"%(self.log_scale_message))

            if self.dif_flag:
                self.df_img = np.sum(self.fdata[:, :, self.by:self.by+self.height, self.bx:self.bx+self.width], axis=(2,3))
                self.df_mask = np.zeros((self.dsy, self.dsx))
                self.df_mask[self.by:self.by+self.height, self.bx:self.bx+self.width] = 0.5
                self.ax[1].imshow(self.df_img, cmap="gray")
                self.ax[1].imshow(self.mask, cmap="Reds", alpha=self.mask)
                self.ax[2].imshow(self.df_mask, cmap="Reds", alpha=self.df_mask)
                    
        if self.roi_flag:
            self.mask = np.zeros((self.sy, self.sx))
            self.ax[0].set_title("[top, bottom, right, left]=[%d, %d, %d, %d]"%(self.by_, self.by_+self.height_, self.bx_, self.bx_+self.width_))
            self.mask[self.by_:self.by_+self.height_, self.bx_:self.bx_+self.width_] = 0.5
            self.ax[0].imshow(self.int_img, cmap="gray")
            self.ax[0].imshow(self.mask, cmap="Reds", alpha=self.mask)
            self.ax[1].imshow(self.df_img, cmap="gray")
            self.ax[1].imshow(self.mask, cmap="Reds", alpha=self.mask)
                
            if self.log_scale == -1:
                self.log_scale_message = "False"
                self.ax[2].imshow(np.sum(self.fdata[self.by_:self.by_+self.height_, self.bx_:self.bx_+self.width_], axis=(0, 1)), cmap="gray")
                self.ax[2].set_title("log scale: %s"%(self.log_scale_message))
            else:
                self.log_scale_message = "True"
                self.ax[2].imshow(np.log(np.sum(self.fdata[self.by_:self.by_+self.height_, self.bx_:self.bx_+self.width_], axis=(0, 1))), cmap="gray")
                self.ax[2].set_title("log scale: %s"%(self.log_scale_message))

            if self.dif_flag:
                self.df_img = np.sum(self.fdata[:, :, self.by:self.by+self.height, self.bx:self.bx+self.width], axis=(2,3))
                self.df_mask = np.zeros((self.dsy, self.dsx))
                self.df_mask[self.by:self.by+self.height, self.bx:self.bx+self.width] = 0.5
                self.ax[1].imshow(self.df_img, cmap="gray")
                self.ax[1].imshow(self.mask, cmap="Reds", alpha=self.mask)
                self.ax[2].imshow(self.df_mask, cmap="Reds", alpha=self.df_mask)
    
        self.ax[0].axis("off")
        self.ax[1].axis("off")
        self.ax[2].axis("off")
    
        self.fig.canvas.draw()


class threed_viewer:
    def __init__(self, fig, ax, fdata, x_scale=1, x_unit="NA"):
        self.fig = fig
        self.ax = ax
        self.fdata = fdata
        self.ind = np.zeros(2).astype(np.int16)
        
        self.sy, self.sx, self.sz = fdata.shape
        self.log_scale = -1
        self.log_scale_message = "False"
        self.ax[0].set_title("[x, y]=[%d, %d]"%(self.ind[1], self.ind[0]))

        self.whole_sum = -1
        self.whole_sum_message = "False"

        if x_scale == 1:
            self.x_range = np.arange(self.sz)

        else:
            self.x_range = np.arange(self.sz) * x_scale

        self.x_unit = x_unit
        self.x_scale = x_scale
        
        self.int_img = np.sum(fdata, axis=2)
        mask = np.zeros((self.sy, self.sx))
        mask[self.ind[0], self.ind[1]] = 1
        
        self.ax[0].imshow(self.int_img, cmap="gray")
        self.ax[0].imshow(mask, cmap="Reds", alpha=mask)
        self.ax[0].axis("off")
        
        self.box = RectangleSelector(self.ax[2], self.onselect)
        self.by, self.bx, self.height, self.width = 0, 0, 0, self.sz
        self.df_img = np.sum(self.fdata[:, :, self.bx:self.bx+self.width], axis=2)
        self.ax[1].imshow(self.df_img, cmap="gray")
        self.ax[1].axis("off")
        
        self.ax[2].set_title("log scale: %s, sum: %s"%(self.log_scale_message, self.whole_sum_message))

        if self.whole_sum == -1:
            if self.log_scale == -1:
                self.ax[2].plot(self.x_range, self.fdata[self.ind[0], self.ind[1]], "k-")
            else:
                self.ax[2].plot(self.x_range, np.log(self.fdata[self.ind[0], self.ind[1]]), "k-")

        else:
            if self.log_scale == -1:
                self.ax[2].plot(self.x_range, np.sum(self.fdata, axis=(0, 1)), "k-")
            else:
                self.ax[2].plot(self.x_range, np.log(np.sum(self.fdata, axis=(0, 1)), "k-"))
        
        self.ax[2].set_xlabel(self.x_unit)
        self.ax[2].grid()
        
    def on_press(self, event):
        if event.key == "up":
            if self.ind[0] != 0:
                self.ind[0] -= 1
        elif event.key == "down":
            if self.ind[0] != self.sy:
                self.ind[0] += 1
        elif event.key == "right":
            if self.ind[1] != self.sx:
                self.ind[1] += 1
        elif event.key == "left":
            if self.ind[1] != 0:
                self.ind[1] -= 1
        elif event.key == "l":
            self.log_scale *= -1
        elif event.key == "t":
            self.whole_sum *= -1
            
        self.update()
        
    def on_pick(self, event):
        if event.inaxes == self.ax[0]:
            my, mx = int(event.ydata), int(event.xdata)
            self.ind[0] = my
            self.ind[1] = mx
            self.update()
            
        else:
            return True
        
    def onselect(self, eclick, erelease):
        self.by, self.bx  = eclick.ydata, eclick.xdata
        self.height, self.width = erelease.ydata-eclick.ydata, erelease.xdata-eclick.xdata
        
        self.update()
        
    def update(self):
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[2].cla()
        
        self.ax[0].set_title("[x, y]=[%d, %d]"%(self.ind[1], self.ind[0]))
        
        mask = np.zeros((self.sy, self.sx))
        mask[self.ind[0], self.ind[1]] = 1
        
        self.ax[0].imshow(self.int_img, cmap="gray")
        self.ax[0].imshow(mask, cmap="Reds", alpha=mask)
        self.ax[0].axis("off")
        
        self.df_img = np.sum(self.fdata[:, :, int(self.bx/self.x_scale):int((self.bx+self.width)/self.x_scale)], axis=2)
        self.ax[1].imshow(self.df_img, cmap="gray")
        self.ax[1].axis("off")

        if self.whole_sum == -1:
            self.whole_sum_message = "False"
            if self.log_scale == -1:
                self.log_scale_message = "False"
                plot_graph = self.fdata[self.ind[0], self.ind[1]]
                self.ax[2].plot(self.x_range, plot_graph, "k-")
                self.ax[2].set_title("log scale: %s, sum: %s"%(self.log_scale_message, self.whole_sum_message))
            else:
                self.log_scale_message = "True"
                plot_graph = np.log(self.fdata[self.ind[0], self.ind[1]])
                self.ax[2].plot(self.x_range, plot_graph, "k-")
                self.ax[2].set_title("log scale: %s, sum: %s"%(self.log_scale_message, self.whole_sum_message))

        else:
            self.whole_sum_message = "True"
            if self.log_scale == -1:
                self.log_scale_message = "False"
                plot_graph = np.sum(self.fdata, axis=(0, 1))
                self.ax[2].plot(self.x_range, plot_graph, "k-")
                self.ax[2].set_title("log scale: %s, sum: %s"%(self.log_scale_message, self.whole_sum_message))
            else:
                self.log_scale_message = "True"
                plot_graph = np.log(np.sum(self.fdata, axis=(0, 1)))
                self.ax[2].plot(self.x_range, plot_graph, "k-")
                self.ax[2].set_title("log scale: %s, sum: %s"%(self.log_scale_message, self.whole_sum_message))            

        self.ax[2].fill_between([self.bx, self.bx+self.width], np.max(plot_graph), alpha=0.5, color="orange")
        self.ax[2].grid()
        self.ax[2].set_xlabel(self.x_unit)
        self.fig.canvas.draw()


class slice_viewer:
    def __init__(self, ax, X):
        self.ax = ax
        
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0

        self.ax.set_title("Series No.%d"%(self.ind+1))
        
        self.im = ax.imshow(self.X[self.ind], cmap="gray")
        self.ax.axis("off")
        self.update()

    def on_press(self, event):
        if event.key == "up" or event.key == "right":
            self.ind = (self.ind + 1) % self.slices
        elif event.key == "down" or event.key == "left":
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_title("tilt series No.%d"%(self.ind+1))
        self.im.axes.figure.canvas.draw()


class FourDSTEM_process():
    def __init__(self, file_adr, scan_per_pixel=1, dp_per_pixel=1, scan_unit="nm", k_unit="1/nm", f_shape=None, datatype=np.float32, visual=True):

        self.file_adr = file_adr
        if f_shape != None:
            o_shape = [f_shape[0], f_shape[1], f_shape[2]+2, f_shape[3]]

        
        if file_adr[-3:] == "raw":
            self.f_stack = load_binary_4D_stack(file_adr, datatype, o_shape, f_shape, log_scale=False)
            self.f_stack = np.flip(self.f_stack, axis=2)
            self.f_stack = np.nan_to_num(self.f_stack)
            
        elif file_adr[-3:] == "tif" or file_adr[:-4] == "tiff":
            self.f_stack = tifffile.imread(file_adr)
            self.f_stack  = np.nan_to_num(self.f_stack)
            
        else:
            print("The format of the file is not supported here")
            
        print(self.f_stack.shape)
        print(self.f_stack.min(), self.f_stack.max())
        print(self.f_stack.mean())

        self.f_stack = self.f_stack.clip(min=0.0)


        self.original_stack = self.f_stack
        self.original_shape = self.f_stack.shape
        self.original_mean_dp = np.mean(self.original_stack, axis=(0, 1))
        self.scan_per_pixel = scan_per_pixel
        self.dp_per_pixel = dp_per_pixel
        self.scan_unit = scan_unit
        self.k_unit = k_unit
        
        self.intensity_integration_map = np.sum(self.f_stack, axis=(2, 3))

        self.ct = None

        if visual:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(self.intensity_integration_map, cmap="inferno")
            ax[0].axis("off")
            ax[1].imshow(self.original_mean_dp, cmap="jet")
            ax[1].axis("off")
            fig.tight_layout()


    def spike_remove(self, percent_thresh, mode, apply_remove=False):
        threshold = np.percentile(self.intensity_integration_map, percent_thresh)
        if mode == "upper":
            spike_ind = np.where(self.intensity_integration_map > threshold)
        elif mode == "lower":
            spike_ind = np.where(self.intensity_integration_map < threshold)
        else:
            print("Wrong mode!")
            return

        print("threshold value = %f"%threshold)
        print("number of abnormal pixels = %d"%len(spike_ind[0]))
        
        self.spike_replaced = self.intensity_integration_map.copy()
        self.spike_replaced[spike_ind] = np.sum(self.original_mean_dp)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(self.spike_replaced, cmap="inferno")
        ax.axis("off")
        fig.tight_layout()
        
        
        if apply_remove:
            self.original_stack[spike_ind] = self.original_mean_dp.copy()
            self.original_mean_dp = np.mean(self.original_stack, axis=(0, 1))

    def find_center(self, cbox_edge=30, center_pos=None, visual=True):

        if center_pos:
            self.ct = center_pos

        else:
            cbox_outy = int(self.original_mean_dp.shape[0]/2 - cbox_edge/2)
            cbox_outx = int(self.original_mean_dp.shape[1]/2 - cbox_edge/2)
            center_box = self.original_mean_dp[cbox_outy:-cbox_outy, cbox_outx:-cbox_outx]
            Y, X = np.indices(center_box.shape)
            com_y = np.sum(center_box * Y) / np.sum(center_box)
            com_x = np.sum(center_box * X) / np.sum(center_box)
            self.ct = [np.around(com_y+cbox_outy), np.around(com_x+cbox_outx)]

        if visual:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(self.original_mean_dp, cmap="jet")
            ax.axis("off")
            ax.scatter(self.ct[1], self.ct[0], s=15, c="k")
            fig.tight_layout()
            

        return self.ct


    def disk_extract(self, buffer_size=0, visual=True):
        grad = np.gradient(self.original_mean_dp)
        grad_map = grad[0]**2 + grad[1] **2
        grad_map = grad_map / np.max(grad_map)
        
        max_ind = np.unravel_index(np.argmax(grad_map, axis=None), grad_map.shape)
        self.least_R = ((max_ind[0]-self.ct[0])**2 + (max_ind[1]-self.ct[1])**2)**(1/2)
        
        print("radius of the BF disk = %.2f mrad"%(self.dp_per_pixel*self.least_R))
        
        self.cropped_size = np.around(self.least_R + buffer_size).astype(int)

        if self.cropped_size > self.ct[0] or self.cropped_size > self.ct[1]:
            self.cropped_size = np.min(self.ct).astype(int)
  
        print("radius of the RoI = %.2f mrad"%(self.dp_per_pixel*self.cropped_size))
        
        h_si = np.floor(self.ct[0]-self.cropped_size).astype(int)
        h_fi = np.ceil(self.ct[0]+self.cropped_size).astype(int)
        w_si = np.floor(self.ct[1]-self.cropped_size).astype(int)
        w_fi = np.ceil(self.ct[1]+self.cropped_size).astype(int)
        
        self.c_ct = [self.cropped_size, self.cropped_size]
        
        self.c_stack = self.original_stack[:, :, h_si:h_fi, w_si:w_fi].copy()
        self.c_shape = self.c_stack.shape
        self.c_mean_dp = np.mean(self.c_stack, axis=(0, 1))

        if visual:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(self.c_mean_dp, cmap="jet")
            ax.scatter(self.c_ct[1], self.c_ct[0], s=15, c="k")
            ax.axis("off")
            
            print(self.c_mean_dp.shape)
            print(self.least_R)


    def virtual_stem(self, BF, ADF, visual=True):
        self.BF_detector = radial_indices(self.original_mean_dp.shape, BF, self.dp_per_pixel, center=self.ct)
        self.BF_stem = np.sum(np.multiply(self.original_stack, self.BF_detector), axis=(2, 3))
        
        self.ADF_detector = radial_indices(self.original_mean_dp.shape, ADF, self.dp_per_pixel, center=self.ct)
        self.ADF_stem = np.sum(np.multiply(self.original_stack, self.ADF_detector), axis=(2, 3))

        if visual:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0][0].imshow(self.original_mean_dp, cmap="jet")
            ax[0][0].imshow(self.BF_detector, cmap="gray", alpha=0.5)
            ax[0][0].scatter(self.ct[1], self.ct[0], s=15, c="k")
            ax[0][0].set_title("BF detector")
            ax[0][0].axis("off")
            ax[0][1].imshow(self.BF_stem, cmap="inferno", origin="lower")
            ax[0][1].set_title("BF-STEM image")
            ax[0][1].axis("off")
            ax[1][0].imshow(self.original_mean_dp, cmap="jet")
            ax[1][0].imshow(self.ADF_detector, cmap="gray", alpha=0.5)
            ax[1][0].scatter(self.ct[1], self.ct[0], s=15, c="k")
            ax[1][0].set_title("ADF detector")
            ax[1][0].axis("off")
            ax[1][1].imshow(self.ADF_stem, cmap="inferno", origin="lower")
            ax[1][1].set_title("ADF-STEM image")
            ax[1][1].axis("off")
            fig.tight_layout()
            

    def DPC(self, correct_rotation=True, n_theta=100, hpass=0.05, lpass=0.05, visual=True):
        """
        Hachtel, J.A., J.C. Idrobo, and M. Chi, Adv Struct Chem Imaging, 2018. 4(1): p. 10. (https://github.com/hachteja/GetDPC)
        Lazic, I., E.G.T. Bosch, and S. Lazar, Ultramicroscopy, 2016. 160: p. 265-280.
        Savitzky, B.H., et al., arXiv preprint arXiv:2003.09523, 2020. (https://github.com/py4dstem/py4DSTEM)
        """
        
        Y, X = np.indices(self.c_mean_dp.shape)
        self.ysh = np.sum(self.c_stack * Y, axis=(2, 3)) / np.sum(self.c_stack, axis=(2, 3)) - self.c_ct[0]
        self.xsh = np.sum(self.c_stack * X, axis=(2, 3)) / np.sum(self.c_stack, axis=(2, 3)) - self.c_ct[1]
        
        self.ysh -= np.mean(self.ysh)
        self.xsh -= np.mean(self.xsh)
        
        if correct_rotation:
            theta = np.linspace(-np.pi/2, np.pi/2, n_theta, endpoint=True)
            self.div = []
            self.curl = []
            for t in theta:
                r_ysh = self.xsh * np.sin(t) + self.ysh * np.cos(t)
                r_xsh = self.xsh * np.cos(t) - self.ysh * np.sin(t)

                gyy, gyx = np.gradient(r_ysh)
                gxy, gxx = np.gradient(r_xsh)
                shift_divergence = gyy + gxx
                shift_curl = gyx - gxy

                self.div.append(np.mean(shift_divergence**2))
                self.curl.append(np.mean(shift_curl**2))
                
            self.c_theta = theta[np.argmin(self.curl)]
            tmp_ysh = self.xsh * np.sin(self.c_theta) + self.ysh * np.cos(self.c_theta)
            tmp_xsh = self.xsh * np.cos(self.c_theta) - self.ysh * np.sin(self.c_theta)
            
            self.ysh = tmp_ysh
            self.xsh = tmp_xsh
            
        self.E_mag = np.sqrt(self.ysh**2 + self.xsh**2)
        self.E_field_y = -self.ysh / np.max(self.E_mag)
        self.E_field_x = -self.xsh / np.max(self.E_mag)
        
        self.charge_density = np.gradient(self.E_field_y)[0] + np.gradient(self.E_field_x)[1]
        
        self.potential = get_icom(self.ysh, self.xsh, hpass, lpass)

        if visual:
            print("optimized angle =", self.c_theta*180/np.pi)
            fig, ax = plt.subplots(1, 4, figsize=(28, 7))
            ax[0].imshow(self.ADF_stem, cmap="inferno", origin="lower")
            ax[0].axis("off")
            ax[1].imshow(self.E_field_y, cmap="gray", origin="lower")
            ax[1].axis("off")
            ax[2].imshow(self.E_field_x, cmap="gray", origin="lower")
            ax[2].axis("off")
            ax[3].imshow(self.E_mag, cmap="inferno", origin="lower")
            ax[3].axis("off")
            fig.tight_layout()
            
            RY, RX = np.indices(self.c_shape[:2])
            fig, ax = plt.subplots(1, 3, figsize=(30, 10))
            ax[0].imshow(self.ADF_stem, cmap="gray", origin="lower")
            ax[0].quiver(RX.flatten(), RY.flatten(), self.E_field_x.flatten(), self.E_field_y.flatten(), color=cm.jet(mcolors.Normalize()(self.E_mag.flatten())))
            ax[0].axis("off")
            #ax[1].imshow(test.ADF_stem, cmap="gray")
            ax[1].imshow(self.charge_density, cmap="RdBu_r", origin="lower")
            ax[1].axis("off")
            #ax[2].imshow(test.ADF_stem, cmap="gray")
            ax[2].imshow(self.potential, cmap="inferno", origin="lower")
            ax[2].axis("off")
            fig.tight_layout()


    def symmetry_evaluation(self, angle, also_mirror=False, visual=True):
        """
        Krajnak, M. and J. Etheridge, Proc Natl Acad Sci U S A, 2020. 117(45): p. 27805-27810.
        """
        self.rotation_stack = []
        self.r_correl = np.zeros(self.original_shape[:2])
        self.m_correl = np.zeros(self.original_shape[:2])
        
        ri = radial_indices(self.c_mean_dp.shape, [0, self.cropped_size], 1, center=self.c_ct)
        
        angle = angle * np.pi/180
        alpha, beta = np.cos(angle), np.sin(angle)
        M = np.array([[alpha, beta, (1-alpha)*self.c_ct[1]-beta*self.c_ct[0]], 
                    [-beta, alpha, beta*self.c_ct[1]+(1-alpha)*self.c_ct[0]]])
        
        for i in range(self.original_shape[0]):
            for j in range(self.original_shape[1]):
                tmp_dp = self.c_stack[i,j,:,:].copy()
                newdata = np.multiply(rotation(tmp_dp, M), ri)
                self.rotation_stack.append(newdata)
                self.r_correl[i,j] = correlation(tmp_dp/np.max(tmp_dp), newdata)
                
                if also_mirror:
                    self.m_correl[i, j] = mirror(newdata, self.c_ct)
        
        self.rotation_stack = np.asarray(self.rotation_stack).reshape(self.c_shape)

        if visual:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.r_correl, cmap="inferno", origin="lower")
            ax[0].set_title("Rotation angle: {}".format(angle))
            ax[0].axis("off")
            ax[1].imshow(self.m_correl, cmap="inferno", origin="lower")
            ax[1].set_title("Mirror angle: {}".format(angle))
            ax[1].axis("off")
            fig.tight_layout()


    def rotational_average(self, rot_variance=True):
        
        self.radial_avg_stack, self.radial_var_stack = fourd_radial_transformation(self.original_stack, center=self.ct, also_variance=rot_variance)

    
    def cepstral(self, dCP=False, datatype=np.float32, rot_average=False, rot_variance=False):
        
        self.real_per_pixel = 1 / (self.dp_per_pixel * self.original_shape[2])
    
        self.ceps, self.dcp = cepstrum_transformation(self.original_stack.copy(), dCP, datatype)

        if rot_average:
            self.ceps_avg_stack, self.ceps_var_stack = fourd_radial_transformation(self.ceps, center=None, also_variance=rot_variance)        


    def show_4d_viewer(self, fdata):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("""1st figure (intensity map) : arrow keys or mouse left button to move the position
        2nd figure (virtual DF image)
        3rd figure (diffraction image) : press 'l' key to turn on or off log-scaling / drag to make a ROI (virtual obj aperture)""")

        self.tracker = fourd_viewer(fig, ax, fdata)

        fig.canvas.mpl_connect("key_press_event", self.tracker.on_press)
        fig.canvas.mpl_connect("button_press_event", self.tracker.on_pick)
        fig.tight_layout()


    def show_3d_viewer(self, fdata, x_scale=1, x_unit="NA"):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("""1st figure (intensity map) : arrow keys or mouse left button to move the position
        2nd figure (selected range intensity image)
        3rd figure (spectrum) : press 'l' key to turn on or off log-scaling / drag to make a ROI (select a range)""")

        self.tracker = threed_viewer(fig, ax, fdata, x_scale, x_unit)

        fig.canvas.mpl_connect("key_press_event", self.tracker.on_press)
        fig.canvas.mpl_connect("button_press_event", self.tracker.on_pick)
        fig.tight_layout()


def load_binary_4D_stack(img_adr, datatype, original_shape, final_shape, log_scale=False):
    stack = np.fromfile(img_adr, dtype=datatype)
    stack = stack.reshape(original_shape)
    if log_scale:
        stack = np.log(stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]])
    else:
        stack = stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]]
    return stack

def radial_indices(shape, radial_range, scale, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    r = np.hypot(y - center[0], x - center[1]) * scale
    ri = np.ones(r.shape)
    
    if len(np.unique(radial_range)) > 1:
        ri[np.where(r < radial_range[0])] = 0
        ri[np.where(r > radial_range[1])] = 0
        
    else:
        r = np.round(r)
        ri[np.where(r != round(radial_range[0]))] = 0
    
    return ri

def segmented_DPC(xsh, ysh, correct_rotation=True, n_theta=100, hpass=0.05, lpass=0.05, visual=True):

    if correct_rotation:
        theta = np.linspace(-np.pi/2, np.pi/2, n_theta, endpoint=True)
        div = []
        curl = []
        for t in theta:
            r_ysh = xsh * np.sin(t) + ysh * np.cos(t)
            r_xsh = xsh * np.cos(t) - ysh * np.sin(t)

            gyy, gyx = np.gradient(r_ysh)
            gxy, gxx = np.gradient(r_xsh)
            shift_divergence = gyy + gxx
            shift_curl = gyx - gxy

            div.append(np.mean(shift_divergence**2))
            curl.append(np.mean(shift_curl**2))
            
        c_theta = theta[np.argmin(curl)]
        tmp_ysh = xsh * np.sin(c_theta) + ysh * np.cos(c_theta)
        tmp_xsh = xsh * np.cos(c_theta) - ysh * np.sin(c_theta)
        
        ysh = tmp_ysh
        xsh = tmp_xsh

        print("optimized rotation angle: ", c_theta*180/np.pi)
        
    E_mag = np.sqrt(ysh**2 + xsh**2)
    E_field_y = -ysh / np.max(E_mag)
    E_field_x = -xsh / np.max(E_mag)
    
    charge_density = np.gradient(E_field_y)[0] + np.gradient(E_field_x)[1]
    
    potential = get_icom(ysh, xsh, hpass, lpass)

    if visual:
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(E_field_y, cmap="gray", origin="lower")
        ax[0].axis("off")
        ax[1].imshow(E_field_x, cmap="gray", origin="lower")
        ax[1].axis("off")
        ax[2].imshow(E_mag, cmap="inferno", origin="lower")
        ax[2].axis("off")
        fig.tight_layout()

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(charge_density, cmap="RdBu_r", origin="lower")
        ax[0].axis("off")
        ax[1].imshow(potential, cmap="inferno", origin="lower")
        ax[1].axis("off")
        fig.tight_layout()

    return E_mag, E_field_x, E_field_y, charge_density, potential


def get_icom(ysh, xsh, hpass=0, lpass=0):
    
    FT_ysh = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ysh)))
    FT_xsh = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(xsh)))
    
    ky = np.fft.fftshift(np.fft.fftfreq(FT_ysh.shape[0])).reshape(-1, 1)
    kx = np.fft.fftshift(np.fft.fftfreq(FT_xsh.shape[1])).reshape(1, -1)

    k2 = ky**2 + kx**2
    zero_ind = np.where(k2 == 0.0)
    k2[zero_ind] = 1.0

    FT_phase = (FT_ysh*ky + FT_xsh*kx) / (2*np.pi*1j*(hpass+k2+lpass*k2))
    FT_phase[zero_ind] = 0.0

    Iicom = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(FT_phase))))
    
    return Iicom

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def indices_at_r(shape, radius, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(y - center[0], x - center[1])
    r = np.around(r)
    
    ri = np.where(r == radius)
    
    angle_arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            angle_arr[i, j] = np.angle(complex(x[i, j]-center[1], y[i, j]-center[0]), deg=True)
            
    angle_arr = angle_arr + 180
    angle_arr = np.around(angle_arr)
    
    ai = np.argsort(angle_arr[ri])
    r_sort = (ri[1][ai], ri[0][ai])
    a_sort = np.sort(angle_arr[ri])
        
    return r_sort, a_sort

def local_similarity(var_map, f_flat, w_size, rows, cols):
    new_shape = (len(rows), len(cols))
    
    surr_avg = []
    surr_std = []
    surr_dif = []
    for i in rows:
        for j in cols:
            local_region = var_map[i:i+w_size, j:j+w_size].flatten()
            
            if np.max(local_region) != 0.0:
                local_region = local_region / np.max(local_region)
            else:
                local_region = local_region * 0.0
            
            temp_avg = np.mean(local_region)
            temp_std = np.std(local_region)
            surr_avg.append(temp_avg)
            surr_std.append(temp_std)
            diff_mse = np.sum(np.square(local_region - local_region[int(w_size**2/2)]))/(w_size**2-1)
            surr_dif.append(diff_mse)
            
    surr_avg = np.asarray(surr_avg).reshape(new_shape)
    surr_std = np.asarray(surr_std).reshape(new_shape)
    surr_dif = np.asarray(surr_dif).reshape(new_shape)
    
    dp_mse = []
    dp_ssim = []
    for i in rows:
        for j in cols:        
            local_region = f_flat[i:i+w_size, j:j+w_size].reshape(w_size**2, -1)
            ref_dp = local_region[int(w_size**2/2)]
            local_region = np.delete(local_region, int(w_size**2/2), axis=0)
            tmp_mse = []
            tmp_ssim = []
            for fdp in local_region:
                tmp_mse.append(mean_squared_error(ref_dp/np.max(ref_dp), fdp/np.max(fdp)))
                tmp_ssim.append(ssim(ref_dp/np.max(ref_dp), fdp/np.max(fdp)))
                
            dp_mse.append(np.mean(tmp_mse))
            dp_ssim.append(np.mean(tmp_ssim))
            
    dp_mse = np.asarray(dp_mse).reshape(new_shape)
    dp_ssim = np.asarray(dp_ssim).reshape(new_shape)
    
    
    return surr_avg, surr_std, surr_dif, dp_mse, dp_ssim, new_shape

def local_var_similarity(var_map, w_size, stride):
    var_map = np.asarray(var_map)
    rows = range(0, var_map.shape[0]-w_size+1, stride)
    cols = range(0, var_map.shape[1]-w_size+1, stride)
    new_shape = (len(rows), len(cols))
    
    surr_avg = []
    surr_std = []
    surr_dif = []
    for i in rows:
        for j in cols:
            local_region = var_map[i:i+w_size, j:j+w_size].flatten()
            
            if np.max(local_region) != 0.0:
                local_region = local_region / np.max(local_region)
            else:
                local_region = local_region * 0.0
            
            temp_avg = np.mean(local_region)
            temp_std = np.std(local_region)
            surr_avg.append(temp_avg)
            surr_std.append(temp_std)
            diff_mse = np.sum(np.square(local_region - local_region[int(w_size**2/2)]))/(w_size**2-1)
            surr_dif.append(diff_mse)
            
    surr_avg = np.asarray(surr_avg).reshape(new_shape)
    surr_std = np.asarray(surr_std).reshape(new_shape)
    surr_dif = np.asarray(surr_dif).reshape(new_shape)
    
    return surr_avg, surr_std, surr_dif, new_shape

def local_DP_similarity(f_flat, w_size, stride):
    f_flat = np.asarray(f_flat)
    rows = range(0, f_flat.shape[0]-w_size+1, stride)
    cols = range(0, f_flat.shape[1]-w_size+1, stride)
    new_shape = (len(rows), len(cols))
    
    dp_mse = []
    dp_ssim = []
    for i in rows:
        for j in cols:        
            local_region = f_flat[i:i+w_size, j:j+w_size].reshape(w_size**2, -1)
            ref_dp = local_region[int(w_size**2/2)]
            local_region = np.delete(local_region, int(w_size**2/2), axis=0)
            tmp_mse = []
            tmp_ssim = []
            for fdp in local_region:
                tmp_mse.append(mean_squared_error(ref_dp/np.max(ref_dp), fdp/np.max(fdp)))
                tmp_ssim.append(ssim(ref_dp/np.max(ref_dp), fdp/np.max(fdp)))
                
            dp_mse.append(np.mean(tmp_mse))
            dp_ssim.append(np.mean(tmp_ssim))
            
    dp_mse = np.asarray(dp_mse).reshape(new_shape)
    dp_ssim = np.asarray(dp_ssim).reshape(new_shape)
    
    return dp_mse, dp_ssim, new_shape

def radial_stats(image, center=None, var=True):
   
    y, x = np.indices(image.shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
        
    r = np.hypot(y - center[0], x - center[1])
    #plt.imshow(r, cmap="Accent")
    #

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = np.around(r_sorted)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    #print(nr)
    
    csim = np.cumsum(i_sorted, dtype=float)
    sq_csim = np.cumsum(np.square(i_sorted), dtype=float)
    radial_avg  = (csim[rind[1:]] - csim[rind[:-1]]) / nr
    
    if var:    
        avg_square = np.square(radial_avg)
        square_avg = (sq_csim[rind[1:]] - sq_csim[rind[:-1]]) / nr
        mask = avg_square.copy()
        mask[np.where(avg_square==0)] = 1.0
        radial_var = (square_avg - avg_square) / mask
        return radial_avg, radial_var
    
    else:
        return radial_avg, None

def fourd_radial_transformation(fdata, center=None, also_variance=False):
    radial_avg_stack = []
    radial_var_stack = []
    len_profile = []

    data_shape = fdata.shape

    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            ravg, rvar = radial_stats(fdata[i, j], center=center, var=also_variance)
            len_profile.append(len(ravg))
            radial_avg_stack.append(ravg)
            if also_variance:
                radial_var_stack.append(rvar)

    if len(np.unique(len_profile)) > 1:
        print(np.unique(len_profile))
        shortest = np.min(len_profile)
        for i in range(len(radial_avg_stack)):
            radial_avg_stack[i] = radial_avg_stack[i][:shortest]
            if also_variance:
                radial_var_stack[i] = radial_var_stack[i][:shortest]

    radial_avg_stack = np.asarray(radial_avg_stack).reshape(data_shape[0], data_shape[1], -1)
    if also_variance:
        radial_var_stack = np.asarray(radial_var_stack).reshape(data_shape[0], data_shape[1], -1)

    return radial_avg_stack, radial_var_stack


def cepstrum_transformation(img, dCP=False, data_type=np.float32):
    
    img[img==0] = 1.0
    mean_dp = np.mean(img, axis=(0, 1))
    fft2_mean = np.fft.fftshift(np.fft.fft2(np.log(mean_dp)))
    fft2_ = np.fft.fftshift(np.fft.fft2(np.log(img.astype(data_type)), axes=(2, 3)), axes=(2,3))

    if dCP:
        return np.abs(fft2_).astype(data_type), np.abs(fft2_-fft2_mean[np.newaxis, np.newaxis, :, :]).astype(data_type)
    else:
        return np.abs(fft2_).astype(data_type), None

def correlation(dat2d, newdata):
    # return correlation value
    dat1d = dat2d.flatten()
    new1d = newdata.flatten()
    correlation = np.correlate(dat1d, new1d)

    return correlation[0]

def rotation(data, RM):
    
    rotated = cv2.warpAffine(data, RM, data.shape)
                
    return rotated/np.max(rotated)

def mirror(data, center):
    #Input data : 2d # Return : Correlation value
    data1 = data[:, :int(center[1]-1)]
    data2 = np.flip(data, axis=1)[:, :int(center[1]-1)]
    value = correlation(data1, data2)

    return value