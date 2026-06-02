import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as pch
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
            self.ax[2].imshow(np.log(np.clip(self.fdata[self.ind[0], self.ind[1]], 1e-9, None)), cmap="gray")
        self.df_mask = np.zeros((self.dsy, self.dsx))
        self.df_mask[self.by:self.by+self.height, self.bx:self.bx+self.width] = 0.5
        self.ax[2].imshow(self.df_mask, cmap="Reds", alpha=self.df_mask)
        self.ax[2].axis("off")

    def on_press(self, event):
        if event.key == "up":
            if self.ind[0] != 0:
                self.ind[0] -= 1
        elif event.key == "down":
            if self.ind[0] != self.sy - 1:
                self.ind[0] += 1
        elif event.key == "right":
            if self.ind[1] != self.sx - 1:
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
                self.ax[2].imshow(np.log(np.clip(self.fdata[self.ind[0], self.ind[1]], 1e-9, None)), cmap="gray")
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
                self.ax[2].imshow(np.log(np.clip(np.sum(self.fdata[self.by_:self.by_+self.height_, self.bx_:self.bx_+self.width_], axis=(0, 1)), 1e-9, None)), cmap="gray")
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
                self.ax[2].plot(self.x_range, np.log(np.clip(self.fdata[self.ind[0], self.ind[1]], 1e-9, None)), "k-")
        else:
            if self.log_scale == -1:
                self.ax[2].plot(self.x_range, np.sum(self.fdata, axis=(0, 1)), "k-")
            else:
                self.ax[2].plot(self.x_range, np.log(np.clip(np.sum(self.fdata, axis=(0, 1)), 1e-9, None)), "k-")
        
        self.ax[2].set_xlabel(self.x_unit)
        self.ax[2].grid()
        
    def on_press(self, event):
        if event.key == "up":
            if self.ind[0] != 0:
                self.ind[0] -= 1
        elif event.key == "down":
            if self.ind[0] != self.sy - 1:
                self.ind[0] += 1
        elif event.key == "right":
            if self.ind[1] != self.sx - 1:
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
                plot_graph = np.log(np.clip(self.fdata[self.ind[0], self.ind[1]], 1e-9, None))
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
                plot_graph = np.log(np.clip(np.sum(self.fdata, axis=(0, 1)), 1e-9, None))
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
        self.im = ax.imshow(self.X[self.ind], cmap="inferno")
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
