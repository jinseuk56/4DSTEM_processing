#!/usr/bin/env python3

import typing
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from pylab import cm
import PIL
from PIL import Image
import os
import pickle
import sys
from scipy.ndimage import center_of_mass
from scipy.ndimage import shift
import multiprocessing
from multiprocessing import Process, Array
#from IPython import get_ipython
import ctypes as c
from numpy.linalg import inv
from skimage.restoration import unwrap_phase
import tifffile


try:
    import pixstem.api as ps
except:
    ps = None
    print('cannot load pixstem package...')
    print('4D plot not possible')

    
#from mpl_toolkits.axes_grid1 import ImageGrid

def load_binary_4D_stack(img_adr, datatype, original_shape, final_shape, log_scale=False):
    stack = np.fromfile(img_adr, dtype=datatype)
    stack = stack.reshape(original_shape)
    if log_scale:
        stack = np.log(stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]])
    else:
        stack = stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]]
    return stack

def spike_remove(data, percent_thresh, mode):

    pacbed = np.mean(data, axis=(0, 1))
    intensity_integration_map = np.sum(data, axis=(2, 3))

    threshold = np.percentile(intensity_integration_map, percent_thresh)
    if mode == "upper":
        spike_ind = np.where(intensity_integration_map > threshold)
    elif mode == "lower":
        spike_ind = np.where(intensity_integration_map < threshold)
    else:
        print("Wrong mode!")
        return

    print("threshold value = %f"%threshold)
    print("number of abnormal pixels = %d"%len(spike_ind[0]))

    data[spike_ind] = pacbed.copy()

    return data


class Data4D():
    def __init__(self,parfile):
        self.init_parameters(parfile)
        self.setup_scanning_parameters()
        #self.center_ronchigrams()
        #self.truncate_ronchigram()
        
    def save_metadata(self):
        dict1 = {'step_size':self.step_size,
                 'step_size_x_reciprocal':self.scan_angle_step_x*1000,
                 'step_size_y_reciprocal':self.scan_angle_step_y*1000,
                 'offset_x_reciprocal': self.scan_angles_x[0],
                 'offset_y_reciprocal': self.scan_angles_y[0]}
        a_file = open(self.path+'data4D_meta.pkl', 'wb')

        pickle.dump(dict1, a_file)
        
        a_file.close()
        
        
    def plot_4D(self,log=False):
        
        #get_ipython().run_line_magic('matplotlib', 'auto') 
        #dict0 = { 'name':'x', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict1 = { 'name':'y', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict2 = { 'name':'angle x', 'units':'rad', 'scale':self.scan_angle_step_x, 'offset':self.scan_angles_x[0]}
        #dict3 = { 'name':'angle <', 'units':'rad', 'scale':self.scan_angle_step_y, 'offset':self.scan_angles_y[0]}
        if ps==None:
            print('4D plot not possible, required pixstem')
            return
        if log:
            s = ps.PixelatedSTEM(np.log(self.data_4D))
        else:
            s = ps.PixelatedSTEM(self.data_4D)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].units = 'A'
        s.axes_manager[0].scale = self.step_size
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].units = 'A'
        s.axes_manager[1].scale = self.step_size
        s.axes_manager[2].name = 'angle x'
        s.axes_manager[2].units = 'mrad'
        s.axes_manager[2].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[2].offset = self.scan_angles_x[0]
        s.axes_manager[3].name = 'angle y'
        s.axes_manager[3].units = 'mrad'
        s.axes_manager[3].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[3].offset = self.scan_angles_y[0]
        s.metadata.General.title = '4D data'
        
        s.plot(cmap='viridis')
        
        self.pixstem = s
       

        return s
    
    def apply_dose(self, dose):
        dose_perpixel=dose*self.step_size**2
        for i in range (self.data_4D.shape[0]):
            for j in range (self.data_4D.shape[1]):
                self.data_4D[i,j,:,:]=np.random.poisson(self.data_4D[i,j,:,:]/(self.data_4D[i,j,:,:].sum())*dose_perpixel)
    
    def Wavelength(self, Voltage): # the unit of voltage should be kV and returned unit of wavelength is angstrom
        """Compute the wave length of the electron 
       
           parameters Voltage: Microscope accelerating voltage.
         """
        emass = 510.99906
        hc = 12.3984244
        wavelength = hc/np.sqrt(Voltage * (2*emass + Voltage))
        return wavelength
    
    def init_parameters(self, parfile):
        par_dictionary = {}

        file = open(parfile)
        for line in file:
            if line.startswith('##'):
                continue
            split_line = line.rstrip().split('\t')
            print(split_line)

            if len(split_line)!=2:
                continue
            key, value = split_line
            par_dictionary[key] = value
        self.file = par_dictionary.get('file','')
        self.path = os.path.abspath(parfile+'/..')+'/'
        os.chdir(self.path)
        print(self.path)


        datatype = "float32"
        f_shape = [256, 256, 128, 128] # the shape of the 4D-STEM data [scanning_y, scanning_x, DP_y, DP_x]
        o_shape = [f_shape[0], f_shape[1], f_shape[2]+2, f_shape[3]]

        if self.file[-3:] == "raw":
            f_stack = load_binary_4D_stack(self.file, datatype, o_shape, f_shape, log_scale=False)
            f_stack = np.flip(f_stack, axis=2)
            f_stack = np.nan_to_num(f_stack)
            
        elif self.file[-3:] == "tif" or self.file[:-4] == "tiff":
            f_stack = tifffile.imread(self.file)
            f_stack  = np.nan_to_num(f_stack )
            
        else:
            print("The format of the file is not supported here")
            
        print(f_stack.shape)
        print(f_stack.min(), f_stack.max())
        print(f_stack.mean())

        f_stack = spike_remove(f_stack, percent_thresh=0.01, mode="lower")
        f_stack = f_stack.clip(min=0.0)

        self.data_4D=f_stack       
        self.aperturesize = float(par_dictionary.get('aperture',0))
        self.voltage = float(par_dictionary.get('voltage'))
        self.step_size = float(par_dictionary.get('stepsize',1))
        self.rotation_angle_deg  = -float(par_dictionary.get('rotation',0))
        self.rotation_angle = self.rotation_angle_deg/180*np.pi
        self.method  = par_dictionary.get('method','ssb')

        # choose any example data for plotting
        self.workers  = int(par_dictionary.get('workers',1))
        self.threshold = float(par_dictionary.get('threshold',0.3))
        self.wave_len = self.Wavelength(self.voltage)
    
    def setup_scanning_parameters(self):
        self.scan_row = self.data_4D.shape[0]
        self.scan_col = self.data_4D.shape[1]
        self.scan_x_len = self.step_size*(self.scan_col-1)
        self.scan_y_len = self.step_size*(self.scan_row-1)
        self.scan_angle_step_x= self.wave_len/self.scan_x_len
        self.scan_angle_step_y= self.wave_len/self.scan_y_len
        #now set scanning reciprocal space spatial frequency and angles.
        self.scan_angles_x=np.arange(self.scan_col)-np.fix(self.scan_col/2)
        self.scan_angles_y=np.arange(self.scan_row)-np.fix(self.scan_row/2)
        self.scan_angles_x *= self.scan_angle_step_x;
        self.scan_angles_y *= self.scan_angle_step_y;
        print("angle step in the x direction is: ", self.scan_angle_step_x)
        
    def center_ronchigrams(self):
        Ronchi_mean=np.mean(self.data_4D,(0,1))
        com = center_of_mass(Ronchi_mean)
        for i in range(self.data_4D.shape[0]):
            for j in range (self.data_4D.shape[1]):
                self.data_4D[i,j,:,:] = shift(self.data_4D[i,j,:,:], (np.array(self.data_4D[i,j,:,:].shape)/2-com).astype(int))
                

    
    def estimate_aperture_size(self):
        Ronchi_mean=np.mean(self.data_4D,(0,1))
        Ronchi_norm = (Ronchi_mean - np.amin(Ronchi_mean)) / np.ptp(Ronchi_mean)
        self.BFdisk = np.ones(Ronchi_norm.shape) * (Ronchi_norm > self.threshold)
        self.edge = (np.sum(np.abs(np.gradient(self.BFdisk)), axis=0)) > self.threshold 
        xx,yy = np.meshgrid(np.arange(0,Ronchi_mean.shape[1]),np.arange(0,Ronchi_mean.shape[0]))
        self.center_x = np.sum(self.BFdisk*xx/np.sum(self.BFdisk))
        self.center_y = np.sum(self.BFdisk*yy/np.sum(self.BFdisk))
        self.aperture_radius = np.average(np.sqrt((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2)[self.edge])
        self.calibration = self.aperturesize/self.aperture_radius
        
        if (np.count_nonzero(self.BFdisk)==0):
            print('Warning, no BF disk detected, you might decrease threshold')
            return 1
    

    ## private    
    def _init_figure(self, rows, cols,figsize,num=None):
        #get_ipython().run_line_magic('matplotlib', 'inline') 
        fig,ax =plt.subplots(rows,cols,figsize=figsize,num=num)
        # grid=ImageGrid(fig, int(str(cols)+str(rows)+str(cols)), nrows_ncols=(rows,cols),
        # axes_pad=0.5,
        # share_all=False,
        # cbar_location="right",
        # cbar_mode="each",
        
        # cbar_size="5%",
        # cbar_pad="2%"
        # )
        return fig, ax
        
        


    def plot_aperture(self):
        aperture_round  = self.circle(self.center_x, self.center_y,self.aperture_radius)
        Ronchi_mean=np.mean(self.data_4D,(0,1))
        
        fig, ax = self._init_figure(1,3, (17,4),num = 'Aperture')
        im=ax[0].imshow(Ronchi_mean,cmap=cm.nipy_spectral)
        fig.colorbar(im, ax=ax[0])        
        ax[0].set_title('Averaged ronchigram')
        im2=ax[1].imshow(self.BFdisk,cmap=cm.nipy_spectral)
        ax[1].set_title('Bright field disk')
        im3=ax[2].imshow(self.edge)
        ax[2].plot(aperture_round[0],aperture_round[1], linewidth=10) 
        ax[2].set_title('Aperture edge')
        
        plt.show()
        #im=grid[3].imshow(self.edge) 
        #grid[3].plot(aperture_round[0],aperture_round[1], linewidth=10) 
        #grid[3].set_title('Estimated aperture size') 


# now we truncate and shift the ronchigram in order to use it in the Trotter generation.
# The center of the new ronchigram should be close to the center pixel

    def truncate_ronchigram(self, expansion_ratio = None):
        if  expansion_ratio == None:
            self.x_hwindow_size  = int(self.data_4D.shape[3]/2)
            self.y_hwindow_size  = int(self.data_4D.shape[2]/2)
            window_start_x = 0 
            window_start_y = 0 
            self.data_4D_trunc=self.data_4D

            Ronchi_angle_step=self.aperturesize/self.aperture_radius
            self.Ronchi_angles_x=(np.arange(self.data_4D.shape[3])-self.center_x)*Ronchi_angle_step
            self.Ronchi_angles_y=(np.arange(self.data_4D.shape[2])-self.center_y)*Ronchi_angle_step  
            #self.calibration = Ronchi_angle_step = self.aperturesize/self.data_4D_trunc.shape[3]*2


        else:           
            self.x_hwindow_size = int(np.fix(self.aperture_radius*expansion_ratio))
            self.y_hwindow_size = int(np.fix(self.aperture_radius*expansion_ratio))       
            window_start_x = int(np.fix(self.center_x)) - self.x_hwindow_size
            window_start_y = int(np.fix(self.center_y)) - self.y_hwindow_size
            new_center_x = self.center_x - window_start_x
            new_center_y = self.center_y - window_start_y
            self.center_x = new_center_x
            self.center_y = new_center_y

            Ronchi_angle_step=self.aperturesize/self.aperture_radius
            self.Ronchi_angles_x=(np.arange(self.x_hwindow_size*2+1)-new_center_x)*Ronchi_angle_step
            self.Ronchi_angles_y=(np.arange(self.y_hwindow_size*2+1)-new_center_y)*Ronchi_angle_step
        if  expansion_ratio != None:
            self.data_4D_trunc=np.zeros((self.scan_row, self.scan_col,self.y_hwindow_size*2+1, self.x_hwindow_size*2+1), dtype = self.data_4D.dtype)
            for i in range(self.data_4D.shape[0]):
                for j in range (self.data_4D.shape[1]):
                    ronchi = self.data_4D[i,j,:,:]
                    ronchi[self.BFdisk==0] =0
                    self.data_4D_trunc[i,j,:,:]=ronchi[window_start_y:window_start_y+ self.y_hwindow_size*2+1,
                                                         window_start_x:window_start_x+ self.x_hwindow_size*2+1]
        self.center_x = int(self.data_4D_trunc.shape[3]/2)
        self.center_y = int(self.data_4D_trunc.shape[2]/2)

    # Now we start to Fourier transform the Ronchigram along the probe position and show the trotters.  
    def apply_FT(self):
        self.data_4D_Reciprocal=np.zeros(self.data_4D_trunc.shape,dtype='complex64') 
        for i in range (self.data_4D_trunc.shape[2]): 
            for j in range (self.data_4D_trunc.shape[3]): 
                self.data_4D_Reciprocal[:,:,i,j]=self.FFT_2D(self.data_4D_trunc[:,:,i,j]) 
 
        self.power_spectra =np.zeros((self.data_4D_trunc.shape[0],self.data_4D_trunc.shape[1])) 
        
        for i in range (self.data_4D_trunc.shape[0]): 
            for j in range (self.data_4D_trunc.shape[1]): 
                g=self.data_4D_Reciprocal[i,j,:,:] 
                self.power_spectra[i,j]=np.sum(g*np.conjugate(g)).real 
                
    def plot_4D_reciprocal(self,signal = 'amplitude',log=True):
        
        if ps == None:
            print('pixstem package not loaded...')
            print('4D plot not possible')
            return
        #get_ipython().run_line_magic('matplotlib', 'auto') 
        #dict0 = { 'name':'x', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict1 = { 'name':'y', 'units':'A', 'scale':self.step_size, 'offset':0}
        #dict2 = { 'name':'angle x', 'units':'rad', 'scale':self.scan_angle_step_x, 'offset':self.scan_angles_x[0]}
        #dict3 = { 'name':'angle <', 'units':'rad', 'scale':self.scan_angle_step_y, 'offset':self.scan_angles_y[0]}
        if signal == 'amplitude':
            if log:
                s = ps.PixelatedSTEM(np.log(np.abs(self.data_4D_Reciprocal)))
            else:
                s = ps.PixelatedSTEM((np.abs(self.data_4D_Reciprocal))) 
        elif signal == 'phase':
            s = ps.PixelatedSTEM(np.angle(self.data_4D_Reciprocal))
        else:
            print('signal keyword not understood')
            return
        s.axes_manager[0].name = 'frequency x'
        s.axes_manager[0].units = '1/A'
        s.axes_manager[0].scale = 1/self.step_size
        s.axes_manager[1].name = 'frequency y'
        s.axes_manager[1].units = '1/A'
        s.axes_manager[1].scale = 1/self.step_size
        s.axes_manager[2].name = 'angle x'
        s.axes_manager[2].units = 'mrad'
        s.axes_manager[2].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[2].offset = self.scan_angles_x[0]
        s.axes_manager[3].name = 'angle y'
        s.axes_manager[3].units = 'mrad'
        s.axes_manager[3].scale = self.aperturesize/self.aperture_radius*1000
        s.axes_manager[3].offset = self.scan_angles_y[0]
        s.metadata.General.title = 'FT of 4D data'
        s.plot(cmap='viridis')
        #circle1 = plt.Circle((int(self.data_4D_trunc.shape[3]/2), int(self.data_4D_trunc.shape[2]/2)), 
         #                    self.aperture_radius, color='r',fill=None)
        #plt.gca().add_patch(circle1)
        return s
        
        
        ## will be an own function
    def plot_FT(self):
        fig, ax = self._init_figure(1,2,(12,4),num='Fourier Transform')
        im=ax[0].imshow(self.power_spectra) 
        fig.colorbar(im, ax=ax[0])        
        ax[0].set_title('Power Spectrum') 
        im2=ax[1].imshow(np.log10(1+self.power_spectra)) 
        fig.colorbar(im2, ax=ax[1])        
        ax[1].set_title('Power Spectrum in logrithm') 
        plt.show()

    def plot_trotter(self, frame):
        row,col = frame
        fig, ax = self._init_figure(1,2,figsize=(12,4),num ='Trotters´')
        im=ax[0].imshow(np.abs(self.data_4D_Reciprocal[row,col]))
        fig.colorbar(im, ax=ax[0])        
        im2=ax[1].imshow(np.angle(self.data_4D_Reciprocal[row,col]))
        fig.colorbar(im2, ax=ax[1])        
        ax[0].set_title('amplitude')
        ax[1].set_title('phase')
        plt.show()
##CH: not required                    
#ind=np.unravel_index(np.argsort(power_spectra, axis=None), power_spectra.shape) # rank from low to high
#reverse_ind=(ind[0][::-1],ind[1][::-1]) # rank from high to low
#
#fig =plt.figure(1, figsize=(80, 80))
#grid=ImageGrid(fig, 111, nrows_ncols=(8,8),
#             axes_pad=0.5,
#             share_all=False,
#             cbar_location="right",
#             cbar_mode="share",
#             cbar_size="5%",
#             cbar_pad="2%")
#for i in range(8):
#    for j in range(8):
#        frame_idx = 8*i+j;
#        frame_y_idx=reverse_ind[0][frame_idx+1]
#        frame_x_idx=reverse_ind[1][frame_idx+1]
#        single_trotter=data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]
#        im=grid[frame_idx].imshow(np.angle(single_trotter))
#        grid.cbar_axes[frame_idx].colorbar(im)
#        grid[frame_idx].set_title('Trotters')
                    
              
    ## CH: To do: example frame as option and not as global variable
    def plot_trotters(self,rotation_angle,selected_frames = None, plot_constrains = True, skip = 0):       
        rotation_angle = -rotation_angle/180*np.pi
        self.rotation_angle = rotation_angle
        ind=np.unravel_index(np.argsort(self.power_spectra, axis=None), self.power_spectra.shape) # rank from low to high 
        
        #example_frames = peaks[:-1]
        #    example_frame = ex
        #reverse_ind=(ind[0][::-1],ind[1][::-1]) # rank from high to low
        fig, ax = self._init_figure(3,3,(10,10),num='Calibrate rotation phase')
        fig2, ax2 = self._init_figure(3,3,(10,10),num='Calibrate rotation amplitude')
        #print('fft peaks at x '+str(ind[1][-2::-10]))
        #print('fft peaks at y '+str(ind[0][-2::-10]))
        self.selected_frames = []
        for i in range(9):
            #take 9 bright spots in fft as a example
            frame_x_idx=ind[1][-i-1-skip]
            frame_y_idx=ind[0][-i-1-skip]
            if selected_frames != None:
                if len(selected_frames)>i:
                    frame_x_idx = selected_frames[i][1]
                    frame_y_idx = selected_frames[i][0]
            #if example_frame is not None and i==0:
            #    frame_x_idx = example_frame[1]
             #   frame_y_idx = example_frame[0]
            self.selected_frames.append([frame_y_idx,frame_x_idx])
            scan_x_angle = self.scan_angles_x[frame_x_idx]*np.cos(rotation_angle) - self.scan_angles_y[frame_y_idx]*np.sin(rotation_angle)
            scan_y_angle = self.scan_angles_x[frame_x_idx]*np.sin(rotation_angle) + self.scan_angles_y[frame_y_idx]*np.cos(rotation_angle)
            #Here we need to consider the coordinate difference in imshow. The scan y angle should be opposite.       
            round1=self.circle(scan_x_angle, -scan_y_angle, self.aperturesize)
            round2=self.circle(-scan_x_angle, scan_y_angle, self.aperturesize)
            round3=self.circle(0,0, self.aperturesize)
            im=ax[int(i/3),i%3].imshow(np.angle(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            
            im2=ax2[int(i/3),i%3].imshow(np.abs(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            fig.colorbar(im, ax=ax[int(i/3),i%3])    
            if plot_constrains:
                ax[int(i/3),i%3].plot(round1[0],round1[1], linewidth=5, color = 'red')
                ax[int(i/3),i%3].plot(round2[0],round2[1], linewidth=5, color = 'blue')
                ax[int(i/3),i%3].plot(round3[0],round3[1], linewidth=5, color = 'green')
                ax[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))
                
            
            fig2.colorbar(im, ax=ax2[int(i/3),i%3]) 
            if plot_constrains:
                ax2[int(i/3),i%3].plot(round1[0],round1[1], linewidth=5, color = 'red')
                ax2[int(i/3),i%3].plot(round2[0],round2[1], linewidth=5, color = 'blue')
                ax2[int(i/3),i%3].plot(round3[0],round3[1], linewidth=5, color = 'green')
                ax2[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax2[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax2[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))
        fig.tight_layout()
        fig2.tight_layout()
        '''   
         
        params = np.array([['file',self.file],
                  ['method',self.method],
                  ['stepsize',str(self.step_size)],
                  ['voltage',str(self.voltage)],
                  ['rotation',str(self.rotation_angle*180/np.pi)],
                  ['threshold',str(self.threshold)]])
        with open(self.path+'parameters.txt','w') as f:
            f.write('file\t'+self.file)
            f.write('\nmethod\t'+self.method)
            f.write('\naperture\t'+str(self.aperturesize))
            f.write('\nstepsize\t'+str(self.step_size))
            f.write('\nvoltage\t'+str(self.voltage))
            f.write('\nrotation\t'+str(self.rotation_angle*180/np.pi))
            f.write('\nthreshold\t'+str(self.threshold))
            f.write('\nworkers\t'+str(self.workers))
         '''  
        
    '''    
    def open_widget(self):
        import tkinter as tk
        
        master = tk.Tk()
        tk.Label(master, 
                 text="Rotation angle").grid(row=0)
        tk.Label(master, 
                 text="Step size").grid(row=1)

        e1 = tk.Entry(master)
        e2 = tk.Entry(master)
        e1.insert(self.rotation_angle)
        e2.insert(self.step_size)
        e1.grid(row=0, column=1)
        e2.grid(row=1, column=1)

        tk.Button(master, text='Update', command=self.update(e1.get(),e2.get()).grid(row=3, column=0, sticky=tk.W,  pady=4))
        tk.Button(master, text='Continue')

        tk.mainloop()
        
    def update(self,rotation_angle,step_size):
        self.fig_trotters_phase.close()  
        self.fig_trotters_amp.close() 
        self.step_size = step_size
        self.setup_scanning_parameters()    
        self.plot_higher_order_trotters(rotation_angle)
           
    '''
    def plot_higher_order_trotters(self,rotation_angle,selected_frames = None, order = 1,log = True, plot_constrains = True, skip = 0):       
        rotation_angle = -rotation_angle/180*np.pi
        self.rotation_angle = rotation_angle
        ind=np.unravel_index(np.argsort(self.power_spectra, axis=None), self.power_spectra.shape) # rank from low to high 
        
        #example_frames = peaks[:-1]
        #    example_frame = ex
        #reverse_ind=(ind[0][::-1],ind[1][::-1]) # rank from high to low
        self.fig_trotters_phase, ax = self._init_figure(3,3,(10,10),num='Calibrate rotation phase')
        self.fig_trotters_amp, ax2 = self._init_figure(3,3,(10,10),num='Calibrate rotation amplitude')
        #print('fft peaks at x '+str(ind[1][-2::-10]))
        #print('fft peaks at y '+str(ind[0][-2::-10]))
        self.selected_frames = []
        for i in range(9):
            #take 9 bright spots in fft as a example
            frame_x_idx=ind[1][-i-1-skip]
            frame_y_idx=ind[0][-i-1-skip]
            if selected_frames != None:
                if len(selected_frames)>i:
                    frame_x_idx = selected_frames[i][1]
                    frame_y_idx = selected_frames[i][0]
            #if example_frame is not None and i==0:
            #    frame_x_idx = example_frame[1]
             #   frame_y_idx = example_frame[0]
            self.selected_frames.append([frame_y_idx,frame_x_idx])
            scan_x_angle = self.scan_angles_x[frame_x_idx]*np.cos(rotation_angle) - self.scan_angles_y[frame_y_idx]*np.sin(rotation_angle)
            scan_y_angle = self.scan_angles_x[frame_x_idx]*np.sin(rotation_angle) + self.scan_angles_y[frame_y_idx]*np.cos(rotation_angle)
            #Here we need to consider the coordinate difference in imshow. The scan y angle should be opposite.       
            round1=self.circle(scan_x_angle, -scan_y_angle, self.aperturesize)
            round2=self.circle(-scan_x_angle, scan_y_angle, self.aperturesize)
            round3=self.circle(0,0, self.aperturesize)
            im=ax[int(i/3),i%3].imshow(np.angle(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            if log:
                im2=ax2[int(i/3),i%3].imshow(np.log(np.abs(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:])),
                                  extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])    
            else:
                im2=ax2[int(i/3),i%3].imshow(np.abs(self.data_4D_Reciprocal[frame_y_idx,frame_x_idx,:,:]),
                              extent=[self.Ronchi_angles_x.min(),self.Ronchi_angles_x.max(),
                                      self.Ronchi_angles_y.min(),self.Ronchi_angles_y.max()])
            self.fig_trotters_phase.colorbar(im, ax=ax[int(i/3),i%3])    
            if plot_constrains:
                ax[int(i/3),i%3].plot(round1[0],round1[1], linewidth=2, color = 'red')
                ax[int(i/3),i%3].plot(round2[0],round2[1], linewidth=2, color = 'blue')
                for j in range(1,order):
                    round1b=self.circle(scan_x_angle*(j+1), -scan_y_angle*(j+1), self.aperturesize)
                    round2b=self.circle(-scan_x_angle*(j+1), scan_y_angle*(j+1), self.aperturesize)
                    ax[int(i/3),i%3].plot(round1b[0],round1b[1], linewidth=2, color = 'red')
                    ax[int(i/3),i%3].plot(round2b[0],round2b[1], linewidth=2, color = 'blue')
                ax[int(i/3),i%3].plot(round3[0],round3[1], linewidth=2, color = 'green')
                ax[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))
                
            
            self.fig_trotters_amp.colorbar(im, ax=ax2[int(i/3),i%3]) 
            if plot_constrains:
                ax2[int(i/3),i%3].plot(round1[0],round1[1], linewidth=2, color = 'red')
                ax2[int(i/3),i%3].plot(round2[0],round2[1], linewidth=2, color = 'blue')
                ax2[int(i/3),i%3].plot(round3[0],round3[1], linewidth=2, color = 'green')
                for j in range(1,order):
                    round1b=self.circle(scan_x_angle*(j+1), -scan_y_angle*(j+1), self.aperturesize)
                    round2b=self.circle(-scan_x_angle*(j+1), scan_y_angle*(j+1), self.aperturesize)
                    ax2[int(i/3),i%3].plot(round1b[0],round1b[1], linewidth=2, color = 'red')
                    ax2[int(i/3),i%3].plot(round2b[0],round2b[1], linewidth=2, color = 'blue')
                ax2[int(i/3),i%3].plot(scan_x_angle,-scan_y_angle,'r*')
                ax2[int(i/3),i%3].plot(-scan_x_angle, scan_y_angle,'b*')
            ax2[int(i/3),i%3].set_title('index '+str(frame_y_idx)+' '+str( frame_x_idx))
        #self.open_widget()

        '''   
        params = np.array([['file',self.file],
                  ['method',self.method],
                  ['stepsize',str(self.step_size)],
                  ['voltage',str(self.voltage)],
                  ['rotation',str(self.rotation_angle*180/np.pi)],
                  ['threshold',str(self.threshold)]])
        with open(self.path+'parameters.txt','w') as f:
            f.write('file\t'+self.file)
            f.write('\nmethod\t'+self.method)
            f.write('\naperture\t'+str(self.aperturesize))
            f.write('\nstepsize\t'+str(self.step_size))
            f.write('\nvoltage\t'+str(self.voltage))
            f.write('\nrotation\t'+str(self.rotation_angle*180/np.pi))
            f.write('\nthreshold\t'+str(self.threshold))
            f.write('\nworkers\t'+str(self.workers))
        '''
        
    def circle(self, x0,y0,ridius):
        phi=np.pi*(np.linspace(-1,1,200))
        x=x0+ridius*np.cos(phi)
        y=y0+ridius*np.sin(phi)
        return x,y
    def FFT_2D (self, array):
        result=np.fft.fft2(array)
        result=np.fft.fftshift(result)
        return result
    def IFFT_2D (self, array):
        result=np.fft.ifftshift(array)
        result=np.fft.ifft2(result)
        return result
    
    

class SVD_AC():
    def __init__(self, data4D, trotters_nb=8,aberr_order=3):
        self.data4D = data4D
        self.trotters_nb = trotters_nb
        self.aberr_order = aberr_order
        self.coeff_len = [3,7,12] # number of aberration coefficients 
        #self.Ronchi_xx,self.Ronchi_yy=np.meshgrid(self.data4D.Ronchi_angles_x,self.data4D.Ronchi_angles_y)
        self.theta_x=(np.arange(self.data4D.data_4D_trunc.shape[3])-self.data4D.center_x)*self.data4D.calibration
        self.theta_y=(np.arange(self.data4D.data_4D_trunc.shape[2])-self.data4D.center_y)*self.data4D.calibration
        self.theta_xx,self.theta_yy=np.meshgrid(self.theta_x,self.theta_y)
        
    def OmnilinePrep(self,kx,ky,qx,qy,nth):

    # AX = b;
    # A is the OmniMatrix, being prepared line by line using this function.
    # X = [C1, C12a, C12b, C23a, C23b, C21a, C21b, C3, C34a, C34b, C32a, C32b]';
    # b = phase{G(Kf,Qp)};

    # G(Kf,Qp) = |A(Kf)|^2.*delta(Qp) + A(Kf-Qp)A.*(Kf)Psi_s(Qp) +  A(Kf)A.*(Kf+Qp)Psi_s.*(-Qp);

    # the phase of A(Kf)A.*(Kf+Qp)Psi_s.*(-Qp)equals to:
    # chi(Kf) - chi(Kf+Qp) - phase(Psi_s(-Qp))
        kx2=kx*kx
        ky2=ky*ky
        kx3=kx2*kx
        ky3=ky2*ky
        kx4=kx3*kx
        ky4=ky3*ky

        sx=kx+qx
        sy=ky+qy
        sx2=sx*sx
        sy2=sy*sy
        sx3=sx2*sx
        sy3=sy2*sy
        sx4=sx3*sx
        sy4=sy3*sy

        if nth==1:
            omniline=1/2*((sx2+sy2)-(kx2+ky2)),1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky

        elif nth==2:
            omniline=1/2*((sx2+sy2)-(kx2+ky2)),1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky,\
                       1/3*((sx3-3*sx*sy2)-(kx3-3*kx*ky2)),1/3*((-sy3+3*sx2*sy)-(-ky3+3*kx2*ky)),\
                       1/3*((sx3+sx*sy2)-(kx3+kx*ky2)),1/3*((sy3+sx2*sy)-(ky3+kx2*ky))
        elif nth==3:
            omniline=1/2*((sx2+sy2)-(kx2+ky2)),1/2*((sx2-sy2)-(kx2-ky2)),sx*sy-kx*ky,\
                       1/3*((sx3-3*sx*sy2)-(kx3-3*kx*ky2)),1/3*((-sy3+3*sx2*sy)-(-ky3+3*kx2*ky)),\
                       1/3*((sx3+sx*sy2)-(kx3+kx*ky2)),1/3*((sy3+sx2*sy)-(ky3+kx2*ky)),\
                       1/4*((sx4+sy4+2*sx2*sy2)-(kx4+ky4+2*kx2*ky2)),\
                       1/4*((sx4-6*sx2*sy2+sy4)-(kx4-6*kx2*ky2+ky4)),\
                       1/4*((-4*sx*sy3+4*sx3*sy)-(-4*kx*ky3+4*kx3*ky)),\
                       1/4*((sx4-sy4)-(kx4-ky4)),1/4*((2*sx3*sy+2*sx*sy3)-(2*kx3*ky+2*kx*ky3))
        elif nth==4:
            print('4th order and above is under development')

        return -np.array(omniline)
    
    def build_omnimatrix(self):
        OmniMatrix_total=np.zeros(self.coeff_len[self.aberr_order-1]+self.trotters_nb,dtype=float)
        self.b=np.zeros(0,dtype=float)
        self.ind=np.unravel_index(np.argsort(self.data4D.power_spectra, axis=None), self.data4D.power_spectra.shape) # rank from low to high 
        
        for i in range(self.trotters_nb):
            yy=self.data4D.selected_frames[i][0]
            xx=self.data4D.selected_frames[i][1]
            
            print('selecting index '+str(xx),' '+str(yy))
            #yy=reverse_ind[0][i+1] # To obtain the corrdinates of each chosen trotters,the reason i+1 is because
            #xx=reverse_ind[1][i+1] # the first one is 0-frequency position. we should count from the second one.

            g=self.data4D.data_4D_Reciprocal[yy,xx,:,:].copy()

            scan_x_angle = self.data4D.scan_angles_x[xx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[yy]*np.sin(self.data4D.rotation_angle)
            scan_y_angle = self.data4D.scan_angles_x[xx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[yy]*np.cos(self.data4D.rotation_angle)
            (gL,LTrotter_amp, LTrotter_phase,gR,RTrotter_amp,
                     RTrotter_phase,Trotter_mask,g_amp, g_phase) = mask_trotter(g,self.theta_x,self.theta_y, scan_x_angle,scan_y_angle,self.data4D.aperturesize)

            ST_phase=np.angle(g)
            ST_phase[Trotter_mask==0]=0
            ST_phase_unwrap=unwrap_phase(ST_phase)
            ST_phase_unwrap[Trotter_mask==0]=0
            ST_phase_unwrap=unwrap_phase(ST_phase_unwrap)
            ST_phase_unwrap[Trotter_mask==0]=0

            constvecR=np.zeros(self.trotters_nb)
            constvecR[i]=1
    


            for iy in range (self.data4D.data_4D_Reciprocal.shape[2]):
                for ix in range (self.data4D.data_4D_Reciprocal.shape[3]):
                    if gL[iy,ix]>0:
                        OmniMatrix_aberration = self.OmnilinePrep(self.theta_x[ix],self.theta_y[iy],
                                                              scan_x_angle,scan_y_angle,self.aberr_order)
                        OmniMatrix=np.hstack((OmniMatrix_aberration,constvecR))
                        self.b=np.hstack((self.b,ST_phase_unwrap[iy,ix]))
                        OmniMatrix_total=np.vstack((OmniMatrix_total,OmniMatrix))
        self.OmniMatrix_total=np.delete(OmniMatrix_total,(0),axis=0)

    def SVD(self,A,b):    
        u,s,vh=np.linalg.svd(A,full_matrices=True)
        u=np.round(u,4)
        s=np.round(s,4)
        vh=np.round(vh,4)
        d=np.round(inv(u).dot(b),4)
        i_nonzero=np.nonzero(s)[0].shape[0]
        new=np.zeros(min(A.shape[0],A.shape[1]))
        for i in range (i_nonzero):
            new[i]=d[i]/s[i]
        aberrations=inv(vh).dot(new)
        return aberrations 

    def run_SVD(self):
        aberrations=self.SVD(self.OmniMatrix_total,self.b)
        self.svdcoeff=aberrations*self.data4D.wave_len*1e-10/(2*np.pi)
        self.aberration_coeffs=np.zeros(12)
        self.aberration_coeffs[:self.coeff_len[self.aberr_order-1]]=self.svdcoeff[:self.coeff_len[self.aberr_order-1]]
        
    def calc_aberrationfunction(self):
        func_aberr   = Aberr_func(self.theta_xx,self.theta_yy,self.aberration_coeffs)
        self.func_transfer= np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr)
        
    def calc_aperturefunction(self):
        theta=np.sqrt(self.theta_xx**2+self.theta_yy**2)
        func_objApt=np.ones(theta.shape)
        func_objApt[theta>self.data4D.aperturesize]=0
        dose=np.sum(self.data4D.data_4D)/(self.data4D.data_4D.shape[0]*self.data4D.data_4D.shape[1])
        scaling=np.sqrt(dose/np.sum(func_objApt))
        self.func_objApt=scaling*func_objApt

    def calc_probefunction(self):        
        A=self.func_objApt*self.func_transfer
        self.probe=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))
        
    def fit_aberrations(self):
        aberr_order_seq = np.repeat([1,2,3],5)  
        aberr_order_seq=np.ones(20).astype(int)# this sequence is to specify which oder of aberations to correct, in this case
        aberr_order_seq[:50]=3 # 3rd order aberration.
        Nitt=np.size(aberr_order_seq)
        aa,bb=np.meshgrid(self.data4D.scan_angles_x,self.data4D.scan_angles_y)
        Q=np.sqrt(aa**2+bb**2)
        #del self.OmniMatrix_total 
        #del self.b
        for itt in range(Nitt):
            aberr_order=aberr_order_seq[itt]
            #count=1
            
            for i in range (self.trotters_nb):
                # M: the numbers of trotters we choose
                yy=self.data4D.selected_frames[i][0]
                xx=self.data4D.selected_frames[i][1]
        #        if theta[yy,xx]>0.1*Aperture:
                    
                g=self.data4D.data_4D_Reciprocal[yy,xx,:,:].copy()

                scan_x_angle = self.data4D.scan_angles_x[xx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[yy]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[xx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[yy]*np.cos(self.data4D.rotation_angle)
                (gL,LTrotter_amp, LTrotter_phase,gR,RTrotter_amp,
                     RTrotter_phase,Trotter_mask,g_amp, g_phase) = mask_trotter(g,self.theta_x,self.theta_y, scan_x_angle,scan_y_angle,self.data4D.aperturesize)
                
                kx_plusQ=self.theta_xx+scan_x_angle
                ky_plusQ=self.theta_yy+scan_y_angle
                k_plusQ=np.sqrt(kx_plusQ**2+ky_plusQ**2)
                    
                kx_minusQ=self.theta_xx-scan_x_angle
                ky_minusQ=self.theta_yy-scan_x_angle
                k_minusQ=np.sqrt(kx_minusQ**2+ky_minusQ**2)
                    
                func_aber        = Aberr_func(self.theta_xx,self.theta_yy,self.aberration_coeffs)
                func_aber_plusQ  = Aberr_func(kx_plusQ,ky_plusQ,self.aberration_coeffs)
                func_aber_minusQ = Aberr_func(kx_minusQ,ky_minusQ,self.aberration_coeffs)
                     
                func_transfer       = np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aber)
                func_transfer_plusQ = np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aber_plusQ)
                func_transfer_minusQ= np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aber_minusQ)
                            
                theta=np.sqrt(self.theta_xx**2+self.theta_yy**2)
    
                func_objApt_plusQ =np.ones(g.shape)
                func_objApt_plusQ[k_plusQ>self.data4D.aperturesize] =0
                func_objApt_plusQ[k_minusQ<self.data4D.aperturesize]=0
                func_objApt_plusQ[theta>self.data4D.aperturesize]   =0
                    
                func_objApt_minusQ=np.ones(g.shape)
                func_objApt_minusQ[k_minusQ>self.data4D.aperturesize]=0
                func_objApt_minusQ[k_plusQ<self.data4D.aperturesize] =0
                func_objApt_minusQ[theta>self.data4D.aperturesize]   =0
                
                dose=np.sum(self.data4D.data_4D)/(self.data4D.data_4D.shape[0]*self.data4D.data_4D.shape[1])
                scaling=np.sqrt(dose/np.sum(self.func_objApt))
                
                A      =self.func_objApt*func_transfer
                A_plusQ=func_objApt_plusQ*func_transfer_plusQ*scaling
                A_minusQ=func_objApt_minusQ*func_transfer_minusQ*scaling
        
                AA=A*np.conjugate(A_plusQ)+np.conjugate(A)*A_minusQ
        
                g_phase=np.angle(g)*(func_objApt_plusQ+func_objApt_minusQ)
        
                gc=g*(func_objApt_plusQ+func_objApt_minusQ)*np.exp(1j*np.angle(AA))
        
                g_unwrap=np.angle(gc)
        
                input_file=g_unwrap
                constvec=np.zeros(self.trotters_nb)
                constvec[i]=1
                OmniMatrix_total=np.zeros(self.coeff_len[aberr_order-1]+self.trotters_nb,dtype=float)
                b=np.zeros(0,dtype=float)
        
                for iy in range (g.shape[0]):
                    for ix in range (g.shape[1]):
                        if gL[iy,ix]>0:
                            OmniMatrix_aberration= self.OmnilinePrep(self.theta_x[ix],
                                                                     self.theta_y[iy],
                                                                     scan_x_angle,scan_y_angle,aberr_order)
                            OmniMatrix=np.hstack((OmniMatrix_aberration,constvec))
                            b=np.hstack((b,g_unwrap[iy,ix]))
                            OmniMatrix_total=np.vstack((OmniMatrix_total,OmniMatrix))
                self.OmniMatrix_total=np.delete(OmniMatrix_total,(0),axis=0)
                x=self.SVD(self.OmniMatrix_total,b)
                self.svdcoeff=x*self.data4D.wave_len*1e-10/(2*np.pi)
                aberr_old=self.aberration_coeffs
                aberr_delta=np.zeros(12,dtype=float)
        
                if aberr_order==1:
                    aberr_delta[:3]=self.svdcoeff[:3]
                elif aberr_order==2:
                    aberr_delta[:7]=self.svdcoeff[:7]
                elif aberr_order==3:
                    aberr_delta[:12]=self.svdcoeff[:12]
                cf = 0.5
                self.aberration_coeffs=self.aberration_coeffs+aberr_delta*cf
            
#        else:
#        print('The trotters are not appropriate') 
    def plot_corrected_trotters(self,frames,aberrations):
        self.corrected_trotters = []
        for i in range(len(frames)):
            single_trotter=self.data4D.data_4D_Reciprocal[frames[i][0],frames[i][1]].copy()
            
            scan_x_angle = self.data4D.scan_angles_x[frames[i][1]]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[frames[i][0]]*np.sin(self.data4D.rotation_angle)
            scan_y_angle = self.data4D.scan_angles_x[frames[i][1]]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[frames[i][0]]*np.cos(self.data4D.rotation_angle)
            (RTrotter_mask,RTrotter_phase,RTrotter_amp,LTrotter_mask,LTrotter_phase,
LTrotter_amp,Trotter_mask,Trotter_phase,Trotter_amp) = mask_trotter(single_trotter,self.theta_x,self.theta_y, scan_x_angle,scan_y_angle,self.data4D.aperturesize)
            

            ST_phase=np.angle(single_trotter)
            ST_phase[Trotter_mask==0]=0
            ST_phase_unwrap=unwrap_phase(ST_phase)
            ST_phase_unwrap[Trotter_mask==0]=0
            ST_phase_unwrap=unwrap_phase(ST_phase_unwrap)

            My_trotter=Trotter_amp*np.exp(1j*ST_phase_unwrap)
            My_trotter[Trotter_mask==0]=0

            func_aberr   = Aberr_func(self.theta_xx,self.theta_yy,aberrations,3)
            func_transfer= np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr)

            Ronchi_x_plus,Ronchi_y_plus=self.theta_xx+scan_x_angle,self.theta_yy+scan_y_angle
            func_aberr_plusQ=Aberr_func(Ronchi_x_plus,Ronchi_y_plus,aberrations,3)
            func_transfer_plusQ=np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr_plusQ)

            Ronchi_x_minus,Ronchi_y_minus=self.theta_xx-scan_x_angle,self.theta_yy-scan_y_angle
            func_aberr_minusQ=Aberr_func(Ronchi_x_minus,Ronchi_y_minus,aberrations,3)
            func_transfer_minusQ=np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr_minusQ)

            AA_plus=func_transfer*np.conj(func_transfer_plusQ)
            AA_minus=np.conj(func_transfer)*func_transfer_minusQ
            #AA=func_transfer*np.conj(func_transfer_plusQ)+np.conj(func_transfer)*func_transfer_minusQ
            AA_plus_true_phase=unwrap_phase(np.angle(AA_plus))
            AA_plus_true_phase[LTrotter_mask==0]=0
            AA_plus_true_phase=unwrap_phase(AA_plus_true_phase)
            AA_plus_true_phase[LTrotter_mask==0]=0

            AA_minus_true_phase=unwrap_phase(np.angle(AA_minus))
            AA_minus_true_phase[RTrotter_mask==0]=0
            AA_minus_true_phase=unwrap_phase(AA_minus_true_phase)
            AA_minus_true_phase[RTrotter_mask==0]=0

            AA_true_phase=AA_plus_true_phase+AA_minus_true_phase
            AA_true_phase[Trotter_mask==0]=0
            AA=Trotter_amp*np.exp(1j*AA_true_phase)
            AA[Trotter_mask==0]=0

            Trotter_correction=Trotter_amp*np.exp(1j*Trotter_phase)*np.exp(-1j*(np.angle(AA)))
            
            '''
            func_aberr   = Aberr_func(self.theta_xx,self.theta_yy,aberrations)
            func_transfer= np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr)
            Ronchi_x_plus,Ronchi_y_plus=self.theta_xx+scan_x_angle,self.theta_yy+scan_y_angle
            func_aberr_plusQ=Aberr_func(Ronchi_x_plus,Ronchi_y_plus,aberrations)
            func_transfer_plusQ=np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr_plusQ)

            Ronchi_x_minus,Ronchi_y_minus=self.theta_xx-scan_x_angle,self.theta_yy-scan_y_angle
            func_aberr_minusQ=Aberr_func(Ronchi_x_minus,Ronchi_y_minus,aberrations)
            func_transfer_minusQ=np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr_minusQ)

            AA_plus=func_transfer*np.conj(func_transfer_plusQ)
            AA_minus=np.conj(func_transfer)*func_transfer_minusQ
            #AA=func_transfer*np.conj(func_transfer_plusQ)+np.conj(func_transfer)*func_transfer_minusQ
            AA_plus_true_phase=unwrap_phase(np.angle(AA_plus))
            AA_plus_true_phase[LTrotter_mask==0]=0
            AA_plus_true_phase=unwrap_phase(AA_plus_true_phase)
            AA_plus_true_phase[LTrotter_mask==0]=0

            AA_minus_true_phase=unwrap_phase(np.angle(AA_minus))
            AA_minus_true_phase[RTrotter_mask==0]=0
            AA_minus_true_phase=unwrap_phase(AA_minus_true_phase)
            AA_minus_true_phase[RTrotter_mask==0]=0

            AA_true_phase=AA_plus_true_phase+AA_minus_true_phase
            AA_true_phase[Trotter_mask==0]=0
            AA=Trotter_amp*np.exp(1j*AA_true_phase)
            AA[Trotter_mask==0]=0
            #if scan_x_idx == 68 and scan_y_idx== 69:
            #    self.tmp = AA_plus_true_phase

            single_trotter=Trotter_amp*np.exp(1j*Trotter_phase)*np.exp(-1j*(np.angle(AA)))
            '''
            self.corrected_trotters.append(
                [np.angle(My_trotter),
                 np.angle(AA),
                 np.angle(Trotter_correction)])
########
        
        fig, ax = self.data4D._init_figure(3,len(self.corrected_trotters), (len(self.corrected_trotters)*3,5),
                                    num = 'corrected trotters')
        for i in range(len(self.corrected_trotters)):
            for j in range(3):
                im = ax[j,i].imshow(self.corrected_trotters[i][j])
                fig.colorbar(im,ax=ax[j,i])
                
            ax[0,i].set_title('index '+str(frames[i][0])+' '+str( frames[i][1]))

        ax[0,0].set_ylabel('uncorrected')
        ax[1,0].set_ylabel('calculated')
        ax[2,0].set_ylabel('corrected')


        plt.show()
        
        
    def print_aberration_coefficients(self):
        print('C10  = ',str(round(-self.aberration_coeffs[0]*1e9,3)),' nm')
        print('C12a = ',str(round(-self.aberration_coeffs[1]*1e9,3)),' nm')
        print('C12b = ',str(round(-self.aberration_coeffs[2]*1e9,3)),' nm')
        print('C21a = ',str(round(-self.aberration_coeffs[5]*1e9,3)),' nm')
        print('C21b = ',str(round(-self.aberration_coeffs[6]*1e9,3)),' nm')
        print('C23a = ',str(round(-self.aberration_coeffs[3]*1e9,3)),' nm')
        print('C23b = ',str(round(-self.aberration_coeffs[4]*1e9,3)),' nm')
        print('C30  = ',str(round(-self.aberration_coeffs[7]*1e9,3)),' nm')
        print('C32a = ',str(round(-self.aberration_coeffs[10]*1e9,3)),' nm')
        print('C32b = ',str(round(-self.aberration_coeffs[11]*1e9,3)),' nm')
        print('C34a = ',str(round(-self.aberration_coeffs[8]*1e9,3)),' nm')
        print('C34b = ',str(round(-self.aberration_coeffs[9]*1e9,3)),' nm')
        
        
    
    
    
    
    
    
    

    
    
    
def Aberr_func(x,y,aberrcoeff,order=3):
    
  
    """
    u:kx
    v:ky
    
    output:
    Chi function
    """
    #u,v=np.meshgrid(x,y)
    u,v = x,y
    u2=u*u
    u3=u2*u
    u4=u3*u
    
    v2=v*v
    v3=v2*v
    v4=v3*v
    
    C1   = aberrcoeff[0]
    C12a = aberrcoeff[1]
    C12b = aberrcoeff[2]
    C23a = aberrcoeff[3]
    C23b = aberrcoeff[4]
    C21a = aberrcoeff[5]
    C21b = aberrcoeff[6]
    C3   = aberrcoeff[7]
    C34a = aberrcoeff[8]
    C34b = aberrcoeff[9]
    C32a = aberrcoeff[10]
    C32b = aberrcoeff[11]
    
    if order==1:
        func_aberr=1/2*C1*(u2+v2)+1/2*(C12a*(u2-v2)+2*C12b*u*v)
    if order==2:
        func_aberr=1/2*C1*(u2+v2)+1/2*(C12a*(u2-v2)+2*C12b*u*v)+\
                   1/3*(C23a*(u3-3*u*v2)+C23b*(-v3+3*u2*v))+\
                   1/3*(C21a*(u3+u*v2)+C21b*(v3+u2*v))
    else:
        func_aberr=1/2*C1*(u2+v2)+1/2*(C12a*(u2-v2)+2*C12b*u*v)+\
                   1/3*(C23a*(u3-3*u*v2)+C23b*(-v3+3*u2*v))+\
                   1/3*(C21a*(u3+u*v2)+C21b*(v3+u2*v))+\
                   1/4*(C3*(u4+v4+2*u2*v2))+\
                   1/4*C34a*(u4-6*u2*v2+v4)+\
                   1/4*C34b*(-4*u*v3+4*u3*v)+\
                   1/4*C32a*(u4-v4)+\
                   1/4*C32b*(2*u3*v+2*u*v3)
    
    return func_aberr

def mask_trotter(single_trotter,theta_x,theta_y, scan_x_angle,scan_y_angle,Aperture):
    dx,dy = np.meshgrid(theta_x + scan_x_angle, theta_y + scan_y_angle)
    d1 = np.sqrt(dx*dx+dy*dy)
    dx,dy = np.meshgrid(theta_x - scan_x_angle, theta_y - scan_y_angle)
    d2 = np.sqrt(dx*dx+dy*dy)
    dx,dy = np.meshgrid(theta_x, theta_y)
    d3 = np.sqrt(dx*dx+dy*dy)
    
    LTrotter_phase = np.angle(single_trotter)
    LTrotter_amp   =np.abs(single_trotter)
    LTrotter_mask = np.ones(single_trotter.shape)
    LTrotter_mask[d1>Aperture]=0
    LTrotter_mask[d3>Aperture]=0
    LTrotter_mask[d2<Aperture]=0
    LTrotter_phase[LTrotter_mask==0] =0
    LTrotter_amp[LTrotter_mask==0] =0 

    RTrotter_phase = np.angle(single_trotter)
    RTrotter_amp   =np.abs(single_trotter)
    RTrotter_mask = np.ones(single_trotter.shape)
    RTrotter_mask[d1<Aperture]=0
    RTrotter_mask[d3>Aperture]=0
    RTrotter_mask[d2>Aperture]=0
    RTrotter_phase[RTrotter_mask==0] =0
    RTrotter_amp[RTrotter_mask==0] =0
    
    Trotter_phase = np.angle(single_trotter)
    Trotter_amp = np.abs(single_trotter)
    Trotter_mask=np.logical_or(RTrotter_mask,LTrotter_mask)
    Trotter_phase[Trotter_mask==0] =0
    Trotter_amp[Trotter_mask==0] =0

    return (RTrotter_mask,RTrotter_phase,RTrotter_amp,LTrotter_mask,LTrotter_phase,
            LTrotter_amp,Trotter_mask,Trotter_phase,Trotter_amp)



class SSB():
    
    def __init__(self, data4D):
        self.data4D = data4D

    def integrate_trotter(self,LTrotterSum,scan_x_idxs,scan_y_idxs):
        for scan_x_idx in scan_x_idxs:
            #print('Progress ' +str(scan_y_idx/self.data4D.scan_row*100)+' %\t', end='\r')
            for scan_y_idx in scan_y_idxs:        
                scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                single_trotter=self.data4D.data_4D_Reciprocal[scan_y_idx,scan_x_idx,:,:]
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x + scan_x_angle, self.data4D.Ronchi_angles_y + scan_y_angle)
                d1 = np.sqrt(dx*dx+dy*dy)
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x - scan_x_angle, self.data4D.Ronchi_angles_y - scan_y_angle)
                d2 = np.sqrt(dx*dx+dy*dy) 
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x, self.data4D.Ronchi_angles_y)
                d3 = np.sqrt(dx*dx+dy*dy)

                RTrotter_mask = np.ones(single_trotter.shape)
                LTrotter_mask = np.ones(single_trotter.shape)
                if scan_x_idx != np.fix(self.data4D.scan_col/2) or scan_y_idx != np.fix(self.data4D.scan_row/2):
                    RTrotter_mask[d1>=self.data4D.aperturesize]=0
                    RTrotter_mask[d3>=self.data4D.aperturesize]=0
                    RTrotter_mask[d2<self.data4D.aperturesize]=0
                    LTrotter_mask[d2>=self.data4D.aperturesize]=0
                    LTrotter_mask[d3>=self.data4D.aperturesize]=0
                    LTrotter_mask[d1<self.data4D.aperturesize]=0
                RTrotter_phase = np.angle(single_trotter)
                LTrotter_phase = np.angle(single_trotter)
                RTrotter_amp = np.abs(single_trotter)
                LTrotter_amp = np.abs(single_trotter)
                RTrotter_phase[RTrotter_mask==0] =0
                LTrotter_phase[LTrotter_mask==0] =0
                RTrotter_amp[RTrotter_mask==0] =0
                LTrotter_amp[LTrotter_mask==0] =0
                Lpixel_num =np.sum(LTrotter_mask)
                Rpixel_num =np.sum(RTrotter_mask)
                
                #Trotter_phase = np.angle(single_trotter)
                #Trotter_amp = np.abs(single_trotter)
                #Trotter_mask=np.logical_or(RTrotter_mask,LTrotter_mask)
                #Trotter_phase[Trotter_mask==0] =0
                #Trotter_amp[Trotter_mask==0] =0
                
                #TrotterPixelNum[scan_y_idx,scan_x_idx]=Lpixel_num
                if Lpixel_num ==0:              
                    LTrotterSum[scan_y_idx,scan_x_idx] = 0
                else:
                    LTrotter = LTrotter_amp*np.exp(1j*(LTrotter_phase))
                    LTrotterSum[scan_y_idx,scan_x_idx] = np.sum(LTrotter[:])#/Lpixel_num
                #if Rpixel_num ==0:
                 #   RTrotterSum[scan_y_idx,scan_x_idx] = 0
                #else:
                #    RTrotter = RTrotter_amp*np.exp(1j*(RTrotter_phase))
                #    RTrotterSum[scan_y_idx,scan_x_idx] = np.sum(RTrotter[:])#/Rpixel_num
                    
    def integrate_trotter_higher_order(self,LTrotterSum,scan_x_idxs,scan_y_idxs,order = 2):
        for scan_x_idx in scan_x_idxs:
            #print('Progress ' +str(scan_y_idx/self.data4D.scan_row*100)+' %\t', end='\r')
            for scan_y_idx in scan_y_idxs: 
                #if scan_x_idx!=44 or  scan_y_idx!=32:
                #    continue
                scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                single_trotter=self.data4D.data_4D_Reciprocal[scan_y_idx,scan_x_idx,:,:]
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x + scan_x_angle*order, self.data4D.Ronchi_angles_y + scan_y_angle*order)
                d1 = np.sqrt(dx*dx+dy*dy)
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x + scan_x_angle*(order-1), 
                                    self.data4D.Ronchi_angles_y + scan_y_angle*(order-1))
                d1b = np.sqrt(dx*dx+dy*dy)
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x - scan_x_angle*order, 
                                    self.data4D.Ronchi_angles_y - scan_y_angle*order)
                d2 = np.sqrt(dx*dx+dy*dy) 
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x - scan_x_angle*(order-1), 
                                    self.data4D.Ronchi_angles_y - scan_y_angle*(order-1))
                d2b = np.sqrt(dx*dx+dy*dy) 
                dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x, self.data4D.Ronchi_angles_y)
                d3 = np.sqrt(dx*dx+dy*dy)

                RTrotter_mask = np.ones(single_trotter.shape)
                LTrotter_mask = np.ones(single_trotter.shape)
                if scan_x_idx != np.fix(self.data4D.scan_col/2) or scan_y_idx != np.fix(self.data4D.scan_row/2):
                    #RTrotter_mask[d1<=self.data4D.aperturesize]=0
                    #RTrotter_mask[d1b<=self.data4D.aperturesize]=0
                    #LTrotter_mask[d2b<=self.data4D.aperturesize]=0
                    #LTrotter_mask[d2b<=self.data4D.aperturesize]=0
                    RTrotter_mask = (d1<=self.data4D.aperturesize)*(d1b<=self.data4D.aperturesize)
                    LTrotter_mask = (d2<=self.data4D.aperturesize)*(d2b<=self.data4D.aperturesize)
                
                RTrotter_phase = np.angle(single_trotter)
                LTrotter_phase = np.angle(single_trotter)
                RTrotter_amp = np.abs(single_trotter)
                LTrotter_amp = np.abs(single_trotter)
                RTrotter_phase[RTrotter_mask==0] =0
                LTrotter_phase[LTrotter_mask==0] =0
                RTrotter_amp[RTrotter_mask==0] =0
                LTrotter_amp[LTrotter_mask==0] =0
                Lpixel_num =np.sum(LTrotter_mask)
                Rpixel_num =np.sum(RTrotter_mask)
                
                #Trotter_phase = np.angle(single_trotter)
                #Trotter_amp = np.abs(single_trotter)
                #Trotter_mask=np.logical_or(RTrotter_mask,LTrotter_mask)
                #Trotter_phase[Trotter_mask==0] =0
                #Trotter_amp[Trotter_mask==0] =0
                
                #TrotterPixelNum[scan_y_idx,scan_x_idx]=Lpixel_num
                if scan_y_idx*order>=self.data4D.scan_row or scan_x_idx*order>=self.data4D.scan_col:
                    continue
                    print(scan_y_idx,scan_x_idx)
                if Lpixel_num ==0:              
                    LTrotterSum[scan_y_idx*order,scan_x_idx*order] = 0
                else:
                    LTrotter = LTrotter_amp*np.exp(1j*(LTrotter_phase))
                    LTrotterSum[scan_y_idx*order,scan_x_idx*order] = np.sum(LTrotter[:])#/Lpixel_num
                #if Rpixel_num ==0:
                 #   RTrotterSum[scan_y_idx,scan_x_idx] = 0
                #else:
                #    RTrotter = RTrotter_amp*np.exp(1j*(RTrotter_phase))
                #    RTrotterSum[scan_y_idx,scan_x_idx] = np.sum(RTrotter[:])#/Rpixel_num

    def integrate_trotter_AC(self,LTrotterSum,RTrotterSum,scan_x_idxs,scan_y_idxs, aberrations):
        #+self.Ronchi_xx,self.Ronchi_yy=np.meshgrid(self.data4D.Ronchi_angles_x,self.data4D.Ronchi_angles_y)
        theta_x=(np.arange(self.data4D.data_4D_trunc.shape[3])-self.data4D.center_x)*self.data4D.calibration
        theta_y=(np.arange(self.data4D.data_4D_trunc.shape[2])-self.data4D.center_y)*self.data4D.calibration
        theta_xx,theta_yy=np.meshgrid(theta_x,theta_y)
        theta=np.sqrt(theta_xx**2+theta_yy**2)
        self.corrected_trotters = []
        for scan_x_idx in scan_x_idxs:
            #print('Progress ' +str(scan_y_idx/self.data4D.scan_row*100)+' %\t', end='\r')
            for scan_y_idx in scan_y_idxs:  
                single_trotter=self.data4D.data_4D_Reciprocal[scan_y_idx,scan_x_idx,:,:]

                if scan_x_idx != np.fix(self.data4D.scan_col/2) or scan_y_idx != np.fix(self.data4D.scan_row/2):

                    scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                    scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                    (RTrotter_mask,RTrotter_phase,RTrotter_amp,LTrotter_mask,LTrotter_phase,
LTrotter_amp,Trotter_mask,Trotter_phase,Trotter_amp) = mask_trotter(single_trotter,theta_x,theta_y, scan_x_angle,scan_y_angle,self.data4D.aperturesize)
   
                    func_aberr   = Aberr_func(theta_xx,theta_yy,aberrations)
                    func_transfer= np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr)
                    Ronchi_x_plus,Ronchi_y_plus=theta_xx+scan_x_angle,theta_yy+scan_y_angle
                    func_aberr_plusQ=Aberr_func(Ronchi_x_plus,Ronchi_y_plus,aberrations)
                    func_transfer_plusQ=np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr_plusQ)

                    Ronchi_x_minus,Ronchi_y_minus=theta_xx-scan_x_angle,theta_yy-scan_y_angle
                    func_aberr_minusQ=Aberr_func(Ronchi_x_minus,Ronchi_y_minus,aberrations)
                    func_transfer_minusQ=np.exp(-1j*2*np.pi/(self.data4D.wave_len*1e-10)*func_aberr_minusQ)

                    AA_plus=func_transfer*np.conj(func_transfer_plusQ)
                    AA_minus=np.conj(func_transfer)*func_transfer_minusQ
                    #AA=func_transfer*np.conj(func_transfer_plusQ)+np.conj(func_transfer)*func_transfer_minusQ
                    AA_plus_true_phase=unwrap_phase(np.angle(AA_plus))
                    AA_plus_true_phase[LTrotter_mask==0]=0
                    AA_plus_true_phase=unwrap_phase(AA_plus_true_phase)
                    AA_plus_true_phase[LTrotter_mask==0]=0

                    AA_minus_true_phase=unwrap_phase(np.angle(AA_minus))
                    AA_minus_true_phase[RTrotter_mask==0]=0
                    AA_minus_true_phase=unwrap_phase(AA_minus_true_phase)
                    AA_minus_true_phase[RTrotter_mask==0]=0

                    AA_true_phase=AA_plus_true_phase+AA_minus_true_phase
                    AA_true_phase[Trotter_mask==0]=0
                    AA=Trotter_amp*np.exp(1j*AA_true_phase)
                    AA[Trotter_mask==0]=0

                    single_trotter=Trotter_amp*np.exp(1j*Trotter_phase)*np.exp(-1j*(np.angle(AA)))
        ########
                else:
                    RTrotter_mask = np.ones(single_trotter.shape)
                    LTrotter_mask = np.ones(single_trotter.shape)


                RTrotter_phase = np.angle(single_trotter)
                LTrotter_phase = np.angle(single_trotter)
                RTrotter_amp = np.abs(single_trotter)
                LTrotter_amp = np.abs(single_trotter)
                RTrotter_phase[RTrotter_mask==0] =0
                LTrotter_phase[LTrotter_mask==0] =0
                RTrotter_amp[RTrotter_mask==0] =0
                LTrotter_amp[LTrotter_mask==0] =0
                Lpixel_num =np.sum(LTrotter_mask)
                Rpixel_num =np.sum(RTrotter_mask)
                #TrotterPixelNum[scan_y_idx,scan_x_idx]=Lpixel_num
                if Lpixel_num ==0:
                    LTrotterSum[scan_y_idx,scan_x_idx] = 0
                else:
                    LTrotter = LTrotter_amp*np.exp(1j*(LTrotter_phase))
                    RTrotterSum[scan_y_idx,scan_x_idx] = np.sum(LTrotter[:])#/Lpixel_num
                if Rpixel_num ==0:
                    RTrotterSum[scan_y_idx,scan_x_idx] = 0
                else:
                    RTrotter = RTrotter_amp*np.exp(1j*(RTrotter_phase))
                    LTrotterSum[scan_y_idx,scan_x_idx] = np.sum(RTrotter[:])#/Rpixel_num
                    
    


    def run(self, aberrations = [],order=1):
        # Now we start to make the reconstruction based on the Trotter information
        if self.data4D.workers>1:
            chunknb = self.data4D.workers
            #self.LTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)
            #mp_arr = Array('f', self.data4D.scan_row*self.data4D.scan_col) # shared, can be used from multiple processes
            # then in each new process create a new numpy array using:
            #arr = np.frombuffer(mp_arr.get_obj(),c.c_float) # mp_arr and arr share the same memory
            # make it two-dimensional
            #self.LTrotterSum  = arr.reshape((self.data4D.scan_row,self.data4D.scan_col))#.astype(complex) # b and arr share the same memory
            shared_array_base = multiprocessing.Array(c.c_double, self.data4D.scan_row*self.data4D.scan_col*2)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            self.LTrotterSum = shared_array.view(np.complex128).reshape(self.data4D.scan_row,self.data4D.scan_col)
            if len(aberrations) > 0:
                self.RTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)
            #TrotterPixelNum = np.zeros((self.data4D.scan_row,self.data4D.scan_col))
            scan_y_idxs = np.array_split(range(self.data4D.scan_row), chunknb)  
            scan_x_idxs = range(self.data4D.scan_col) 
            processes = []
            for i in range(chunknb):
                #print(i)
                if len(aberrations) == 0:
                    p = Process(target=self.integrate_trotter, args=(self.LTrotterSum,scan_x_idxs,scan_y_idxs[i]))
                else:
                    p = Process(target=self.integrate_trotter_AC, args=(self.LTrotterSum,self.RTrotterSum,scan_x_idxs,scan_y_idxs[i],aberrations))

                p.daemon = True
                p.start()
                processes.append(p)
            [p.join() for p in processes]
            
        else:
            self.LTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)
            self.RTrotterSum = np.zeros((self.data4D.scan_row,self.data4D.scan_col),dtype=complex)

            scan_y_idxs = range(self.data4D.scan_row) 
            scan_x_idxs = range(self.data4D.scan_col) 
#            self.integrate_trotter_higher_order(self.LTrotterSum,scan_x_idxs,scan_y_idxs,order=order)
            if len(aberrations) == 0:
                self.integrate_trotter(self.LTrotterSum,scan_x_idxs,scan_y_idxs)
            else:
                self.integrate_trotter_AC(self.LTrotterSum,self.RTrotterSum,scan_x_idxs,scan_y_idxs,aberrations)

        #dx,dy = np.meshgrid(self.data4D.Ronchi_angles_x, self.data4D.Ronchi_angles_y)
        #d3 = np.sqrt(dx*dx+dy*dy)
        
                
                
                
        # fig =plt.figure(1, figsize=(60, 20))
        # grid=ImageGrid(fig, 236, nrows_ncols=(1,3),
        #              axes_pad=0.5,
        #              share_all=False,
        #              cbar_location="right",
        #              cbar_mode="each",
        #              cbar_size="5%",
        #              cbar_pad="2%")
        # im=grid[0].imshow(np.square(np.abs(LTrotterSum)))
        # grid.cbar_axes[0].colorbar(im)
        # grid[0].set_title('Power Spectrum')
        # im=grid[1].imshow(np.log10(1+np.square(np.abs(LTrotterSum))))
        # grid.cbar_axes[1].colorbar(im)
        # grid[1].set_title('Power Spectrum in logrithm')
        # im=grid[2].imshow(np.square(TrotterPixelNum))
        # grid.cbar_axes[2].colorbar(im)
        # grid[2].set_title('Pixel Number')
        
        objectL =self.data4D.IFFT_2D(self.LTrotterSum)
        #objectR =self.data4D.IFFT_2D(RTrotterSum)
        
        self.complex = objectL
        self.phase = np.angle(objectL)
        self.amplitude = np.abs(objectL)
        
     
        
    def plot_result(self,sample=1):
        self.fig,ax = self.data4D._init_figure(1,2,(12,4),num = 'Result')
        if sample >1:
            phase = np.array(Image.fromarray(self.phase).resize(np.array(self.phase.shape)*sample,resample=PIL.Image.BICUBIC))
            amplitude = np.array(Image.fromarray(self.amplitude).resize(np.array(self.amplitude.shape)*sample,resample=PIL.Image.BICUBIC))
        else:
            phase = self.phase
            amplitude = self.amplitude

        im0 = ax[0].imshow(phase, extent= [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im0,ax=ax[0])        
        ax[0].set_title('phase')
        im1 = ax[1].imshow(amplitude, extent = [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im1, ax=ax[1])        
        ax[1].set_title("amplitude")
        self.fig.savefig(self.data4D.path+'Result.pdf')
        plt.show()
        
    def save(self):
        try:
            from tifffile import tifffile
            save_tif = True
        except:
            print('Warning: cannot save as .tif (tifffile package required); saving as .txt instead')
            save_tif = False
        
        if save_tif:
            tifffile.imsave(self.data4D.path+'phase_ssb_'+self.data4D.file[:-4]+'.tif', self.phase.astype('float32'), imagej=True)
            tifffile.imsave(self.data4D.path+'amplitude_ssb'+self.data4D.file[:-4]+'.tif', self.amplitude.astype('float32'), imagej=True)

        else:
            np.savetxt(self.data4D.path+'phase_ssb'+self.data4D.file[:-4]+'.txt',self.phase)
            np.savetxt(self.data4D.path+'amplitude_ssb'+self.data4D.file[:-4]+'.txt',self.amplitude)



class WDD():
    def __init__(self, data4D):
        self.data4D = data4D
        self.theta_x=(np.arange(self.data4D.data_4D_trunc.shape[3])-self.data4D.center_x)*self.data4D.calibration
        self.theta_y=(np.arange(self.data4D.data_4D_trunc.shape[2])-self.data4D.center_y)*self.data4D.calibration
        self.theta_xx,self.theta_yy=np.meshgrid(self.theta_x,self.theta_y)
        
    def ProbeFunction(self,Ronchi_angles_x,Ronchi_angles_y,coefficients,Aperture,wave_len):
        angle_xx, angle_yy=np.meshgrid(Ronchi_angles_x,Ronchi_angles_y)
        theta=np.sqrt(angle_xx**2+angle_yy**2)
        func_aberr   = Aberr_func(angle_xx,angle_yy,coefficients,3)
        func_transfer= np.exp(-1j*2*np.pi/(wave_len*1e-10)*func_aberr)
        func_transfer[theta>Aperture]=0
        return func_transfer    
   
    
    
    def run(self,aberrations=np.zeros(12), epsilon_=0.01):
        #generate the Wigner distribution deconvolution of the probe function. W(r, Q)
        #W(r, Q)=FT(P(k)P*(k+Q))
        probe_function_c = self.ProbeFunction(self.theta_x, self.theta_y, aberrations, self.data4D.aperturesize,self.data4D.wave_len)
        WDD_Probe_Zero = self.data4D.IFFT_2D(probe_function_c*np.conjugate(probe_function_c))
        epsilon=(np.max(np.abs(WDD_Probe_Zero)))**2*epsilon_
        WDD_Probe= np.zeros(self.data4D.data_4D_Reciprocal.shape,dtype=self.data4D.data_4D_Reciprocal.dtype)
        for scan_y_idx in range (self.data4D.scan_row):
            print('WDD probe generation progress',(scan_y_idx+1)/self.data4D.scan_row*100,'%',end='\r')
            for scan_x_idx in range (self.data4D.scan_col):
                scan_x_angle = self.data4D.scan_angles_x[scan_x_idx]*np.cos(self.data4D.rotation_angle) - self.data4D.scan_angles_y[scan_y_idx]*np.sin(self.data4D.rotation_angle)
                scan_y_angle = self.data4D.scan_angles_x[scan_x_idx]*np.sin(self.data4D.rotation_angle) + self.data4D.scan_angles_y[scan_y_idx]*np.cos(self.data4D.rotation_angle)
                probe_function_n = self.ProbeFunction(self.theta_x + scan_x_angle,self.theta_y + scan_y_angle,aberrations, self.data4D.aperturesize,self.data4D.wave_len)
                WDD_Probe[scan_y_idx,scan_x_idx]= self.data4D.IFFT_2D(probe_function_c* np.conjugate(probe_function_n))
        data_4D_H=np.zeros(self.data4D.data_4D_Reciprocal.shape,dtype=complex)
        for i in range (self.data4D.scan_row):
            for j in range (self.data4D.scan_col):
                data_4D_H[i,j,:,:]=self.data4D.IFFT_2D(self.data4D.data_4D_Reciprocal[i,j,:,:])

        #determine the object WDD and make the Fourier transfrom
        WDD_probe_conj = np.conjugate(WDD_Probe)
        WDD_Obj =  WDD_probe_conj* data_4D_H / (WDD_Probe*WDD_probe_conj + epsilon)
        data_4D_D=np.zeros(WDD_Obj.shape,dtype=complex)
        for i in range (self.data4D.scan_row):
            for j in range (self.data4D.scan_col):
                data_4D_D[i,j,:,:]=self.data4D.FFT_2D(WDD_Obj[i,j,:,:])
                
        data_4D_D_conj = np.conjugate(data_4D_D)
        D00= data_4D_D[int(np.fix(self.data4D.scan_row/2)), int(np.fix(self.data4D.scan_col/2)), self.data4D.y_hwindow_size, self.data4D.x_hwindow_size]
        D00= np.sqrt(D00)
        #self.Obj_function=data_4D_D_conj[:,:,self.data4D.y_hwindow_size, self.data4D.x_hwindow_size]/D00
        self.Obj_function=data_4D_D_conj[:,:,int(np.fix(data_4D_D.shape[2]/2)),int(np.fix(data_4D_D.shape[3]/2))]/D00
        self.Obj_function = self.data4D.IFFT_2D(self.Obj_function)
        self.phase = np.angle(self.Obj_function)
        self.amplitude = np.abs(self.Obj_function)
    

    def plot_result(self,sample=1):
        self.fig,ax = self.data4D._init_figure(1,2,(12,4),num = 'Result')
        if sample > 1:
            phase = np.array(Image.fromarray(self.phase).resize(np.array(self.phase.shape)*sample,resample=PIL.Image.BICUBIC))
            amplitude = np.array(Image.fromarray(self.amplitude).resize(np.array(self.amplitude.shape)*sample,resample=PIL.Image.BICUBIC))
        else:
            phase = self.phase
            amplitude = self.amplitude

        im0 = ax[0].imshow(phase, extent= [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im0,ax=ax[0])        
        ax[0].set_title('phase')
        im1 = ax[1].imshow(amplitude, extent = [0,self.phase.shape[1]*self.data4D.step_size,0,self.phase.shape[0]*self.data4D.step_size])
        self.fig.colorbar(im1, ax=ax[1])        
        ax[1].set_title("amplitude")
        #self.fig.savefig(self.data4D.path+'Result.pdf')
        plt.show()
        
    def save(self,appendix=''):
        try:
            from tifffile import tifffile
            save_tif = True
        except:
            print('Warning: cannot save as .tif (tifffile package required); saving as .txt instead')
            save_tif = False
        
        if save_tif:
            tifffile.imsave(self.data4D.path+'phase_wdd'+appendix+'.tif', self.phase.astype('float32'), imagej=True)
            tifffile.imsave(self.data4D.path+'amplitude_wdd'+appendix+'.tif', self.amplitude.astype('float32'), imagej=True)

        else:
            np.savetxt(self.data4D.path+'phase_wdd'+appendix+'.txt',self.phase)
            np.savetxt(self.data4D.path+'amplitude_wdd'+appendix+'.txt',self.amplitude)
            
            
            
  
class iCoM:
           
    def __init__(self,data4D):
        self.data4D = data4D
    
    def calc_iCoM(self, dat4d: np.ndarray, Mask: np.ndarray, RCX: float, RCY: float, RCal: float) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get Ronchigram Center of Mass Shifts from 4D Dataset

        :param dat4d: 4D Dataset, 2-spatial, 2-diffraction dimensions
        :param RCX: X Center of the Ronchigram (pixels)
        :param RCY: Y Center of the Ronchigram (pixels)
        :param RCal: Calibration of the Ronchigram (pixels/mrad)
        :param RI: Inner Radius for CoM Measurement (mrad)
        :param RO: Outer Radius for CoM Measurement (mrad)
        :return iCoM as ndarray
        """
        #get the integrated centr of mass image along x and y direction.
        X, Y = np.meshgrid((np.arange(0, dat4d.shape[3]) - RCX)/RCal, (np.arange(0, dat4d.shape[2]) - RCY)/RCal)
        maskeddat4d = dat4d * (Mask > 0)
        return np.average(maskeddat4d * X, axis=(2, 3)), np.average(maskeddat4d * Y, axis=(2, 3))  
    
    def run(self):
        self.icom = self.calc_iCoM(self.data_4D.data_4D, self.data_4D.BFdisk,self.data_4D.center_x ,
                              self.data_4D.center_y, self.data_4D.scan_angle_step_x)
        #ic = self.icom[0]+self.icom[1]
        #f,ax = plt.subplots(1,2)
        #ax[0].imshow(gf(ic,2))
        #ax[1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(ic)))))
        #icom = get_iCOM(ps.PixelatedSTEM(data_4D.data_4D), transpose = True, angle = 180, save = True, path =data_4D.path)
        #icom.plot()
        
    def plot_result(self):
        self.fig,ax = self.data4D._init_figure(1,2,(12,4),num = 'iCOM')
        ic = self.icom[0]+self.icom[1]
        im0 = ax[0].imshow(ic, extent= [0,self.ic.shape[0]*self.data4D.step_size,0,self.phase.shape[1]*self.data4D.step_size])
        self.fig.colorbar(im0,ax=ax[0])        
        ax[0].set_title('iCoM')
        im1 = ax[1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(ic)))), 
                           extent = [0,self.ic.shape[0]*self.data4D.step_size,0,self.phase.shape[1]*self.data4D.step_size])
        self.fig.colorbar(im1, ax=ax[1])        
        ax[1].set_title("FFT")
        #self.fig.savefig(self.data4D.path+'Result_iCoM.pdf')
        plt.show()
        
    def save(self):
        try:
            from tifffile import tifffile
            save_tif = True
        except:
            print('Warning: cannot save as .tif (tifffile package required); saving as .txt instead')
            save_tif = False
        
        if save_tif:
            tifffile.imsave(self.data4D.path+'icom_x.tif', self.icom[0].astype('float32'), imagej=True)
            tifffile.imsave(self.data4D.path+'icom_y.tif', self.icom[1].astype('float32'), imagej=True)

        else:
            np.savetxt(self.data4D.path+'icom_x.txt',self.icom[0])
            np.savetxt(self.data4D.path+'icom_y.txt',self.icom[1])


'''
def get_iCOM(pixstem_obj, transpose = True, angle = 180,save = True, path =''):
    s = pixstem_obj.deepcopy()
    print(s.axes_manager)
    if transpose:
        #s = s.flip_diffraction_y()
        #s = s.rotate_diffraction(90)
        #s = s.flip_diffraction_x(
        for i in range(s.data.shape[0]):
            for j in range(s.data.shape[1]):
                s.data[i,j] = np.transpose(s.data[i,j])
        #s.data= np.transpose(s.data)
        #s.data[0,1] = np.transpose(s.data[0,1])
    
    
    
    scom = s.center_of_mass()
    # comx = scom.data[1]
    # comy = scom.data[0]

    # comx[np.isnan(comx)] = np.nanmean(comx)
    # comy[np.isnan(comy)] = np.nanmean(comy)

    # scom.data[1] = comx
    # scom.data[0] = comy
    icom = scom.get_integrated_signal(sigma=1, angle=angle)
    icom.axes_manager[0].name = pixstem_obj.axes_manager[0].name
    icom.axes_manager[0].units = pixstem_obj.axes_manager[0].units
    icom.axes_manager[0].scale = pixstem_obj.axes_manager[0].scale
    icom.axes_manager[1].name = pixstem_obj.axes_manager[1].name
    icom.axes_manager[1].units = pixstem_obj.axes_manager[1].units
    icom.axes_manager[1].scale = pixstem_obj.axes_manager[1].scale
    icom.metadata.General.title = 'iCOM'
    if save:
        try:
            from tifffile import tifffile
            save_tif = True
        except:
            print('Warning: cannot save as .tif (tifffile package required); saving as .txt instead')
            save_tif = False
        if save_tif:
            tifffile.imsave(path+'iCOM.tif', icom.data.astype('float32'), imagej=True)

        else:
            np.savetxt(path+'iCOM.txt',icom.data)
    return icom 
'''
