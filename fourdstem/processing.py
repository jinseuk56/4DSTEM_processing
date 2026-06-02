import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tifffile
import tkinter.filedialog as tkf

try:
    import cv2
except:
    cv2 = None
    print('The package "OpenCV" is not installed.')
    print('Thus, symmetry STEM imaging cannot be implemented')

from .io import load_binary_4D_stack
from .viewers import fourd_viewer, threed_viewer
from .fem import fourd_radial_transformation

class FourDSTEM_process():
    def __init__(self, file_adr, scan_per_pixel=1, dp_per_pixel=1, scan_unit="nm", k_unit="1/nm", f_shape=None, datatype=np.float32, visual=True):
        self.file_adr = file_adr
        if f_shape != None:
            o_shape = [f_shape[0], f_shape[1], f_shape[2]+2, f_shape[3]]
        
        if file_adr[-3:] == "raw":
            self.f_stack = load_binary_4D_stack(file_adr, datatype, o_shape, f_shape, log_scale=False)
            self.f_stack = np.flip(self.f_stack, axis=2)
            self.f_stack = np.nan_to_num(self.f_stack)
            
        elif file_adr[-3:] == "tif" or file_adr[-4:] == "tiff":
            self.f_stack = tifffile.imread(file_adr)
            self.f_stack  = np.nan_to_num(self.f_stack)

        elif file_adr[-3:] == "dm3" or file_adr[-3:] == "dm4":
            try:
                import hyperspy.api as hs
                self.f_stack = hs.load(file_adr).data
                self.f_stack = fourd_roll_axis(self.f_stack)
                self.f_stack  = np.nan_to_num(self.f_stack)
            except:
                print("HyperSpy must be installed first")
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
            ax[1].imshow(self.charge_density, cmap="RdBu_r", origin="lower")
            ax[1].axis("off")
            ax[2].imshow(self.potential, cmap="inferno", origin="lower")
            ax[2].axis("off")
            fig.tight_layout()

    def symmetry_evaluation(self, angle, also_mirror=False, visual=True):
        """
        Krajnak, M. and J. Etheridge, Proc Natl Acad Sci U S A, 2020. 117(45): p. 27805-27810.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required for symmetry evaluation.")
            
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

def radial_indices(shape, radial_range, scale=1, center=None):
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
    """
    Integrate center of mass (COM) shifts to reconstruct phase/potential.
    Computes Chellappa DPC integration in Fourier space.
    """
    FT_ysh = np.fft.fftshift(np.fft.fft2(ysh))
    FT_xsh = np.fft.fftshift(np.fft.fft2(xsh))
    
    ky = np.fft.fftshift(np.fft.fftfreq(FT_ysh.shape[0])).reshape(-1, 1)
    kx = np.fft.fftshift(np.fft.fftfreq(FT_xsh.shape[1])).reshape(1, -1)

    k2 = ky**2 + kx**2
    zero_ind = np.where(k2 == 0.0)
    k2[zero_ind] = 1.0

    # In frequency space: Potential(k) = ( S_y * ky + S_x * kx ) / ( i * 2 * pi * k^2 )
    FT_phase = (FT_ysh * ky + FT_xsh * kx) / (2 * np.pi * 1j * (hpass + k2 + lpass * k2))
    FT_phase[zero_ind] = 0.0

    Iicom = np.real(np.fft.ifft2(np.fft.ifftshift(FT_phase)))
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

def cepstrum_transformation(img, dCP=False, data_type=np.float32):
    """
    Calculate Cepstral 4D-STEM transformation of a diffraction stack.
    Adds a small epsilon offset to prevent taking the logarithm of zero/negative values.
    """
    img_clipped = np.clip(img.astype(data_type), 1e-9, None)
    mean_dp = np.mean(img_clipped, axis=(0, 1))
    
    fft2_mean = np.fft.fftshift(np.fft.fft2(np.log(mean_dp)))
    fft2_ = np.fft.fftshift(np.fft.fft2(np.log(img_clipped), axes=(2, 3)), axes=(2,3))

    if dCP:
        return np.abs(fft2_).astype(data_type), np.abs(fft2_ - fft2_mean[np.newaxis, np.newaxis, :, :]).astype(data_type)
    else:
        return np.abs(fft2_).astype(data_type), None

def correlation(dat2d, newdata):
    dat1d = dat2d.flatten()
    new1d = newdata.flatten()
    correlation = np.correlate(dat1d, new1d)
    return correlation[0]

def rotation(data, RM):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for image rotation.")
    rotated = cv2.warpAffine(data, RM, data.shape)
    return rotated/np.max(rotated)

def mirror(data, center):
    data1 = data[:, :int(center[1]-1)]
    data2 = np.flip(data, axis=1)[:, :int(center[1]-1)]
    value = correlation(data1, data2)
    return value

def fourd_roll_axis(stack):
    stack = np.rollaxis(np.rollaxis(stack, 2, 0), 3, 1)
    return stack
