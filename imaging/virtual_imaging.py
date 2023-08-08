import numpy as np
try:
    import cv2
except:
    print('The package "OpenCV" is not installed.')
    print('Thus, symmetry STEM imaging cannot be implemented')

class sstem_python():
    
    def __init__(self, f_stack, scan_per_pixel, mrad_per_pixel):
        self.original_stack = f_stack
        self.original_shape = f_stack.shape
        self.original_pacbed = np.mean(self.original_stack, axis=(0, 1))
        self.scan_per_pixel = scan_per_pixel
        self.mrad_per_pixel = mrad_per_pixel
        
        self.intensity_integration_map = np.sum(f_stack, axis=(2, 3))
        
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
        self.spike_replaced[spike_ind] = np.sum(self.original_pacbed)
        
        if apply_remove:
            self.original_stack[spike_ind] = self.original_pacbed.copy()
            self.original_pacbed = np.mean(self.original_stack, axis=(0, 1))
        
    def find_center(self):
        
        Y, X = np.indices(self.original_pacbed.shape)
        com_y = np.sum(self.original_pacbed * Y) / np.sum(self.original_pacbed)
        com_x = np.sum(self.original_pacbed * X) / np.sum(self.original_pacbed)
        self.ct = [com_y, com_x]        

    def disk_extract(self, buffer_size=0):
        grad = np.gradient(self.original_pacbed)
        grad_map = grad[0]**2 + grad[1] **2
        grad_map = grad_map / np.max(grad_map)
        
        max_ind = np.unravel_index(np.argmax(grad_map, axis=None), grad_map.shape)
        self.least_R = ((max_ind[0]-self.ct[0])**2 + (max_ind[1]-self.ct[1])**2)**(1/2)
        
        print("radius of the BF disk = %.2f mrad"%(self.mrad_per_pixel*self.least_R))
        
        self.cropped_size = np.around(self.least_R + buffer_size).astype(int)

        if self.cropped_size > self.ct[0] or self.cropped_size > self.ct[1]:
            self.cropped_size = np.min(self.ct).astype(int)
  
        print("radius of the RoI = %.2f mrad"%(self.mrad_per_pixel*self.cropped_size))
        
        h_si = np.floor(self.ct[0]-self.cropped_size).astype(int)
        h_fi = np.ceil(self.ct[0]+self.cropped_size).astype(int)
        w_si = np.floor(self.ct[1]-self.cropped_size).astype(int)
        w_fi = np.ceil(self.ct[1]+self.cropped_size).astype(int)
        
        self.c_ct = [self.cropped_size, self.cropped_size]
        
        self.c_stack = self.original_stack[:, :, h_si:h_fi, w_si:w_fi].copy()
        self.c_shape = self.c_stack.shape
        self.c_pacbed = np.mean(self.c_stack, axis=(0, 1))
        
    def virtual_stem(self, BF, ADF):
        self.BF_detector = radial_indices(self.original_pacbed.shape, BF, self.mrad_per_pixel, center=self.ct)
        self.BF_stem = np.sum(np.multiply(self.original_stack, self.BF_detector), axis=(2, 3))
        
        self.ADF_detector = radial_indices(self.original_pacbed.shape, ADF, self.mrad_per_pixel, center=self.ct)
        self.ADF_stem = np.sum(np.multiply(self.original_stack, self.ADF_detector), axis=(2, 3))
        
    def symmetry_evaluation(self, angle, also_mirror=False):
        """
        Krajnak, M. and J. Etheridge, Proc Natl Acad Sci U S A, 2020. 117(45): p. 27805-27810.
        """
        rotation_stack = []
        r_correl = np.zeros(self.original_shape[:2])
        m_correl = np.zeros(self.original_shape[:2])
        
        ri = radial_indices(self.c_pacbed.shape, [0, self.cropped_size], 1, center=self.c_ct)
        
        angle = angle * np.pi/180
        alpha, beta = np.cos(angle), np.sin(angle)
        M = np.array([[alpha, beta, (1-alpha)*self.c_ct[1]-beta*self.c_ct[0]], 
                      [-beta, alpha, beta*self.c_ct[1]+(1-alpha)*self.c_ct[0]]])
        
        for i in range(self.original_shape[0]):
            for j in range(self.original_shape[1]):
                tmp_dp = self.c_stack[i,j,:,:].copy()
                newdata = np.multiply(rotation(tmp_dp, M), ri)
                rotation_stack.append(newdata)
                r_correl[i,j] = correlation(tmp_dp/np.max(tmp_dp), newdata)
                
                if also_mirror:
                    m_correl[i, j] = mirror(newdata, self.c_ct)
        
        rotation_stack = np.asarray(rotation_stack).reshape(self.c_shape)
        
        return rotation_stack, r_correl, m_correl
    
    def DPC(self, correct_rotation=True, n_theta=100, hpass=0.0, lpass=0.0):
        """
        Hachtel, J.A., J.C. Idrobo, and M. Chi, Adv Struct Chem Imaging, 2018. 4(1): p. 10. (https://github.com/hachteja/GetDPC)
        Lazic, I., E.G.T. Bosch, and S. Lazar, Ultramicroscopy, 2016. 160: p. 265-280.
        Savitzky, B.H., et al., arXiv preprint arXiv:2003.09523, 2020. (https://github.com/py4dstem/py4DSTEM)
        """
        
        Y, X = np.indices(self.c_pacbed.shape)
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

def load_binary_4D_stack(img_adr, datatype, original_shape, final_shape, log_scale=False):
    stack = np.fromfile(img_adr, dtype=datatype)
    stack = stack.reshape(original_shape)
    if log_scale:
        stack = np.log(stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]])
    else:
        stack = stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]]
    return stack