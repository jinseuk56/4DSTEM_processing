import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy import ndimage

def radial_stats(image, center=None, var=True):
    """
    Calculate the radial average and (optional) radial variance of a 2D image.
    """
    y, x = np.indices(image.shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
        
    r = np.hypot(y - center[0], x - center[1])

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
    """
    Apply radial average/variance transformation to a 4D stack.
    Returns:
        radial_avg_stack of shape (Ny, Nx, Nr)
        radial_var_stack of shape (Ny, Nx, Nr)
    """
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
        shortest = np.min(len_profile)
        for i in range(len(radial_avg_stack)):
            radial_avg_stack[i] = radial_avg_stack[i][:shortest]
            if also_variance:
                radial_var_stack[i] = radial_var_stack[i][:shortest]

    radial_avg_stack = np.asarray(radial_avg_stack).reshape(data_shape[0], data_shape[1], -1)
    if also_variance:
        radial_var_stack = np.asarray(radial_var_stack).reshape(data_shape[0], data_shape[1], -1)

    return radial_avg_stack, radial_var_stack

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

def angular_correlation_fft(values):
    """
    Fast circular angular correlation using 1D FFT.
    Calculates: C(l) = <x[m] * x[(m - l) % N]>_m / <x>^2 - 1
    
    Parameters:
    -----------
    values : ndarray, shape (..., N_angles)
        The intensity profile along the azimuthal angles.
    """
    N = values.shape[-1]
    
    # Compute circular autocorrelation using Wiener-Khinchin theorem:
    # corr = IFFT(|FFT(x)|^2)
    fft_vals = np.fft.fft(values, axis=-1)
    power_spec = np.abs(fft_vals)**2
    autocorr = np.real(np.fft.ifft(power_spec, axis=-1)) / N
    
    # Calculate normalization factor (mean squared of values along the angle axis)
    mean_vals = np.mean(values, axis=-1, keepdims=True)
    mean_vals_sq = mean_vals ** 2
    # Prevent division by zero
    mean_vals_sq = np.where(mean_vals_sq == 0, 1.0, mean_vals_sq)
    
    ang_corr = (autocorr / mean_vals_sq) - 1
    return ang_corr

def angular_correlation_direct(values, method='linear'):
    """
    Compute direct angular correlation (either using standard linear boundaries with nanmean, 
    matching the original code's implementation, or circular wrap-around).
    
    Parameters:
    -----------
    values : ndarray, shape (N_angles,)
        The intensity profile along the azimuthal angles.
    method : str
        'linear': replicates original code's linear autocorrelation with triangular mask (nanmean, length decreases with lag).
        'circular': standard circular autocorrelation using direct rolling.
    """
    N = len(values)
    
    # Reconstruct value_stack using list comprehension (much faster than loop np.vstack)
    value_stack = np.array([np.roll(values, l) for l in range(N)])
    
    if method == 'linear':
        # Replicate original implementation with upper-triangular nan masking
        tril_mask = np.ones((N, N))
        tril_mask = np.triu(tril_mask, 0)
        tril_mask[np.where(tril_mask == 0)] = np.nan
        
        ang_corr = np.multiply(value_stack, values[np.newaxis, :])
        ang_corr = np.multiply(np.triu(ang_corr, 0), tril_mask)
        
        value_avgsq = np.mean(values)**2 if np.mean(values) != 0 else 1.0
        ac_spectrum = np.nanmean(ang_corr, axis=1)
        ac_spectrum = (ac_spectrum / value_avgsq) - 1
        return ac_spectrum
        
    elif method == 'circular':
        ang_corr = np.multiply(value_stack, values[np.newaxis, :])
        ac_spectrum = np.mean(ang_corr, axis=1)
        value_avgsq = np.mean(values)**2 if np.mean(values) != 0 else 1.0
        ac_spectrum = (ac_spectrum / value_avgsq) - 1
        return ac_spectrum
    else:
        raise ValueError(f"Unknown method: {method}")
