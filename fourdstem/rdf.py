import numpy as np
from scipy.integrate import simpson
from py4DSTEM.process.utils import single_atom_scatter

def rif_to_rdf(rif, r_list, k_list):
    """
    Convert reduced intensity function (RIF) to radial distribution function G(r)
    using Simpson's integration rule.
    G(r) = 8 * pi * integral( RIF(k) * sin(2 * pi * r * k) dk )
    """
    gr = []
    dk = k_list[1] - k_list[0]
    for r in r_list:
        sin_rk = np.sin(2 * np.pi * r * k_list)
        rif_sin = rif * sin_rk
        gr_tmp = 8 * np.pi * simpson(rif_sin, dx=dk)
        gr.append(gr_tmp)
    return np.asarray(gr)

def calculate_scattering_factors(elements, compositions, q_coords, method='standard'):
    """
    Calculate the mean square scattering factor <f^2> and square of the mean scattering factor <f>^2.
    
    Parameters:
    -----------
    elements : array_like
        Atomic numbers Z of the elements in the sample.
    compositions : array_like
        Atomic fraction compositions c_i of the elements.
    q_coords : array_like
        Scattering vector q coordinates (typically in 1/A).
    method : str
        'standard' for scientifically correct electron diffraction theory:
            - subtracted_term = <f^2> = sum(c_i * f_i^2) (atomic/independent background)
            - divisor_term = <f>^2 = (sum(c_i * f_i))^2 (normalization factor)
        'legacy' for replicating the original codebase which contains name confusion and Z-multiplication:
            - subtracted_term = (sum(c_i * Z_i * f_i))^2
            - divisor_term = sum((c_i * Z_i * f_i)^2)
    """
    elements = np.asarray(elements)
    compositions = np.asarray(compositions)
    
    if method == 'standard':
        # Unscaled atomic form factors f_i(q)
        f_i_list = []
        for i, Z in enumerate(elements):
            AFF = single_atom_scatter(elements=[Z], composition=[1.0], q_coords=q_coords, units='A')
            AFF.get_scattering_factor()
            f_i_list.append(AFF.fe)
        f_i_list = np.array(f_i_list) # shape (n_elements, n_q)
        
        # c_i * f_i(q) and c_i * f_i(q)^2
        c_i = compositions[:, np.newaxis]
        f_mean = np.sum(c_i * f_i_list, axis=0)
        f_mean_sq = f_mean ** 2 # <f>^2
        f_sq_mean = np.sum(c_i * (f_i_list ** 2), axis=0) # <f^2>
        
        # In correct physics: independent scattering background (<f^2>) is subtracted,
        # and coherent normalization factor (<f>^2) is divided by.
        return f_sq_mean, f_mean_sq
        
    elif method == 'legacy':
        # Legacy method which contains name confusion and scales form factors by Z.
        # single_atom_scatter returns compositions[i]*f_i by default when composition is passed.
        AFFs = []
        for i, Z in enumerate(elements):
            AFF = single_atom_scatter(elements=[Z], composition=[compositions[i]], q_coords=q_coords, units='A')
            AFF.get_scattering_factor()
            AFFs.append(AFF.fe)
        AFFs = np.array(AFFs) # shape (n_elements, n_q)
        
        # Multiply by Z (atomic numbers)
        scaled_AFFs = AFFs * elements[:, np.newaxis]
        
        # Swapped and scaled terms:
        # subtracted_term = (sum(c_i * Z_i * f_i))^2
        subtracted_term = np.sum(scaled_AFFs, axis=0)**2
        # divisor_term = sum((c_i * Z_i * f_i)^2)
        divisor_term = np.sum(scaled_AFFs**2, axis=0)
        
        return subtracted_term, divisor_term
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_damping_filter(filter_type, N):
    """
    Natively calculate the damping/window filter of length N.
    Supported types: 'boxcar', 'triangular', 'trapezoidal', 'Happ-Genzel', '3TEM BH', '4TEM BH'.
    """
    if isinstance(filter_type, (int, float)):
        filter_types = ["boxcar", "triangular", "trapezoidal", "Happ-Genzel", "3TEM BH", "4TEM BH"]
        idx = int(filter_type)
        if 0 <= idx < len(filter_types):
            filter_type_str = filter_types[idx]
        else:
            filter_type_str = "boxcar"
    else:
        filter_type_str = str(filter_type)
        
    filter_type_lower = filter_type_str.lower().replace("-", " ").replace("_", " ")
    
    i = np.arange(N)
    
    if 'boxcar' in filter_type_lower:
        filt = np.ones(N)
        filt[-1] = 0.0
    elif 'triangular' in filter_type_lower or 'trangular' in filter_type_lower:
        filt = 1.0 - i / N
    elif 'trapezoidal' in filter_type_lower:
        N_flat = int(0.4 * N)
        filt = np.ones(N)
        filt[N_flat:] = (N - i[N_flat:]) / (N - N_flat)
    elif 'happ genzel' in filter_type_lower:
        filt = 0.54 + 0.46 * np.cos(np.pi * i / N)
    elif '3tem' in filter_type_lower or '3-term' in filter_type_lower:
        a0, a1, a2 = 0.42323, 0.49755, 0.07922
        filt = a0 + a1 * np.cos(np.pi * i / N) + a2 * np.cos(2 * np.pi * i / N)
    elif '4tem' in filter_type_lower or '4-term' in filter_type_lower:
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        filt = a0 + a1 * np.cos(np.pi * i / N) + a2 * np.cos(2 * np.pi * i / N) + a3 * np.cos(3 * np.pi * i / N)
    else:
        filt = np.ones(N)
        filt[-1] = 0.0
        
    return filt

def get_filter_type_from_filename(filename):
    """
    Parse a filename or path to determine which damping filter to use.
    """
    if not filename:
        return 'boxcar'
    import os
    fn_lower = os.path.basename(str(filename)).lower()
    if 'boxcar' in fn_lower:
        return 'boxcar'
    elif 'triangular' in fn_lower or 'trangular' in fn_lower:
        return 'triangular'
    elif 'trapezoidal' in fn_lower:
        return 'trapezoidal'
    elif 'happ' in fn_lower:
        return 'Happ-Genzel'
    elif '3tem' in fn_lower:
        return '3TEM BH'
    elif '4tem' in fn_lower:
        return '4TEM BH'
    return 'boxcar'

def calculate_atomic_form_factors(elements, compositions, q_coords):
    """
    Calculate the atomic form factors of specified elements scaled by their composition fraction.
    """
    from py4DSTEM.process.utils import single_atom_scatter
    AFFs = []
    for i, Z in enumerate(elements):
        AFF = single_atom_scatter(elements=[Z], composition=[compositions[i]], q_coords=q_coords, units='A')
        AFF.get_scattering_factor()
        AFFs.append(AFF.fe)
    return np.array(AFFs)

def calculate_rif(intensity, subtracted_term, divisor_term, alpha, N, q_coords, damping_filter=None, low_cut=0.0, high_cut=1.0):
    """
    Calculate the reduced intensity function (RIF).
    RIF(q) = q * [ (I(q) + alpha - N * subtracted_term) / (N * divisor_term) ] * damping_filter
    """
    # Calculate initial fraction
    num = intensity + alpha - N * subtracted_term
    den = N * divisor_term
    
    # Avoid divide by zero
    den = np.where(den == 0, 1.0, den)
    
    rif = (num / den) * q_coords
    n_q = len(q_coords)
    
    # Apply damping/window filter if provided
    if damping_filter is not None:
        if isinstance(damping_filter, (np.ndarray, list)):
            filt = np.asarray(damping_filter).copy()
        elif isinstance(damping_filter, str) and (damping_filter.endswith('.tif') or damping_filter.endswith('.tiff')):
            import os
            if os.path.exists(damping_filter):
                import tifffile
                filt = tifffile.imread(damping_filter)
                if filt.ndim > 1:
                    filt = filt[5] if len(filt) > 5 else filt[0]
                filt = filt.copy()
            else:
                filter_type = get_filter_type_from_filename(damping_filter)
                filt = calculate_damping_filter(filter_type, n_q)
        else:
            filt = calculate_damping_filter(damping_filter, n_q)
            
        filt[:int(low_cut * len(filt))] = 0
        filt[int(high_cut * len(filt)):] = 0
        
        # Interpolate filter to match q_coords length if needed
        if len(filt) != n_q:
            ind = np.linspace(0, len(filt) - 1, n_q).astype(np.int16)
            filt = filt[ind]
            
        rif = rif * filt
        
    return rif


def gr_analysis(intensity, elements, compositions, q_coords, r_list, alpha, N, damping_filter=None, low_cut=0.0, high_cut=1.0, method='standard'):
    """
    Full workflow to compute the radial distribution function G(r) from an experimental intensity profile.
    """
    subtracted_term, divisor_term = calculate_scattering_factors(elements, compositions, q_coords, method=method)
    rif = calculate_rif(intensity, subtracted_term, divisor_term, alpha, N, q_coords, damping_filter, low_cut, high_cut)
    gr = rif_to_rdf(rif, r_list, q_coords)
    return rif, gr

def fit_rdf_parameters(intensity, subtracted_term, fit_range=None, alpha_range=(-5, 5, 0.01)):
    """
    Fit scale factor N and correction term alpha to match experimental intensity
    at the high-q range (fit_range) to the independent scattering factor (subtracted_term).
    """
    if fit_range is None:
        fit_range = [int(0.7 * len(intensity)), int(0.9 * len(intensity))]
    
    q_start, q_end = fit_range
    int_slice = intensity[q_start:q_end]
    sub_slice = subtracted_term[q_start:q_end]
    
    mean_int = np.mean(int_slice)
    mean_sub = np.mean(sub_slice)
    fit_dif = mean_int - mean_sub
    
    alphas = np.arange(alpha_range[0], alpha_range[1], alpha_range[2])
    best_error = float('inf')
    best_alpha = 0.0
    best_N = 1.0
    
    for alpha in alphas:
        denom = mean_sub + alpha * fit_dif
        if denom == 0:
            denom = 1e-9
        N = mean_int / denom
        if N < 0:
            N = 0.0
            
        fit_profile = (sub_slice + alpha * fit_dif) * N
        error = np.linalg.norm(int_slice - fit_profile)
        if error < best_error:
            best_error = error
            best_alpha = alpha
            best_N = N
            
    return best_alpha, best_N
