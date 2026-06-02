import numpy as np
import tifffile

def load_binary_4D_stack(img_adr, datatype, original_shape, final_shape, log_scale=False):
    """
    Load raw binary 4D-STEM stack (e.g. from EMPAD).
    """
    stack = np.fromfile(img_adr, dtype=datatype)
    stack = stack.reshape(original_shape)
    if log_scale:
        stack = np.log(stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]])
    else:
        stack = stack[:final_shape[0], :final_shape[1], :final_shape[2], :final_shape[3]]
    return stack

def load_empad_raw(img_adr, datatype="float32", f_shape=None, flip_axis=2):
    """
    Load EMPAD raw binary data and apply standard preprocessing (flipping and nan replacing).
    """
    if f_shape is None:
        f_shape = [128, 128, 128, 128]
    # EMPAD detector files are 128x130 pixels per frame (with 2 extra columns for metadata/reference)
    o_shape = [f_shape[0], f_shape[1], f_shape[2] + 2, f_shape[3]]
    
    stack = load_binary_4D_stack(img_adr, datatype, o_shape, f_shape, log_scale=False)
    if flip_axis is not None:
        stack = np.flip(stack, axis=flip_axis)
    stack = np.nan_to_num(stack)
    return stack

def save_as_tiff(img_adr, data):
    """
    Save data to TIFF format.
    """
    tifffile.imwrite(img_adr, data)
