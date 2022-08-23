#!/usr/bin/python
'''
----------------------------------------------------------------------------------------
Image preprocessing
----------------------------------------------------------------------------------------
Alexis Coullomb
2022-08-15
alexis.coullomb.pro@gmail.com
----------------------------------------------------------------------------------------
'''

import numpy as np
from scipy import ndimage as ndi

def make_hot_pixels_mask(img, thresh):
    """
    Make a boolean mask indication positions of hot pixels.

    Parameters
    ----------
    img : ndarray
        2D image or (3D) stack of 2D images acquired
        in dark condition.
    thresh : int
        Threshold above which mean pixel intensities
        are considered due to being a hot pixel.

    Returns
    -------
    hot_pxs : 2D boolean array
        Position of hot pixels.
    """
    if len(img.shape) > 2:
        img_mean = img.mean(axis=0)
    else:
        img_mean = img
    hot_pxs = img_mean > thresh
    return hot_pxs

def correct_hot_pixels_2D(img, hot_pxs):
    """
    Correct hot pixels in a 2D image.

    Parameters
    ----------
    img : ndarray
        2D image.
    hot_pxs : 2D boolean array
        Position of hot pixels.

    Returns
    -------
    img_darkcorr : 2D array
        Image corrected for hot pixels.
    """
    img_darkcorr = img.copy()
    img_darkcorr[hot_pxs] = ndi.median_filter(img, 3)[hot_pxs]
    return img_darkcorr

def correct_hot_pixels_3D(img, hot_pxs):
    """
    Correct hot pixels in a 3D image.

    Parameters
    ----------
    img : ndarray
        3D image.
    hot_pxs : 2D boolean array
        Position of hot pixels.

    Returns
    -------
    img_darkcorr : 3D array
        Image corrected for hot pixels.
    """
    img_darkcorr = img.copy()
    for i in range(len(img_darkcorr)):
        img_darkcorr[i][hot_pxs] = ndi.median_filter(img[i], 3)[hot_pxs]
    return img_darkcorr

def rebin(arr, new_shape):
    """
    from: https://scipython.com/blog/binning-a-2d-array-in-numpy/
    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def denoise_2D(img, method, size):
    """
    Denoise a 2D image.

    Parameters
    ----------
    img : ndarray
        2D image.
    method : str
        Either 'gaussian', 'median' or 'bin'.
    size : int or float
        Size of gaussian kernel, or median or binning window.

    Returns
    -------
    img_denoised : 2D array
        Denoised image.
    """

    if method == "gaussian":
        img_denoised = np.zeros_like(img, dtype=np.float)
        img_denoised = ndi.gaussian_filter(img, size)
    elif method == "median":
        img_denoised = np.zeros_like(img, dtype=np.float)
        img_denoised = ndi.median_filter(img, size)
    elif method == "bin":
        # TODO: handle mismatch imgs shape / bin size
        # potentially by deleting or padding last rows / columns of img
        new_shape = np.array(img.shape) // size
        img_denoised = np.zeros_like(img, dtype=np.float)
        img_denoised = rebin(img, new_shape)
    else:
        img_denoised = img
    return img_denoised

def denoise_3D(img, method, size):
    """
    Denoise a 3D image.

    Parameters
    ----------
    img : ndarray
        3D image.
    method : str
        Either 'gaussian', 'median' or 'bin'.
    size : int or float
        Size of gaussian kernel, or median or binning window.

    Returns
    -------
    img_denoised : 3D array
        Denoised image.
    """

    if method == "gaussian":
        img_denoised = np.zeros_like(img, dtype=np.float)
        for i in range(len(img)):
            img_denoised[i, :, :] = ndi.gaussian_filter(img[i, :, :], size)
    elif method == "median":
        img_denoised = np.zeros_like(img, dtype=np.float)
        for i in range(len(img)):
            img_denoised[i, :, :] = ndi.median_filter(img[i, :, :], size)
    elif method == "bin":
        # TODO: handle mismatch imgs shape / bin size
        # potentially by deleting or padding last rows / columns of img
        new_shape = np.array(img[0].shape) // size
        img_denoised = np.zeros_like(img, dtype=np.float)
        for i in range(len(img)):
            img_denoised[i, :, :] = rebin(img[i, :, :], new_shape)
    else:
        img_denoised = img
    return img_denoised