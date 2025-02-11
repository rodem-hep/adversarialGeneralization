import numpy as np
from scipy.ndimage import gaussian_filter


def sum_1_norm(x, batch=True):
    """Normalise each image in a batch to sum 1 (summing over chanels as well!!!)
    or a single image
    Args:
        x (numpy.array): batch of images of formats (N, W, H, C), (N, W, H) or (N, W*H)
         or a single image
        batch (bool, optional): Specify if a given array is a batch or a single image
        (not clear from the dimensionality). Defaults to True.
    Returns:
        numpy.array: normalised images/image
    """
    if batch:
        if len(x.shape) == 4:  #
            return x / (np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))
        elif len(x.shape) == 3:
            return x / (np.sum(x, (1, 2))).reshape((-1, 1, 1))
        elif len(x.shape) == 2:
            return x / (np.sum(x, (1))).reshape((-1, 1))
        else:
            print("strange data format!")  # TODO add exception here
            return x
    else:
        return x / (np.sum(x))


def gaussian_smearing(x, sigma, batch=False, channels="first"):
    """Smear images wiht a gaussian kernel
    Args:
        x (numpy.array): batch of images of formats (N, W, H, C), (N, W, H) or a single
         image of formats (W, H, C), (W, H)
        sigma (float): standard deviation of a gaussian
        batch (bool, optional): Specify if a given array is a batch or a single image
        (not clear from the dimensionality). Defaults to True.
    Returns:
        numpy.array: smeared images/image
    """
    sima_dims = [sigma, sigma]
    if channels == "last":
        sima_dims.append(0)
    elif channels == "first":
        sima_dims.insert(0, 0)

    if batch:
        sima_dims.insert(0, 0)

    return gaussian_filter(x, sigma=sima_dims)


def abs_fft(sample):
    return np.abs(np.fft.fft2(sample)).astype(np.float32)


def pixel_pow(x, n):
    if n == 0:
        return x != 0
    else:
        return x**n


def scale(x, scl):
    return x * scl


def reproc_heavi(x):
    x[x > 0] = 1
    return sum_1_norm(x)


def reproc_log(x, scl):
    x = scl * x
    x = np.log(x + 1)
    return x / (np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))


def image_transformations(image, image_transform):
    "Transforms a given image according to the given image_transform"
    for transform in image_transform:
        temp = transform.split(" ")
        transform = temp[0]
        param = []
        for i in range(1, len(temp)):
            param.append(float(temp[i]))
        if transform == "sum_1_norm":
            image = sum_1_norm(image, batch=False)
        elif transform == "gaussian_smearing":
            image = gaussian_smearing(image, sigma=param[0], batch=False)
        elif transform == "pixel_pow":
            image = pixel_pow(image, n=param[0])
        elif transform == "reproc_heavi":
            image = reproc_heavi(image)
        elif transform == "reproc_log":
            image = reproc_log(image, scl=param[0])
        elif transform == "scale":
            image = scale(image, scl=param[0])
        elif transform == "abs_fft":
            image = abs_fft(image)
        else:
            print("No transformation applied")
    return image
