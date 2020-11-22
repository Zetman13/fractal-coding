import numpy as np
from numba import njit, prange
from src.metrics import timing_val


@njit(parallel=True)
def encode(img, block_size):
    exp_size = block_size * 2
    i_size = img.shape[0]
    j_size = img.shape[1]
    transformed_blocks = get_transformed_blocks(img, exp_size)
    transforms = [[(0, 0, 0, 0, 0.0, 0.0) for _ in range(j_size // block_size)] for _ in range(i_size // block_size)]
    for i in prange(i_size // block_size):
        for j in prange(j_size // block_size):
            min_diff = np.inf
            r = img[i*block_size:i*block_size + block_size, j*block_size:j*block_size + block_size]
            r_code = get_code(r)
            for x, y, direction, angle, d, d_code in transformed_blocks:
                if np.count_nonzero(r_code != d_code) == 0:
                    k, b = fit_contrast_brightness(r, d)
                    d = k*d + b
                    diff = np.sum(np.square(r - d))
                    if diff < min_diff:
                        min_diff = diff
                        transforms[i][j] = (x, y, direction, angle, k, b)
    return transforms


def decode(transforms, block_size, n_iters=8):
    exp_size = block_size * 2
    height = len(transforms) * block_size
    width = len(transforms[0]) * block_size
    results = [np.random.randint(0, 256, (height, width))]
    result = np.zeros((height, width))
    for _ in range(n_iters):
        for i in range(0, height, block_size):
            for j in prange(0, width, block_size):
                x, y, direction, angle, k, b = transforms[i // block_size][j // block_size]
                d = avg_pool(results[-1][x*exp_size:(x+1)*exp_size, y*exp_size:(y+1)*exp_size])
                r = transform(d, direction, angle, k, b)
                result[i:i+block_size, j:j+block_size] = r
        results.append(normalize(result))
    return results


@njit
def avg_pool(img, factor=2):
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in prange(result.shape[0]):
        for j in prange(result.shape[1]):
            result[i, j] = np.mean(img[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    return result


@njit
def rot(img, angle):
    result = img
    for i in range(angle):
        result = np.flipud(result.T)
    return result


@njit
def flip(img, direction):
    return img[::direction, :]


@njit
def transform(img, direction, angle, k=1.0, b=0.0):
    return k*rot(flip(img, direction), angle) + b


@njit
def normalize(img):
    return np.maximum(np.minimum(np.rint(img), 255), 0).astype(np.uint8)


@njit
def fit_contrast_brightness(r, d):
    a1 = np.ones((d.size, 1))
    a2 = d.copy().reshape((d.size, 1))
    a = np.concatenate((a1, a2), axis=1)
    b = r.flatten().astype(np.float64)
    result, _, _, _ = np.linalg.lstsq(a, b)
    return result[1], result[0]


@njit
def get_code(d):
    mean_d = d.mean()
    d1 = d[:d.shape[0] // 2, :d.shape[1] // 2]
    d2 = d[:d.shape[0] // 2, d.shape[1] // 2:]
    d3 = d[d.shape[0] // 2:, :d.shape[1] // 2]
    d4 = d[d.shape[0] // 2:, d.shape[1] // 2:]
    mean_d1, mean_d2, mean_d3, mean_d4 = d1.mean(), d2.mean(), d3.mean(), d4.mean()
    return np.array((mean_d1 > mean_d, mean_d2 > mean_d, mean_d3 > mean_d, mean_d4 > mean_d))


@njit
def get_transformed_blocks(img, block_size):
    candidates = [(direction, angle) for direction in [1, -1] for angle in [0, 1, 2, 3]]
    transformed_blocks = []
    for x in prange(img.shape[0] // block_size):
        for y in prange(img.shape[1] // block_size):
            d = avg_pool(img[x*block_size:x*block_size+block_size, y*block_size:y*block_size+block_size])
            for direction, angle in candidates:
                transformed_d = transform(d, direction, angle)
                transformed_blocks.append((x, y, direction, angle, transformed_d, get_code(transformed_d)))
    return transformed_blocks
