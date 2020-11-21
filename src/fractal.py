import numpy as np
from numba import njit, prange
from src.metrics import timing_val


@timing_val
@njit(parallel=True)
def encode(img, block_size):
    exp_size = block_size * 2
    i_size = img.shape[0]
    j_size = img.shape[1]
    transformed_blocks = get_transformed_blocks(img, exp_size)
    transforms = [[(0, 0, 0, 0, 0.0, 0.0) for _ in range(0, j_size, block_size)] for _ in range(0, i_size, block_size)]
    for i in prange(i_size // block_size):
        for j in prange(j_size // block_size):
            min_d = np.inf
            R = img[i*block_size:i*block_size + block_size, j*block_size:j*block_size + block_size]
            for x, y, direction, angle, D in transformed_blocks:
                k, b = fit_contrast_brightness(R, D)
                D = k*D + b
                d = np.sum(np.square(R - D))
                if d < min_d:
                    min_d = d
                    transforms[i][j] = (x, y, direction, angle, k, b)
    return transforms


@timing_val
def decode(transforms, block_size, n_iters=8):
    exp_size = block_size * 2
    height = len(transforms) * block_size
    width = len(transforms[0]) * block_size
    results = [np.random.randint(0, 256, (height, width))]
    result = np.zeros((height, width))
    for _ in range(n_iters):
        for i in range(0, height, block_size):
            for j in prange(0, width, block_size):
                x, y, flip, angle, k, b = transforms[i // block_size][j // block_size]
                D = avg_pool(results[-1][x*exp_size:(x+1)*exp_size, y*exp_size:(y+1)*exp_size])
                R = transform(D, flip, angle, k, b)
                result[i:i+block_size, j:j+block_size] = R
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
def fit_contrast_brightness(R, D):
    a1 = np.ones((D.size, 1))
    a2 = D.copy().reshape((D.size, 1))
    A = np.concatenate((a1, a2), axis=1)
    b = R.flatten().astype(np.float64)
    result, _, _, _ = np.linalg.lstsq(A, b)
    return result[1], result[0]


@njit
def get_transformed_blocks(img, block_size):
    candidates = [[direction, angle] for direction in [1, -1] for angle in [0, 1, 2, 3]]
    transformed_blocks = []
    for x in prange((img.shape[0] - block_size) // block_size + 1):
        for y in prange((img.shape[1] - block_size) // block_size + 1):
            D = avg_pool(img[x*block_size:x*block_size+block_size, y*block_size:y*block_size+block_size])
            for direction, angle in candidates:
                transformed_blocks.append((x, y, direction, angle, transform(D, direction, angle)))
    return transformed_blocks
