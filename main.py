import matplotlib.pyplot as plt
from pathlib import Path
import sys

from src.fractal import encode, decode
from src.metrics import psnr, timing_val


@timing_val
def encode_decode(img_name, output_dir, block_size):
    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
    img = plt.imread(f"input/{img_name}.bmp")
    code = encode(img, block_size)
    print(f"Coding size: {sys.getsizeof(code)} bytes")
    iters = decode(code, block_size, n_iters=16)

    for i, step in enumerate(iters):
        plt.imsave(f"{output_dir}/{img_name}_{i}.bmp", step, cmap='gray')
    return iters


if __name__ == '__main__':
    img_name = 'Lena'
    block_size = 8
    iters_2 = encode_decode(img_name, f"output/{img_name}_{block_size}", block_size)
    psnr_result = psnr(plt.imread(f"input/{img_name}.bmp"), iters_2[-1])
    print(f"PSNR: {psnr_result}")
