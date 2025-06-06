import numpy as np

import radpsd


def main():
    contour = np.random.randn(100, 100).astype(np.float32)  # Simulated contour data

    print(f'contour.shape: {contour.shape}')
    print(f'contour.dtype: {contour.dtype}')
    print(f'contour.min(): {contour.min()}, contour.max(): {contour.max()}')

    cmap = radpsd.apply_color_map(contour, 'viridis')
    print(f'cmap.shape: {cmap.shape}')
    print(f'cmap.dtype: {cmap.dtype}')
    print(f'cmap.min(): {cmap.min()}, cmap.max(): {cmap.max()}')


if __name__ == '__main__':
    main()
