import numpy as np
from PIL import Image

def load_depth_image(path):
    """Loads an unprocessed depth map extracted from the raw dataset."""
    with open(path, 'rb') as f:
        return Image.fromarray(read_pgm(f), mode='I')

def load_color_image(path):
    """Loads an unprocessed color image extracted from the raw dataset."""
    with open(path, 'rb') as f:
        return Image.fromarray(read_ppm(f), mode='RGB')

def read_pgm(pgm_file):
    """Reads a PGM file from a buffer.

    Returns a numpy array of the appropiate size and dtype.
    """

    # First line contains some image meta-info
    p5, width, height, depth = pgm_file.readline().split()

    # Ensure we're actually reading a P5 file
    assert p5 == b'P5'
    assert depth == b'65535', "Only 16-bit PGM files are supported"

    width, height = int(width), int(height)

    data = np.frombuffer(pgm_file.read(), dtype='<u2', count=width*height)

    return data.reshape(height, width).astype(np.uint32)

def read_ppm(ppm_file):
    """Reads a PPM file from a buffer.

    Returns a numpy array of the appropiate size and dtype.
    """

    p6, width, height, depth = ppm_file.readline().split()

    assert p6 == b'P6'
    assert depth == b'255', "Only 8-bit PPM files are supported"

    width, height = int(width), int(height)

    data = np.frombuffer(ppm_file.read(), dtype=np.uint8, count=3*width*height)

    return data.reshape(height, width, 3)
