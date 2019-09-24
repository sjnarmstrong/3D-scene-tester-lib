from PIL import Image
import numpy as np


def align_images(src_img, width, height):
    # Todo this is heavy simplified
    pil_src = Image.fromarray(src_img).resize((width, height), Image.BICUBIC)
    return np.array(pil_src, dtype=np.uint8)
