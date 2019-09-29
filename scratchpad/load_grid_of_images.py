import os, struct, math
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import misc
from PIL import Image


def load_pose(filename):
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def resize_crop_image(image, new_image_dims):
    image_dims = [image.shape[1], image.shape[0]]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image

def load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize):
    color_image = misc.imread(color_file)
    depth_image = misc.imread(depth_file)
    pose = load_pose(pose_file)
    # preprocess
    depth_image = resize_crop_image(depth_image, depth_image_dims)
    color_image = resize_crop_image(color_image, color_image_dims)
    depth_image = depth_image.astype(np.float32) / 1000.0
    color_image =  np.transpose(color_image, [2, 0, 1])  # move feature to front
    color_image = normalize(torch.Tensor(color_image.astype(np.float32) / 255.0))
    return depth_image, color_image, pose

def load_scene_image_info_multi(filename, scene_name, image_path, depth_image_dims, color_image_dims, num_classes, color_mean, color_std):
    assert os.path.isfile(filename)
    fin = open(filename, 'rb')
    # read header
    width = struct.unpack('<Q', fin.read(8))[0]
    height = struct.unpack('<Q', fin.read(8))[0]
    max_num_images = struct.unpack('<Q', fin.read(8))[0]
    numElems = width * height * max_num_images
    frame_ids = struct.unpack('i'*numElems, fin.read(numElems*4))  #grid3<int>
    _width = struct.unpack('<Q', fin.read(8))[0]
    _height = struct.unpack('<Q', fin.read(8))[0]
    assert width == _width and height == _height
    numElems = width * height * 4 * 4
    world_to_grids = struct.unpack('f'*numElems, fin.read(numElems*4))  #grid2<mat4f>
    fin.close()
    # Seems to be IDs of frames pointing at each part in the map
    frame_ids = np.asarray(frame_ids, dtype=np.int32).reshape([max_num_images, height, width])
    # Seems to be transforms for each of these frames
    world_to_grids = np.asarray(world_to_grids, dtype=np.float32).reshape([height, width, 4, 4])
    # load data
    unique_frame_ids = np.unique(frame_ids)
    depth_images = {}
    color_images = {}
    poses = {}
    normalize = transforms.Normalize(mean=color_mean, std=color_std)
    for f in unique_frame_ids:
        if f == -1:
            continue
        depth_file = os.path.join(image_path, scene_name, 'depth', str(f) + '.png')
        color_file = os.path.join(image_path, scene_name, 'color', str(f) + '.jpg')
        pose_file = os.path.join(image_path, scene_name, 'pose', str(f) + '.txt')
        depth_image, color_image, pose = load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims, normalize)
        depth_images[f] = torch.from_numpy(depth_image.astype(np.float32))
        color_images[f] = color_image
        poses[f] = pose
    return depth_images, color_images, poses, frame_ids, world_to_grids


load_scene_image_info_multi("/mnt/1C562D12562CEDE8/DATASETS/3DMV/scenenn_test/scene0707_00.image",
                            "scene_name",
                            "image_path",
                            "depth_image_dims",
                            "color_image_dims",
                            "num_classes",
                            "color_mean",
                            "color_std")