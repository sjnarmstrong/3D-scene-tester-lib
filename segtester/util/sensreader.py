import struct
import numpy as np
import zlib
import imageio

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


class RGBDFrame(object):

    def __init__(self, file_handle, color_compression_type, depth_compression_type, depth_shape):
        self.camera_to_world = np.asarray(struct.unpack('f' * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(
            4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

        self.get_color_image = lambda: self.decompress_color(color_compression_type)
        self.get_depth_image = lambda: np.frombuffer(self.decompress_depth(depth_compression_type), dtype=np.uint16)\
            .reshape(depth_shape)

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:

    def __init__(self, filename):
        self.version = 4
        self.filename = filename
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = f.read(strlen).decode('utf-8')
            self.intrinsic_color = np.array(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
            self.extrinsic_color = np.array(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
            self.intrinsic_depth = np.array(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
            self.extrinsic_depth = np.array(struct.unpack('f' * 16, f.read(16 * 4))).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]	 # conversion from float[m] to ushort (typically 1000f)
            self.num_frames = struct.unpack('Q', f.read(8))[0]
            self.frame_offset = f.tell()  # OFFSET IN FILE TO START OF FRAME DATA

    def get_image_generator(self):
        with open(self.filename, 'rb') as f:
            f.seek(self.frame_offset)
            for i in range(self.num_frames):
                yield RGBDFrame(f, self.color_compression_type, self.depth_compression_type,
                                (self.depth_height, self.depth_width))

    def __repr__(self):
        return f"""SensorData:
    sensor_name: {self.sensor_name}
    intrinsic_color: 
{self.intrinsic_color}
    extrinsic_color: 
{self.extrinsic_color}
    intrinsic_depth: 
{self.intrinsic_depth}
    extrinsic_depth: 
{self.extrinsic_depth}
    color_compression_type: {self.color_compression_type}
    depth_compression_type: {self.depth_compression_type}
    color_width: {self.color_width}
    color_height: {self.color_height}
    depth_width: {self.depth_width}
    depth_height: {self.depth_height}
    depth_shift: {self.depth_shift}
    num_frames: {self.num_frames}
    frame_offset: {self.frame_offset}
------------------------------------------------------------------------------------------------------------------------
"""


if __name__ == "__main__":
    from PIL import Image
    from time import sleep
    _sdata = SensorData("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans/scene0706_00/scene0706_00.sens")
    print(_sdata)
    for i, _image in enumerate(_sdata.get_image_generator()):
        if i % 100 !=0:
            continue
        Image.fromarray(_image.get_color_image()).show()
        d_img = _image.get_depth_image()
        coloured_img = np.zeros(d_img.shape+(3,), dtype=np.uint8)
        non_zero_d_mask = d_img > 0
        non_zero_d = d_img[non_zero_d_mask]
        x_wh, y_wh = np.where(non_zero_d_mask)
        d_img_min = non_zero_d.min()
        d_img_max = non_zero_d.max()
        d_img_range = d_img_max- d_img_min

        mask = non_zero_d < d_img_range/3 + d_img_min
        masked_dimage = non_zero_d[mask]
        coloured_img[x_wh[mask], y_wh[mask], 2] = 100*(masked_dimage-masked_dimage.min())/masked_dimage.max()+60
        coloured_img[x_wh[mask], y_wh[mask], 1] = 100*(masked_dimage-masked_dimage.min())/masked_dimage.max()+60
        coloured_img[x_wh[mask], y_wh[mask], 0] = 100*(masked_dimage-masked_dimage.min())/masked_dimage.max()+60

        mask = non_zero_d >= 2*d_img_range/3 + d_img_min
        masked_dimage = non_zero_d[mask]
        coloured_img[x_wh[mask], y_wh[mask], 0] = 100*(masked_dimage-masked_dimage.min())/masked_dimage.max()+60

        mask = np.logical_and(non_zero_d < 2*d_img_range/3 + d_img_min, non_zero_d >= d_img_range/3 + d_img_min)
        masked_dimage = non_zero_d[mask]
        coloured_img[x_wh[mask], y_wh[mask], 1] = 100*(masked_dimage-masked_dimage.min())/masked_dimage.max()+60
        coloured_img[x_wh[mask], y_wh[mask], 0] = 100*(masked_dimage-masked_dimage.min())/masked_dimage.max()+60

        #Image.fromarray(coloured_img).show()
        sleep(1)

