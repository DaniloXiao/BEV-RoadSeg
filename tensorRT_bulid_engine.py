import argparse
import tensorrt as trt
import  os, cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=1):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = self.load_data(training_data)
        self.batch_size = batch_size
        self.current_index = 0
        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    # Returns a numpy buffer of shape (num_images, 3, 288, 224)
    def load_data(self, datapath):
        imgs = os.listdir(datapath)
        dataset = []
        for data in imgs:
            rgb_image = cv2.cvtColor(cv2.imread(datapath + data), cv2.COLOR_BGR2RGB)
            oriHeight, oriWidth, _ = rgb_image.shape
            use_size = (224, 288)
            rgb_image = cv2.resize(rgb_image, use_size)
            img = rgb_image.astype(np.float32) / 255
            img=np.transpose(img, (2, 0, 1))
            dataset.append(img)
            print(np.array(dataset).shape)
        return np.array(dataset)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_engine(onnx_path, trt_file_path, mode, int8_data_path, verbose=False):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        print('Loading TRT file from path {}...'.format(trt_file_path))
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 30
        print('Loading ONNX file from path {}...'.format(onnx_path))
        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print("num layers:", network.num_layers)
                return None
        if mode == 'fp16':
            builder.fp16_mode = True
            network.get_input(0).shape = [1, 3, 288, 224]
            trt_file_path = trt_file_path[:-4] + '_fp16.trt'
            print("build fp16 engine...")
        elif mode == 'fp32':
            network.get_input(0).shape = [1, 3, 288, 224]
            trt_file_path = trt_file_path[:-4] + '_fp32.trt'
            print("build fp32 engine...")
        else:
            # build an int8 engine
            calibration_cache = "calibration.cache"
            calib = EntropyCalibrator(int8_data_path, cache_file=calibration_cache)
            builder.int8_mode = True
            builder.int8_calibrator = calib
            network.get_input(0).shape = [1, 3, 288, 224]
            trt_file_path = trt_file_path[:-4] + '_int8.trt'
            print("build int8 engine...")
        engine = builder.build_cuda_engine(network)
        print("engine:", engine)
        print("Completed creating Engine")
        with open(trt_file_path, "wb") as f:
            f.write(engine.serialize())
    return engine



def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', action='store_true')
    parser.add_argument(
        '--onnx_path', type=str, default='./checkpoints/kitti/kitti_net_RoadSeg.onnx')
    parser.add_argument(
        '--trt_path', type=str, default='RoadSeg.trt')
    parser.add_argument(
        '--mode', type=str, default='int8', help='fp32 fp16 int8')
    parser.add_argument(
        '--int8_calibration_path', type=str,
        default='./datasets/kitti/training/image_2/',
        help='set if you want to do int8 inference')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # paths and configs
    configs = set_config()
    # build engine
    engine = build_engine(configs.onnx_path, configs.trt_path, configs.mode, configs.int8_calibration_path, configs.verbose)
