import os
from data import CreateDataLoader
import torch
import numpy as np
import cv2
import argparse
import pycuda.driver as cuda
import pycuda.autoinit#need
import tensorrt as trt
import time
from util.util import confusion_matrix, getScores,merge_rgb_to_bev,tensor2labelim

#python tensorRT_detect.py --dataroot datasets/kitti --dataset kitti --name kitti --no_label --epoch kitti --output_video_fn trt_video


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_static(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, default='datasets/kitti',
                        help='path to images, should have training, validation and testing')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--epoch', type=str, default='latest', help='chooses which epoch to load')
    parser.add_argument('--dataset', type=str, default='kitti', help='chooses which dataset to load.')
    parser.add_argument('--use_sne', action='store_true', help='chooses if using sne')
    parser.add_argument('--useWidth', type=int, default=224, help='scale images to this width')
    parser.add_argument('--useHeight', type=int, default=288, help='scale images to this height')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--results_dir', type=str, default='./testresults/', help='saves results here.')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test')
    parser.add_argument('--no_label', action='store_true', help='chooses if we have gt labels in testing phase')
    parser.add_argument('--view-img', action='store_true', default=True, help='show results')
    parser.add_argument('--save-video', action='store_true', default=True,help='if true, save video, otherwise save image results')
    parser.add_argument('--output_video_fn', type=str, default='detect', metavar='PATH',
                        help='the video filename if the output format is save-video')
    parser.add_argument('--seed', type=int, default=0, help='seed for random generators')
    parser.add_argument('--trt_path', type=str, default='./RoadSeg_int8.trt')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt  = set_config()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    save_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#images = %d' % dataset_size)

    time_torch, seen_torch = 0, 0
    out_cap=None
    palet_file = 'datasets/palette.txt'
    impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3 * 256))
    conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=np.float)

    engine = load_engine(opt.trt_path, verbose=False)
    h_inputs, h_outputs, bindings, stream = allocate_buffers(engine)

    with engine.create_execution_context() as context:
        for i, data in enumerate(dataset):
            gt = data['label'].int().numpy()

            t1 = time.time()
            img = data['rgb_image'][0].numpy()
            h_inputs[0].host = img
            trt_outputs = do_inference_static(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs,
                                              stream=stream, batch_size=1)
            pred = torch.from_numpy(trt_outputs[0].reshape(1, 2, 288, 224))
            oriSize = (data['oriSize'][0].item(), data['oriSize'][1].item())
            image_name = data['path'][0]
            img_channel = tensor2labelim(pred, impalette)
            im = img_channel[:, :, (1, 0, 2)]# green channel to blue channel
            im = cv2.resize(im, oriSize)
            t2 = time.time()
            time_torch += time.time() - t1
            seen_torch += 1

            # cv2.imwrite(os.path.join(save_dir, image_name), im)#only save pred_img

            img_cam = cv2.imread('./datasets/kitti/image_cam/' + image_name)#in order to merge
            img_bev = cv2.imread('./datasets/kitti/testing/image_2/' + image_name)#in order to merge

            result_img = cv2.addWeighted(img_bev, 1, im, 1, 0)
            out_img = merge_rgb_to_bev(img_cam, img_bev, result_img, output_width=460)
            cv2.putText(out_img, '{:.1f} ms,{:.2f} FPS'.format((t2 - t1) * 1000, 1 / (t2 - t1)), (6, 30), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)

            if opt.view_img:
                cv2.imshow('detect', out_img)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break

            if opt.save_video:
                if out_cap is None:
                    out_cap_h, out_cap_w = out_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out_cap = cv2.VideoWriter(
                        os.path.join(save_dir, '{}.avi'.format(opt.output_video_fn)),
                        fourcc, 30, (out_cap_w, out_cap_h))
                out_cap.write(out_img)
            else:
                cv2.imwrite(os.path.join(save_dir, image_name), out_img)


            Height, Width, _ = img_channel.shape
            label = np.zeros((Height, Width), dtype=np.uint8)
            label[img_channel[:, :, 1] > 0] = 1
            label[label > 0] = 1
            pred = [label]
            gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST),
                                  axis=0)
            conf_mat += confusion_matrix(gt, pred, dataset.dataset.num_labels)


    print("Inference time with PyTorch = %.3f ms" % (time_torch / seen_torch * 1E3))
    globalacc, pre, recall, F_score, iou = getScores(conf_mat)
    print('Epoch {0:} glob acc : {1:.3f}, pre : {2:.3f}, recall : {3:.3f}, F_score : {4:.3f}, IoU : {5:.3f}'.format(
        opt.epoch, globalacc, pre, recall, F_score, iou))


