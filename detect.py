import os
from options.detect_options import  DetectOptions
from data import CreateDataLoader
from models import create_model
from util.util import confusion_matrix, getScores
import torch
import numpy as np
import cv2
from util.util import time_synchronized,tensor2labelim,merge_rgb_to_bev

#python detect.py --dataroot datasets/kitti --dataset kitti --name kitti  --no_label --epoch kitti --output_video_fn roadseg_video


if __name__ == '__main__':
    opt = DetectOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.isTrain = False

    save_dir = os.path.join(opt.results_dir, opt.name, opt.phase + '_' + opt.epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    model.eval()

    test_loss_iter = []
    epoch_iter = 0
    conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=np.float)
    out_cap=None

    with torch.no_grad():
        for i, data in enumerate(dataset):
            t1 = time_synchronized()
            model.set_input(data)
            model.forward()
            model.get_loss()
            epoch_iter += opt.batch_size
            gt = model.label.cpu().int().numpy()

            _, pred = torch.max(model.output.data.cpu(), 1)
            pred = pred.float().detach().int().numpy()
            t2 = time_synchronized()

            """save images to disk"""
            image_name = model.get_image_names()[0]
            oriSize = (model.get_image_oriSize()[0].item(), model.get_image_oriSize()[1].item())
            palet_file = 'datasets/palette.txt'
            impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3 * 256))

            for label, im_data in model.get_current_visuals().items():
                if label == 'output':
                    im = tensor2labelim(im_data, impalette)
                    im = im[:, :, (1, 0, 2)]  # green channel to blue channel
                    im = cv2.resize(im, oriSize)

                    # cv2.imwrite(os.path.join(save_dir, image_name), im)#only save pred_img

                    img_cam = cv2.imread('./datasets/kitti/image_cam/' + image_name)#in order to merge
                    img_bev = cv2.imread('./datasets/kitti/testing/image_2/' + image_name)#in order to merge
                    
                    result_img = cv2.addWeighted(img_bev, 1, im, 1, 0)
                    out_img = merge_rgb_to_bev(img_cam, img_bev, result_img, output_width=460)
                    cv2.putText(out_img, '{:.1f} ms,{:.2f} FPS'.format((t2 - t1) * 1000, 1 / (t2 - t1)),(6, 30), 0,
                                fontScale=1, color=(0, 255, 0), thickness=2)

                    if opt.view_img:
                        cv2.imshow('detect', out_img)
                        if cv2.waitKey(1) == ord('q'):
                            cv2.destroyAllWindows()
                            raise StopIteration

                    if opt.save_video :
                        if out_cap is None:
                            out_cap_h, out_cap_w = out_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            out_cap = cv2.VideoWriter(
                                os.path.join(save_dir, '{}.avi'.format(opt.output_video_fn)),
                                fourcc, 30, (out_cap_w, out_cap_h))
                        out_cap.write(out_img)
                    else:
                        cv2.imwrite(os.path.join(save_dir, image_name), out_img)

            # Resize images to the original size for evaluation
            image_size = model.get_image_oriSize()
            oriSize = (image_size[0].item(), image_size[1].item())
            gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
            conf_mat += confusion_matrix(gt, pred, dataset.dataset.num_labels)

            test_loss_iter.append(model.loss_segmentation)
            print('Epoch {0:}, iters: {1:}/{2:}, loss: {3:.3f} '.format(opt.epoch,
                                                                        epoch_iter,
                                                                        len(dataset) * opt.batch_size,
                                                                        test_loss_iter[-1]), end='\r')

        avg_test_loss = torch.mean(torch.stack(test_loss_iter))
        print ('Epoch {0:} test loss: {1:.3f} '.format(opt.epoch, avg_test_loss))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)
        print ('Epoch {0:} glob acc : {1:.3f}, pre : {2:.3f}, recall : {3:.3f}, F_score : {4:.3f}, IoU : {5:.3f}'.format(opt.epoch, globalacc, pre, recall, F_score, iou))
