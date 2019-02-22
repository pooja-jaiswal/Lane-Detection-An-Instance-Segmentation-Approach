"""
Test the LaneNet model
"""
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

import merge_model
import cluster_module
import postprocess
import global_config

CFG = global_config.cfg
# Weight Initializations for VGG
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """
    Adding CLI argument to Argument parser
    :return: arguments 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=32)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def minmax_scale(input_arr):
    """
    Normalizing the Numpy Array
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, use_gpu):
    """
    The method is used when the prediction is done for single image and not in batch 
    :param image_path: That needs to be predicted
    :param weights_path: The training weight on which prediction needs to be done
    :param use_gpu:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('Start reading image data and pre-processing')
    t_start = time.time()
    # Reading Image in COLOR 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    # Resizing the image to match tensor requirement
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    # Subtracting the dataset mean serves to "center" the data
    image = image - VGG_MEAN
    log.info('Image is read, time consuming: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    # Craeting an object of Lanenet Merge Model
    net = merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    # Getting the binary and instance model that will be used for restoring and prediction
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    # Creating the cluster object
    cluster = cluster_module.Cluster_Module()
    # Creating the post process object
    postprocessor = postprocess.PoseProcessor()

    # The Saver class adds ops to save and restore variables to and from checkpoints
    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'CPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        # Performing the prediction using model on the image
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret],
                                                        feed_dict={input_tensor: [image]})
        t_cost = time.time() - t_start
        log.info('Single image lane line prediction time consuming: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        # Performing the lane masking image on the predicted lane points
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        for i in range(4):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        mask_image = cv2.resize(mask_image, (image_vis.shape[1],image_vis.shape[0]),interpolation=cv2.INTER_LINEAR)
        mask_image = cv2.addWeighted(image_vis, 1.0, mask_image, 1.0, 0)
        plt.figure('final_image')
        plt.imshow(mask_image)
        plt.show()

    sess.close()

    return


def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """
    This method is called when the prtediction is donw in the batches. 
    The prediction is done and the image is written in the "ret" directory
    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('Start getting the image file path...')
    # Fetching the list of images path using glob
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = cluster_module.Cluster_Module()
    postprocessor = postprocess.PoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        # Epochs is generated by dividing total image by batch size which is passed as the argument
        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        # Looping for iterations and processing batch wise
        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] Start image reading and preprocessing...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] Pretreatment{:d}Image, Total time consuming: {:.5f}s, Average time per sheet: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            # Performing Prediction
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] prediction{:d}Image lane line, Total time consuming: {:.5f}s, Average time per sheet: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            # looping on the predicted image
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                # Drawing the lane on the original image
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                
                cluster_time.append(time.time() - t_start)
                # Resizing the image
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                # When No save directory us available, it will display the processed image
                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                # If the directory is available, the image will be saved to the location with the predicted lane drawn over image
                if save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name)
                    cv2.imwrite(image_save_path, mask_image)

            log.info('[Epoch:{:d}] get on{:d}Image lane line clustering, Total time consuming: {:.5f}s, Average time per sheet: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(args.image_path, args.weights_path, args.use_gpu)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size)
