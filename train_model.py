
"""
Training the lanenet model
"""
import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf

import global_config
import merge_model
import data_processor

CFG = global_config.cfg
# Weight Initializations for VGG
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    parser = argparse.ArgumentParser()
    # Adding CLI argument to parser
    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Base net work to use', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')
    return parser.parse_args()

# Min-max normalisation of a NumPy array
def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr


def train_net(dataset_dir, weights_path=None, net_flag='vgg'):
    """
    Training Module for performing training of the CNN
    :param dataset_dir:
    :param net_flag: choose which base network to use (Used is vgg)
    :param weights_path:
    :return:
    """
    # Training and Validation file Path
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file)

    # Fetchng the train and validation dataset contaning the list of file paths from the text file 
    # in random faishon.  This list will be further used for pulling the data in batches
    train_dataset = data_processor.DataSet(train_dataset_file)
    val_dataset = data_processor.DataSet(val_dataset_file)

    with tf.device('/gpu:1'):
        #Inserts a placeholder for a tensor that will be always fed.
        # A Tensor that may be used as a handle for feeding a value, but not evaluated directly.
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                             CFG.TRAIN.IMG_WIDTH, 3],
                                      name='input_tensor')
        binary_label_tensor = tf.placeholder(dtype=tf.int64,
                                             shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                    CFG.TRAIN.IMG_WIDTH, 1],
                                             name='binary_input_label')
        instance_label_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                      CFG.TRAIN.IMG_WIDTH],
                                               name='instance_input_label')
        phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

        # Craeting an object of Merge Model
        net = merge_model.LaneNet(net_flag=net_flag, phase=phase)

        # calculate the loss
        # Loss is computed for Binary, instance, discriminative and total
        # CNN Model is used for encode and decoder module for calculating loss
        compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label_tensor,
                                       instance_label=instance_label_tensor, name='lanenet_model')
        
        # Saperating all the loss and value received from compute loss step.
        total_loss = compute_ret['total_loss']
        binary_seg_loss = compute_ret['binary_seg_loss']
        disc_loss = compute_ret['discriminative_loss']
        pix_embedding = compute_ret['instance_seg_logits']
        out_logits = compute_ret['binary_seg_logits']

        ######### calculate the accuracy #########
        # The shape of output of a softmax is the same as the input - 
        # It just normalizes the values. The outputs of softmax can be interpreted as probabilities.
        out_logits = tf.nn.softmax(logits=out_logits)
        # Returns the index with the largest value across axes of a tensor
        out_logits_out = tf.argmax(out_logits, axis=-1)
        out = tf.argmax(out_logits, axis=-1)
        # Inserts a dimension of 1 at the dimension index axis of input's shape
        out = tf.expand_dims(out, axis=-1)

        # Returns the truth value of (binary_label_tensor == 1) element-wise. and 
        # returns the coordinates of true elements of condition
        idx = tf.where(tf.equal(binary_label_tensor, 1))
        # Gather slices from out into a Tensor with shape specified by indices i.e idx.
        pix_cls_ret = tf.gather_nd(out, idx)
        # Computes number of nonzero elements across dimensions of pix_cls_ret tensor
        accuracy = tf.count_nonzero(pix_cls_ret)
        accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

        # When training a model, it is often recommended to lower the learning rate as the training progresses. 
        # This function applies an exponential decay function to a provided initial learning rate. 
        # It requires a global_step value to compute the decayed learning rate. 
        # The global_step is increment at each epochs and the decay rate is 0.1
        global_step = tf.Variable(0, trainable=False)
        # Initially we can approach to take bigger steps but as learning approaches convergance,  
        # than having a slower learning rate allowes us to take smaller steps and improve learning
        learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                                   CFG.TRAIN.LR_DECAY_STEPS, CFG.TRAIN.LR_DECAY_RATE, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Momendul Optimizer is used
            # As we needs to minimize the calculated error which is done using back propagation.
            # The current error is typically propagated backwards to a previous layer,
            # where it is used to modify the weights and bias and the error is minimized. 
            # The weights are modified using a function called Optimization Function.
            #optimizer = tf.train.AdamOptimizer(
            #    learning_rate=learning_rate).minimize(loss=total_loss,
            #                                                        var_list=tf.trainable_variables(),
            #                                                        global_step=global_step)
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(loss=total_loss,
                                                                    var_list=tf.trainable_variables(),
                                                                    global_step=global_step)

    # Set tf saver
    # The model and weights are being saved in the model directory at root location
    saver = tf.train.Saver()
    model_save_dir = 'model/'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # Model is appended with the current time stamp
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set tf summary
    # Tensor Board is configured for viewing the training progress
    tboard_save_path = './tboard/{:s}'.format(net_flag)
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)
    train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss', tensor=disc_loss)
    val_instance_seg_loss_scalar = tf.summary.scalar(name='val_instance_seg_loss', tensor=disc_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    # Merges summaries
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar, train_binary_seg_loss_scalar,
                                               train_instance_seg_loss_scalar])
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar,
                                             val_binary_seg_loss_scalar, val_instance_seg_loss_scalar])

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        # If the Weight file (saved weight) is passed, training will start using weights
        # else training will start from scratch
        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # Load pre-training parameters
        # Here the pretrained VGG16 model is used as pretrained parameters
        if net_flag == 'vgg' and weights_path is None:
            pretrained_weights = np.load(
                'vgg16.npy',
                encoding='latin1').item()

            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

        train_cost_time_mean = []
        val_cost_time_mean = []
        start_time = time.time()
        # Looping for No of Epochs
        for epoch in range(train_epochs):
            ######### TRAINING BLOCK #########
            t_start = time.time()

            with tf.device('/cpu:0'):
                # Data sets is prepared from the text file which is passed as system argument
                # The datasets are gt_imgs - original image, binary_gt_labels - binary images 
                # and instance_gt_labels - instance images
                gt_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)

                # Resizing all original images with train_images width - 512 and height - 256
                gt_imgs = [cv2.resize(tmp,
                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      dst=tmp,
                                      interpolation=cv2.INTER_LINEAR)
                           for tmp in gt_imgs]

                # Subtracting the dataset mean serves to "center" the data and 
                # making the sure the range is similar in order to get stable gradients
                # This actually normalizes feature values i.e. the reason we batch normalize is when
                # we can't fit the full dataset in memory.
                gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]

                # Resizing all the binary images with train_images width - 512 and height - 256
                binary_gt_labels = [cv2.resize(tmp,
                                               dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                               dst=tmp,
                                               interpolation=cv2.INTER_NEAREST)
                                    for tmp in binary_gt_labels]
                binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]

                # Resizing all the instance images with train_images width - 512 and height - 256
                instance_gt_labels = [cv2.resize(tmp,
                                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                 dst=tmp,
                                                 interpolation=cv2.INTER_NEAREST)
                                      for tmp in instance_gt_labels]
            phase_train = 'train'

            # Initiating the training process by passing all the required parameters computed above
            _, c, train_accuracy, train_summary, binary_loss, instance_loss, embedding, binary_seg_img = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          disc_loss,
                          pix_embedding,
                          out_logits_out],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label_tensor: binary_gt_labels,
                                    instance_label_tensor: instance_gt_labels,
                                    phase: phase_train})

            if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
                log.error('cost is: {:.5f}'.format(c))
                log.error('binary cost is: {:.5f}'.format(binary_loss))
                log.error('instance cost is: {:.5f}'.format(instance_loss))
                #cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                #cv2.imwrite('nan_instance_label.png', instance_gt_labels[0])
                #cv2.imwrite('nan_binary_label.png', binary_gt_labels[0] * 255)
                return
            '''
            # For Testing purpose, at every 100th epochs, the images are written in the root directory.
            if epoch % 100 == 0:
                cv2.imwrite('image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('binary_label.png', binary_gt_labels[0] * 255)
                cv2.imwrite('instance_label.png', instance_gt_labels[0])
                cv2.imwrite('binary_seg_img.png', binary_seg_img[0] * 255)

                for i in range(4):
                    embedding[0][:, :, i] = minmax_scale(embedding[0][:, :, i])
                embedding_image = np.array(embedding[0], np.uint8)
                cv2.imwrite('embedding.png', embedding_image)
            '''

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            # Updating the global_steps on which the learning decay depends.
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            ######### VALIDATION BLOCK #########
            with tf.device('/cpu:0'):
                # Preparing all the images i.e. original, binary and instance for validation 
                gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val \
                    = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
                
                # Resizing all original images with train_images width - 512 and height - 256
                gt_imgs_val = [cv2.resize(tmp,
                                          dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                          dst=tmp,
                                          interpolation=cv2.INTER_LINEAR)
                               for tmp in gt_imgs_val]
                gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]

                # Resizing all the binary images with train_images width - 512 and height - 256
                binary_gt_labels_val = [cv2.resize(tmp,
                                                   dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                   dst=tmp)
                                        for tmp in binary_gt_labels_val]
                binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]

                # Resizing all the instance images with train_images width - 512 and height - 256
                instance_gt_labels_val = [cv2.resize(tmp,
                                                     dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                     dst=tmp,
                                                     interpolation=cv2.INTER_NEAREST)
                                          for tmp in instance_gt_labels_val]
            phase_val = 'test'

            # Initiating the validation process by passing all the desired parameters.
            t_start_val = time.time()
            c_val, val_summary, val_accuracy, val_binary_seg_loss, val_instance_seg_loss = \
                sess.run([total_loss, val_merge_summary_op, accuracy, binary_seg_loss, disc_loss],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label_tensor: binary_gt_labels_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    phase: phase_val})

            '''
            # For testing purpose,at every 100th epoch, writting the image to the root directory
            if epoch % 100 == 0:
                cv2.imwrite('test_image.png', gt_imgs_val[0] + VGG_MEAN)
            '''

            # Updating the global steps
            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            # Printing the training parameters like , loss, accuracy, cost, etc to the console
            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} instance_seg_loss= {:6f} accuracy= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, binary_loss, instance_loss, train_accuracy,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(epoch + 1, c_val, val_binary_seg_loss, val_instance_seg_loss, val_accuracy,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            # At every 2000th steps , save the fresh model file.
            if epoch % 200 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
        print("\n---------------FINAL RESULTS-----------------") 
        print("Results after last Epochs")
        print("Train binary loss:     ",binary_loss)
        print("Train instance loss:   ",instance_loss)
        print("Train accuracy:        ",train_accuracy)
        print("Total training loss    ",c)

        print("Test binary loss:      ",val_binary_seg_loss)
        print("Test instance loss:    ",val_instance_seg_loss)
        print("Test accuracy:         ",val_accuracy)
        print("Total test loss        ",c_val)

        print("\n---------------------------------------------")
        print("\nRuntime: " + str(int((time.time() - start_time) / 60)) + " minutes")
        print("\n---------------------------------------------")

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)
