"""
Implementing data parsing class
"""
import os.path as ops

import cv2
import numpy as np

try:
    from cv2 import cv2
except ImportError:
    pass


class DataSet(object):

    def __init__(self, dataset_info_file):

        # Parse the complete JSON file and store all the paths to the appropriate list
        self._gt_img_list, self._gt_label_binary_list, \
        self._gt_label_instance_list = self._init_dataset(dataset_info_file)
        # Randomizing the above list variables
        self._random_dataset()
        self._next_batch_loop_count = 0

    # JSON Parser , parsing and assigning the path to the saperate arry of the path type
    def _init_dataset(self, dataset_info_file):
        gt_img_list = []
        gt_label_binary_list = []
        gt_label_instance_list = []

        # Checks if file doesnot exists.
        assert ops.exists(dataset_info_file), '{:s} does not exist'.format(dataset_info_file)

        # looping over the text file and saperating the blocks and storing it into image, binary and instanace array
        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                gt_img_list.append(info_tmp[0])
                gt_label_binary_list.append(info_tmp[1])
                gt_label_instance_list.append(info_tmp[2])

        return gt_img_list, gt_label_binary_list, gt_label_instance_list

    # Shuffling of array records of image, binary and instance file is done is done
    def _random_dataset(self):
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        # Randomly permute a sequence, or return a permuted range. The suffeled array index
        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_binary_list = []
        new_gt_label_instance_list = []

        # Looping over the shuffled array index and creating a new array array 
        # with the new values depending on the shuffled location
        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_binary_list.append(self._gt_label_binary_list[index])
            new_gt_label_instance_list.append(self._gt_label_instance_list[index])

        # Returning the new array with shuffled values 
        self._gt_img_list = new_gt_img_list
        self._gt_label_binary_list = new_gt_label_binary_list
        self._gt_label_instance_list = new_gt_label_instance_list

    # Data set is prepared in the batches and returned
    def next_batch(self, batch_size):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        # Start and end point of the batch is defined 
        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_binary_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_binary_list = self._gt_label_binary_list[idx_start:idx_end]
            gt_label_instance_list = self._gt_label_instance_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels_binary = []
            gt_labels_instance = []

            # Orignal Image is read from the mentioned location and placed into the list
            for gt_img_path in gt_img_list:
                gt_imgs.append(cv2.imread(gt_img_path, cv2.IMREAD_COLOR))

            # Binary Image is read from the mentioned location and placed into the list
            for gt_label_path in gt_label_binary_list:
                label_img = cv2.imread(gt_label_path, cv2.IMREAD_COLOR)
                label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
                idx = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
                label_binary[idx] = 1
                gt_labels_binary.append(label_binary)

            # Instance Image is read from the mentioned location and placed into the list
            for gt_label_path in gt_label_instance_list:
                #label_img = cv2.imread(gt_label_path, cv2.IMREAD_UNCHANGED)
                label_img = cv2.imread(gt_label_path,cv2.IMREAD_GRAYSCALE)
                gt_labels_instance.append(label_img)

            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels_binary, gt_labels_instance

