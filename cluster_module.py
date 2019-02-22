"""
Implement the clustering part of instance segmentation in LaneNet
"""
import numpy as np
import glog as log
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import time
import warnings
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


class Cluster_Module(object):
    """
    Instance segmentation clusterer
    """

    def __init__(self):
        """
        Defining a color map Which will be used for plotting the lanes on prediction
        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        pass

    @staticmethod
    def _cluster(prediction, bandwidth):
        meanShift = MeanShift(bandwidth, bin_seeding=True)
        # log.info('Start Mean shift clustering...')
        tic = time.time()
        try:
            meanShift.fit(prediction)
        except ValueError as err:
            log.error(err)
            return 0, [], []
        # log.info('Mean Shift time consuming: {:.5f}s'.format(time.time() - tic))
        labels = meanShift.labels_
        cluster_centers = meanShift.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        # log.info('The number of clusters is: {:d}'.format(num_clusters))

        return num_clusters, labels, cluster_centers

    @staticmethod
    def _cluster_v2(prediction):
        """
        dbscan cluster
        :param prediction:
        :return:
        """
        db = DBSCAN(eps=0.7, min_samples=200).fit(prediction)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        unique_labels = [tmp for tmp in unique_labels if tmp != -1]
        log.info('The number of clusters is: {:d}'.format(len(unique_labels)))

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        return num_clusters, db_labels, cluster_centers

    @staticmethod
    def _get_lane_area(binary_seg_ret, instance_seg_ret):
        """
        Obtain eigenvectors of all lane lines on the instance segmentation map by binary partitioning mask
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 1)

        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])
            lane_coordinate.append([idx[0][i], idx[1][i]])

        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):
        """
        Filter the instance lane line position coordinate point, 
            assuming that the lane line is continuous, so the coordinate 
            transformation of the lane line point should be smooth and should not occur.
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x)

        return coord[idx[0]]

    @staticmethod
    def _lane_fit(lane_pts):
        """
        Lane line polynomial fitting to draw lines for thr detected lanes
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        x_fit = []
        y_fit = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f1 = np.polyfit(x, y, 3)
                p1 = np.poly1d(f1)
                x_min = int(np.min(x))
                x_max = int(np.max(x))
                x_fit = []
                for i in range(x_min, x_max + 1):
                    x_fit.append(i)
                y_fit = p1(x_fit)
            except Warning as e:
                x_fit = x
                y_fit = y
            finally:
                return zip(x_fit, y_fit)

    # The menthod is used for drawing the lanes lines using polylines and return the images
    def get_lane_mask(self, binary_seg_ret, instance_seg_ret):
        lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, instance_seg_ret)

        num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1.5)

        # If there are more than eight clusters, the eight clusters with the most samples in the class are selected to remain.
        if num_clusters > 8:
            cluster_sample_nums = []
            for i in range(num_clusters):
                cluster_sample_nums.append(len(np.where(labels == i)[0]))
            sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
            cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
        else:
            cluster_index = range(num_clusters)

        mask_image = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)

        for index, i in enumerate(cluster_index):
            idx = np.where(labels == i)
            coord = lane_coordinate[idx]
            coord = np.flip(coord, axis=1)
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))
            coord = np.array([coord])
            cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=2)

        return mask_image

