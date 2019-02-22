"""
LaneNet model post processing
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


class PoseProcessor(object):
    def __init__(self):
        pass

    @staticmethod
    def _morphological_process(image, kernel_size=5):
        """

        :param image:
        :param kernel_size:
        :return:
        """
        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)
        if len(image.shape) == 3:
            # Converting Image to Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #  On Passing the shape and size of the kernel, it returns the desired kernel
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

        # close operation fille hole
        # It is useful in removing noise
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closing

    @staticmethod
    def _connect_components_analysis(image):
        """

        :param image:
        :return:
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)

    def postprocess(self, image, minarea_threshold=15):
        """

        :param image:
        :param minarea_threshold: Connected domain analysis threshold
        :return:
        """
        # First, image morphology operation
        morphological_ret = self._morphological_process(image, kernel_size=5)

        # Connected domain analysis
        connect_components_analysis_ret = self._connect_components_analysis(image=morphological_ret)

        # Sort connected domains and delete too small connected domains
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]

        for index, stat in enumerate(stats):
            if stat[4] <= minarea_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        return morphological_ret
