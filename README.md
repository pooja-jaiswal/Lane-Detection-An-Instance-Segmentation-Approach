#Title: Towards End-to-End Lane Detection: An Instance Segmentation Approach

#Introduction:

This project introduces how modern methods have replaced the hand-crafted feature detectors with deep networks to learn dense predictions of pixel-wise lane segmentations. Inspired by the success of dense prediction networks in semantic segmentation and instance segmentation we design a branched, multi-task network for lane instance segmentation, consisting of a lane segmentation branch and a lane embedding branch that can be trained end-to-end. The lane segmentation branch has two output classes, background or lane, while the lane embedding branch further disentangles the segmented lane pixels into different lane instances. By splitting the lane detection problem into the two tasks, we can fully utilize the power of the lane segmentation branch without it having to assign different classes to different lanes. The network, which we will refer to as LaneNet, combines the benefits of binary lane segmentation with a clustering loss function designed for one-shot instance segmentation. In the output of LaneNet, each lane pixel is assigned the id of their corresponding lane.

#Problem Statement:

The ability of the networks to segment out lane markings better, their big receptive field allows them to also estimate lanes even in cases when no markings are present in the image. At a final stage, however, the generated binary lane segmentations still need to be disentangled into the different lane instances. The limitations and propose to cast the lane detection problem as an instance segmentation problem, in which each lane forms its own instance within the lane class. To overcome this problem a multi-task network can jointly handle lane and road marking detection and recognition under adverse weather and low illumination conditions.

#Requirements:

The project will be done using Python 3.4 with Open CV libraries, NumPy 1.13.1, Matplotlib 2.0.2, Scikit Learn 0.19.1 and TensorFlow 1.10.0 for convolutional neural networks. The training of the CNN will be done using google Collaboratory online portal. The image dataset is selected from here [4a] [4b]. These images will be stored in the data folder inside the project. Images folder will have three component containing different images (Original Image, Binary Segmentation Image, and Instance Segmentation Image). The convolutional neural network will be using the image set for model training. The trained model will be further used for lane detection on providing the images.

#Data Used:

The tuSimple lane dataset is the only large-scale dataset for testing deep learning methods on the lane detection task which consists 3626 training and 2782 testing images as mention circumstances:
• Different disturbances, e.g. road paint, vehicles, mud, writings, and lens flare.
• Various illumination conditions, including daytime, dusk, a night with or without lighting.
• Different road types, e.g. 2-lane/3-lane/4-lane or more highway roads.
Paper link: Towards End-to-End Lane Detection: An Instance Segmentation Approach
https://ieeexplore-ieee-org.ezproxy.gl.iit.edu/stamp/stamp.jsp?tp=&arnumber=8500547
 
#Proposed Solution:

In this paper, we have presented a method for end-to-end lane detection at 50 fps. Inspired by recent instance segmentation techniques, our method can detect a variable number of lanes and can cope with lane change maneuvers, in contrast to other related deep learning approaches. To parameterize the segmented lanes using low order polynomials, we have trained a network to generate the parameters of a perspective transformation, conditioned on the image, in which lane fitting is optimal. This network is trained using a custom loss function for lane fitting. Unlike the popular” bird’s-eye view” approach, our method is robust against ground-plane’s slope changes, by adapting the parameters for the transformation accordingly. By doing so, we ensure a lane fitting which is robust against road plane changes, unlike existing approaches that rely on a fixed, predefined transformation. In summary, we propose a fast lane detection algorithm, running at 50 fps, which can handle a variable number of lanes and cope with lane changes. We verify our method on the tuSimple dataset and achieve competitive results.
