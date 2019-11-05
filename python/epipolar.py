# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

import numpy as np
import cv2 
from matplotlib import pyplot as plt
img1 = cv2.imread("../input/0000000000.png",0)  #queryimage # left image
img2 = cv2.imread("../input/0000000001.png",0) #trainimage # right image

nfeatures = 2000 # The number of best features to retain, ranked by their score
nOctaveLayers = 3  # Layers per octave. The number of octaves is computed automatically from the image resolution
contrastThreshold = 0.04 # Filters out weak features in low-contrast regions. Larger value -> less keypoints
edgeThreshold = 10 # Threshold to filter out edge-like features. Larger value -> more (possibly edge-like) keypoints
sigma = 1.6 
sift = sift = cv2.AKAZE_create() 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
print(F)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]