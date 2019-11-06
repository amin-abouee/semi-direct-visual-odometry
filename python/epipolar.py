# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://stackoverflow.com/questions/25018423/opencv-python-error-when-using-orb-images-feature-matching

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
img1 = cv2.imread("/home/amin/Workspace/cplusplus/semi-direct-visual-odometry/input/0000000000.png",0)  #queryimage # left image
img2 = cv2.imread("/home/amin/Workspace/cplusplus/semi-direct-visual-odometry/input/0000000001.png",0) #trainimage # right image

print(os.getcwd())
nfeatures = 2000 # The number of best features to retain, ranked by their score
nOctaveLayers = 3  # Layers per octave. The number of octaves is computed automatically from the image resolution
contrastThreshold = 0.04 # Filters out weak features in low-contrast regions. Larger value -> less keypoints
edgeThreshold = 10 # Threshold to filter out edge-like features. Larger value -> more (possibly edge-like) keypoints
sigma = 1.6 
akaze = cv2.AKAZE_create() 
# find the keypoints and descriptors with SIFT
kp1, des1 = akaze.detectAndCompute(img1,None)
kp2, des2 = akaze.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_LSH = 6

index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
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

K = np.array([[721.5377, 0.0, 609.5593],
 [ 0.0, 721.5377, 172.8540],
 [0.0, 0.0, 1.0]])
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
print("F: ", F)
E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.FM_LMEDS)
print("E:, ", E)
R1, R2, t = cv2.decomposeEssentialMat(E)
print("R1: ", R1)
print("R2: ", R2)
print("t: ", t)
e1 = -R1.transpose().dot(t)
print("e1: ", e1/e1[2])
e2 = -R2.transpose().dot(t)
print("e2: ", e2/e2[2])
e3 = -R1.transpose().dot(-t)
print("e3: ", e3/e3[2])
e4 = -R2.transpose().dot(-t)
print("e4: ", e4/e4[2])

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]