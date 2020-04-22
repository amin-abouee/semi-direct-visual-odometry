import cv2 
import numpy as np 
  
# Open the image files. 
img1_color = cv2.imread("original.jpg", cv2.IMREAD_GRAYSCALE)
img2_color = cv2.imread("change.jpg", cv2.IMREAD_GRAYSCALE)

img_ssd = np.square(img2_color - img1_color)
cv2.normalize(img_ssd, img_ssd, 0, 255, cv2.NORM_MINMAX);
cv2.imshow('ssd',img_ssd)


mean_1 = np.mean(img1_color)
mean_2 = np.mean(img2_color)
print("mean_1", mean_1)
print("mean_2", mean_2)
tem1 = (img1_color - mean_1)
print("term1: ", tem1[100:110, 100:110])
tem2 = (img2_color - mean_2)
print("term2: ", tem2[100:110, 100:110])

img_zssd = np.square(tem2 - tem1)
cv2.normalize(img_zssd, img_zssd, 0, 1, cv2.NORM_MINMAX);
print("img_zssd: ", img_zssd[100:110, 100:110])
cv2.imshow('zssd',img_zssd)

tem1 /= np.std(img1_color)
tem2 /= np.std(img2_color)
img_znssd = np.square(tem2 - tem1)
cv2.normalize(img_znssd, img_znssd, 0, 1, cv2.NORM_MINMAX);
cv2.imshow('znssd',img_znssd)



# cv2.normalize(tem1, tem1, 0, 255, cv2.NORM_MINMAX);
# cv2.imshow('mean1',tem1)
# cv2.normalize(tem2, tem2, 0, 255, cv2.NORM_MINMAX);
# cv2.imshow('mean2',tem2)


# img_zssd = np.abs(tem2 - tem1) **2
# print(img_zssd[50:70, 50:70])


# cv2.imshow('image1',img1_color)
# cv2.imshow('image2',img2_color)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     break


gx = cv2.Sobel(img1_color, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img1_color, cv2.CV_32F, 0, 1)
mag1, ang = cv2.cartToPolar(gx, gy)
# cv2.imshow('mag1',mag1)
# print("mag1")
# print(mag1[100:110, 100:110])

gx = cv2.Sobel(img2_color, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img2_color, cv2.CV_32F, 0, 1)
mag2, ang = cv2.cartToPolar(gx, gy)
# cv2.imshow('mag2',mag2)

img_gradient = np.square(mag2 - mag1)
cv2.normalize(img_gradient, img_gradient, 0, 255, cv2.NORM_MINMAX);
cv2.imshow('gradient',img_gradient)
# print("diff")
# print(img_gradient[100:110, 100:110])




cv2.waitKey(0)
cv2.destroyAllWindows()