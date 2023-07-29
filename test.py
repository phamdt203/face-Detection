# from dataCrop import crop_faces_from_image
# import os
import cv2
# from mtcnn import MTCNN
import time
import glob


my_list = glob.glob("lfw\lfw\*\*.jpg")
time_start = time.time()
for it in my_list:
    image = cv2.imread(it)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    print(11111)
time_end = time.time()
print(( time_end - time_start) / 5)
