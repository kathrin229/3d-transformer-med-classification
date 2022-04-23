import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = 'data/dataset_seg/CP/1065/3104/0040.png'

im = cv.imread(path)
img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

ret,binary = cv.threshold(img,127,255,cv.THRESH_TRUNC)
contours,hierarchy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
print("Number of contours:" + str(len(contours)))
x = 512
y = 512
x_down = 0
y_down = 0
for i in range(len(contours)):
    x_new,y_new,w_new,h_new = cv.boundingRect(contours[i])
    x_down_new = x_new + w_new
    y_down_new = y_new + h_new
    if x_new < x:
        x = x_new
    if y_new < y:
        y = y_new
    if x_down_new > x_down:
        x_down = x_down_new
    if y_down_new > y_down:
        y_down = y_down_new

w = x_down -x
h = y_down -y
cv.rectangle(im,(x,y),(x+w,y+h),(255,0,0),3)


plt.imshow(im)
plt.show()
print('done')
