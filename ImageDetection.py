import numpy as np
import os
import cv2

print(os.getcwd())
print(os.path.exists(r'C:\Users\Sheetal\AppData\Local\Programs\Python\Python311\aadhar.pdf'))
SampleImg=cv2.imread(r'SampleAadhar\Ujwala AADHAR.jpeg',cv2.COLOR_BGR2GRAY)
img = cv2.imread(r'AadharConcrete\NationalEmblem.png',cv2.COLOR_BGR2GRAY)


# get dimensions of image
dimensions = img.shape
#w,h = img.shape[:-1]
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
#channels = img.shape[2]

print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
#print('Number of Channels : ',channels) #eval(methods[5])
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
res = cv2.matchTemplate(SampleImg,img,cv2.TM_CCOEFF_NORMED)
print(res)
loc=np.where(res >= 0.8)
print(loc)
for pt in zip(*loc[::-1]):  # Switch columns and rows
    cv2.rectangle(SampleImg, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)

cv2.imwrite('result.png', SampleImg)
