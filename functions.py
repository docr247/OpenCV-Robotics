import cv2 as cv

img = cv.imread("Cat2.jpg")

cv.imshow('Cat', img)

#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)
#blur
#blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
#cv.imshow("Blur", blur)
#Edge Cascade
canny = cv.Canny(img,125,175)
cv.imshow("Canny", canny)

#Dilating the image
dilated = cv.dilate(canny,(3,3), iterations=1)
cv.imshow("Dilated", dilated)
cv.waitKey(0)