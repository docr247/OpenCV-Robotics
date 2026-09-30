import cv2 as cv

img = cv.imread('Cat2.jpg')

cv.imshow('cat', img)

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height  = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

cv.waitKey(0)
