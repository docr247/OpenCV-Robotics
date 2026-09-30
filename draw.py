import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
#cv.imshow("blank", blank)
#paint image in certain colors
#blank[200:300, 300:400] = 0,255,0
#cv.imshow("Green", blank)

#draw a rectangle
#cv.rectangle(blank, (50,50), (250,250), (0,255,0), thickness=2)
#cv.rectangle(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (0,255,0), thickness=cv.FILLED)
#cv.imshow("Rectabgle", blank)

#draw a circle
#cv.circle(blank,(250,250), 40, (0,0,255), thickness=3)
#cv.imshow("Circle", blank)

#draw a line
#cv.line(blank, (20,20), (blank.shape[1]//2, blank.shape[0]//3),(255,255,255),thickness=3)
#cv.imshow("Line", blank)

#Write Text on image
cv.putText(blank, "Hello World", (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0),2 )
cv.imshow('Text', blank)

cv.waitKey(0)
cv.destroyAllWindows()