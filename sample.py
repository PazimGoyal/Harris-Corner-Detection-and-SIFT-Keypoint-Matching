
import cv2
import numpy as np
import sys
import getopt
import operator

def readImage(filename):

    img = cv2.imread(filename, 0)
    return img

def findCorners(img, window_size, k, thresh):
##
##    sobelx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
##    sobely=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
##
##    DX=cv2.filter2D(img,-1,sobelx)
##    DY=cv2.filter2D(img,-1,sobely)
##
##    Ixy=DX*DY
##    Ixx=DX*DX
##    Iyy=DY*DY
##
##    Ixx= cv2.GaussianBlur(Ixx,(5,5),0)
##    Iyy = cv2.GaussianBlur(Iyy,(5,5),0)
##    Ixy = cv2.GaussianBlur(Ixy,(5,5),0)
##



    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = (window_size-1)/2
    offset=np.uint8(offset)

    #Loop through image and find our corners
    print ("Finding Corners...")
    lis=[]
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
                lis.append(r)
                cornerList.append([x, y])
                color_img.itemset((y, x, 0), 0)
                color_img.itemset((y, x, 1), 255)
                color_img.itemset((y, x, 2), 255)
                print(x,y,r)


    return color_img, cornerList

def main():
        img = cv2.imread("checkerboard.png",0)
##        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


        print( "Shape: " + str(img.shape))
        print( "Size: " + str(img.size))
        print( "Type: " + str(img.dtype))
        print( "Printing Original Image...")
        cv2.imshow("r",img)

##        finalImg, cornerList = findCorners(img, int(5), float(0.04), int(10000))
##        if finalImg is not None:
##            cv2.imwrite("finalimage.png", finalImg)
##            cv2.imshow("dd",finalImg)
        img2 =cv2.drawKeypoints(img,cv2.KeyPoint(x=100,y=100,1),img,color=(0,255,0), flags=0)

##
##        # Write top 100 corners to file
##        cornerList.sort(key=operator.itemgetter(2))
##        outfile = open('corners.txt', 'w')
##        for i in range():
##            outfile.write(str(cornerList[i][0]) + ' ' + str(cornerList[i][1]) + ' ' + str(cornerList[i][2]) + '\n')
##        outfile.close()


if __name__ == "__main__":
    main()
