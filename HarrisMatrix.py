import cv2
import numpy as np
import math







def gradientCalculation(image,window_size,k,threshold):    

    dx, dy = np.gradient(image)

    Ixx = dx*dx
    Ixy = dy*dx
    Iyy = dy*dy


##GAUSSIAN BLUR
    Ixx= cv2.GaussianBlur(Ixx,(5,5),1)     
    Iyy = cv2.GaussianBlur(Iyy,(5,5),1)
    Ixy = cv2.GaussianBlur(Ixy,(5,5),1)

    imageHeight,imageWidth=image.shape
    neighbours=np.uint8((window_size-1)/2)

    keypoint=[]
    reg=((Ixx*Iyy)-(Ixy*Ixy))
    trace=Ixx+Iyy
    r2=(reg-(k*trace))
    a,b,c,d=cv2.minMaxLoc(r2)
    print(b)
    r3=r2.copy()
    keypoint=[]
    for y in range(neighbours, imageHeight - neighbours):
            for x in range(neighbours, imageWidth -neighbours):
                    r = r3[y-neighbours:y+neighbours+1, x-neighbours:x+1+neighbours]
                    a,b,c,d=cv2.minMaxLoc(r)
                    abc=np.zeros((3,3),np.float32)
                    abc[d]=b                    
                    r3[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]=abc
      
                    

    for y in range(0, imageHeight-1):
            for x in range(0, imageWidth-1):
                b=r3[y][x]
                if b>threshold:
                        keypoint.append(cv2.KeyPoint(x,y,1,-1,0,0,-1))                    
    img2 =cv2.drawKeypoints(image,keypoint,image,color=(0,255,0), flags=0)

    np.savetxt('test.csv', r2, delimiter=',', fmt='%s')
    np.savetxt('test2.csv', r3, delimiter=',', fmt='%s')
    return r2,r3,img2,image,keypoint 



def main():
    image=cv2.imread("Resources/img.jpg",0)
    image2=cv2.imread("Resources/img2.jpg",0)

    window_size=3
    k=0.04
    threshold=50000000
    r2,r3,img2,image,keypoints=gradientCalculation(image,window_size,k,threshold)
    cv2.imshow("Marks",img2)
    cv2.imwrite("nd.png",img2)
    r2x,r3x,img2x,imagex,keypointsx=gradientCalculation(image2,window_size,k,threshold)
    cv2.imshow("Marks2",img2x)
    cv2.imwrite("nd2.png",img2x)
    abc=[]
    img3 = cv2.drawMatches(img2,keypoints,img2x,keypointsx,(keypoints,keypointsx),None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Marks3",img3)


    


main()




















##    sobelx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
##    sobely=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
##
##
##    DX=cv2.filter2D(image,-1,sobelx)
##    DY=cv2.filter2D(image,-1,sobely)
##
##
##    Ixy=DX*DY
##    Ixx=DX*DX
##    Iyy=DY*DY
##
##    dx = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=3)
##    dy = cv2.Sobel(image,cv2.CV_8U,0,1,ksize=3)






##for y in range(neighbours, imageHeight-neighbours):
##        for x in range(neighbours, imageWidth-neighbours):
##            windowIxx = Ixx[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
##            windowIxy = Ixy[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
##            windowIyy = Iyy[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
##            Sxx = np.float32(windowIxx.sum())
##            Sxy = np.float32(windowIxy.sum())
##            Syy = np.float32(windowIyy.sum())
##            reg=((Sxx*Syy)-(Sxy*Sxy))
##            trace=Sxx+Syy
##            r=(reg-(k*trace))
##            if r> threshold:
##                    keypoint.append(cv2.KeyPoint(x,y,1,-1,0,0,-1))

