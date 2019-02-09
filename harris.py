import cv2
import numpy as np
import math

def gradientCalculation(image):    
## Gradient
    dx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    Ixx = dx*dx
    Ixy = dx*dy
    Iyy = dy*dy
##GAUSSIAN BLUR
    Ixx= cv2.GaussianBlur(Ixx,(5,5),0)     
    Iyy = cv2.GaussianBlur(Iyy,(5,5),0)
    Ixy = cv2.GaussianBlur(Ixy,(5,5),0)
    return dx,dy,Ixx,Iyy,Ixy


def keypointcal(Ixx,Iyy,Ixy,window_size,k,threshold):
    imageHeight,imageWidth=Ixx.shape
    neighbours=np.uint8((window_size-1)/2)
    keypoint=[]
    r4=np.zeros(Ixx.shape,np.float32)
    extra=[]
    for y in range(neighbours, imageHeight-neighbours):
        for x in range(neighbours, imageWidth-neighbours):
            #Calculate sum of squares
            windowIxx = Ixx[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
            windowIxy = Ixy[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
            windowIyy = Iyy[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            r4[y][x]=r
    a,b,c,d=cv2.minMaxLoc(r4)
    threshold=b*.3
    for y in range(0, imageHeight):
            for x in range(0, imageWidth):
                b=r4[y][x]
                if b>threshold:
                    r4[y][x]=b     
                else:
                    r4[y][x]=0
                    
    
    for y in range(neighbours, imageHeight-neighbours):
            for x in range(neighbours, imageWidth-neighbours):
                    r = r4[y-neighbours:y+neighbours+1, x-neighbours:x+1+neighbours]
                    a,b,c,d=cv2.minMaxLoc(r)
                    abc=np.zeros((window_size,window_size),np.float32)
                    abc[d]=b
                    r4[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]=abc
##                    extra.append(b)

##    extra=np.array(extra)
##    threshold=extra[np.argsort(extra)[-1000:]][0]
    for y in range(0, imageHeight):
            for x in range(0, imageWidth):
                b=r4[y][x]
                if b>threshold:
                        keypoint.append(cv2.KeyPoint(x,y,1,-1,0,0,-1))                    
    return keypoint





def discriptors(gx,gy,keypoint):
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mainwindow=16
    smallwindow=4
    binarray=[]
    for ikp in keypoint:
        x,y=ikp.pt
        x=int(x)
        y=int(y)
        win =angle[y-8:y+8, x-8:x+8]
        win2 =mag[y-8:y+8, x-8:x+8]
        x1=-2
        y1=-2
##        binlist = np.empty(shape=(4,4,8))
        bindup=[]
        for x in range(0,4):
##            bindup = np.copy(binlist)
            x1=-2
            y1=y1+4
            for y in range(0,4):
                x1=x1+4
                rsmall=win[y1-2:y1+2, x1-2:x1+2]
                rsmall2=win2[y1-2:y1+2, x1-2:x1+2]
                bi=dict()
                for k in range(0,8):
                    bi[k]=0

                for i in range(0,4):
                    for j in range(0,4):
                        no=np.uint8(rsmall[i][j]/45)
                        bi[no]=bi.get(no,0)+rsmall2[i][j]
##                bindup[x][y]=list(bi.values())
                bindup.extend(list(bi.values()))
        barr=np.array(bindup,np.float32)
        div=((barr**2).sum())**.5
        binarray.append(np.clip((barr/div),0,.2))
    return np.array(binarray,np.float32)


    
def match(binarray,binarray2):
    for i in binarray:
        for j in binarray2:
            k=i.sum()-j.sum()
            if -0.02<k<0.02:
                print(k)
        print("----------------")

            
def main():
    image=cv2.imread("Resources/graf/img1.ppm",0)
    f = open("Resources/graf/img2.key", "r")
    window_size=5
    k=0.04
    threshold=9.26973600e+16
    dx,dy,Ixx,Iyy,Ixy=gradientCalculation(image)
##    keypoint=[]
##    keypoint.append(cv2.KeyPoint(100,100,1,-1,0,0,-1))                    
    bf = cv2.BFMatcher( crossCheck=True)

    keypoint=keypointcal(Ixx,Iyy,Ixy,window_size,k,threshold)
    img2 =cv2.drawKeypoints(image,keypoint,image,color=(0,255,0), flags=0)
    cv2.imshow("Marks",img2)
    cv2.imwrite("nd.png",img2)
    binarray=discriptors(dx,dy,keypoint)
    image2=cv2.imread("Resources/graf/img2.ppm",0)
    dx1,dy1,Ixx1,Iyy1,Ixy1=gradientCalculation(image2)
    keypoints=keypointcal(Ixx1,Iyy1,Ixy1,window_size,k,threshold)
    img3 =cv2.drawKeypoints(image2,keypoints,image2,color=(0,255,0), flags=0)
    cv2.imshow("Marks2",img3)
    cv2.imwrite("nd2.png",img3)
    binarray2=discriptors(dx1,dy1,keypoints)
    print(type(binarray))

    matches = bf.match(binarray,binarray2)


# Draw first 10 matches.
    imgx = cv2.drawMatches(image,keypoint,image2,keypoints,matches, image,flags=2)
    cv2.imshow("final",imgx)
    cv2.imwrite("nd23.png",imgx)


    
main()
