import cv2
import numpy as np
import math

def gradientCalculation(image):    
## Gradient Calculation Using Sobel

    
    dx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=5)
    dy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=5)
    Ixx = dx*dx
    Ixy = dx*dy
    Iyy = dy*dy

##GAUSSIAN BLUR 
    Ixx= cv2.GaussianBlur(Ixx,(5,5),0)    
    Iyy = cv2.GaussianBlur(Iyy,(5,5),0)
    Ixy = cv2.GaussianBlur(Ixy,(5,5),0)
    return dx,dy,Ixx,Iyy,Ixy


def discriptors(gx,gy,keypoint):
        mag=np.zeros(gx.shape)
        angle=np.zeros(gx.shape)
        angle=np.degrees(np.arctan2(gy,gx))%360
        mag=((gx*gx)+(gy*gy))**.5
        mainwindow=16
        smallwindow=4
        binarray=[]

        for ikp in keypoint:
         try:
            x,y=ikp.pt
            x=int(x)
            y=int(y)
 # 16 * 16 window of angle
            win =angle[y-8:y+8, x-8:x+8]
 # 16 * 16 window of magnitude
            win2 =mag[y-8:y+8, x-8:x+8]
            if len(win)<16:
                continue
            x1=-2
            y1=-2
            bindup=[]
 #Getting 16 4*4 windows
            for x in range(0,4):
                x1=-2
                y1=y1+4
                for y in range(0,4):
                    x1=x1+4
                    rsmall=win[y1-2:y1+2, x1-2:x1+2] #4*4 window of magnitude and angle
                    rsmall2=win2[y1-2:y1+2, x1-2:x1+2]
                    bi=dict()
                    for k in range(0,8):
                        bi[k]=0
                    for i in range(0,4):
                        for j in range(0,4):
 # getting the magnitude for each pixel and according to the angle distribute portions of magnitude between multiple bins                           
                            no=np.uint8(rsmall[i][j]/45)
                            bi[no]=(rsmall2[i][j]/45)*((no*45+45)-rsmall[i][j])
                            if no+1 in bi:
                                bi[no+1]=(rsmall2[i][j]/45)*(rsmall[i][j]-no*45)
                            else:
                                bi[0]=(rsmall2[i][j]/45)*(rsmall[i][j]-no*45)
 #128 dimensional 16*8 bin
                                
                    bindup.extend(list(bi.values()))


 #Normalising the Histogram
                    
            barr=np.array(bindup,np.float32)
            div=((barr**2).sum())**.5
 #clipping the value to .2
            binarray.append(np.clip((barr/div),0,.2))
         except:
             continue
        return np.array(binarray,np.float32)

## Function For Matching Descriptors : - SSD
    
def match(binarray,binarray2):
    templis=[0.0,0,0]
    matchess=[]
    matche=[]
    for i in range(len(binarray)):
        temp=100
#SSD 
        for j in range(len(binarray2)):
            k=binarray[i]-binarray2[j]
            k=(k*k).sum()
            if k<temp:
                    temp=k
                    templis=[k,i,j]                
        t=templis.copy()
        matche.append(t)

    sort = sorted(matche, key=lambda tup: tup[2])
    last_used=sort[0][2]
    xyz=[]
    mainmatche=[]

#Custom function to remove bad matches
    for num in sort:
        if num[2]==last_used:
            xyz.append(num)
        elif num[2]>last_used:
            ans=sorted(xyz, key=lambda tup: tup[0])
            mainmatche.append(ans[0])
            xyz=[]
            last_used=num[2]
            xyz.append(num)

    for t in mainmatche:
        matchess.append(cv2.DMatch(t[1],t[2],t[0]))
    return matchess



## Function For Corner Detection


def keypointcal(Ixx,Iyy,Ixy,window_size,k,threshold):
    imageHeight,imageWidth=Ixx.shape
    neighbours=np.uint8((window_size-1)/2)
    keypoint=[]
    r4=np.zeros(Ixx.shape,np.float32)
    extra=[]
    for y in range(neighbours, imageHeight-neighbours):
        for x in range(neighbours, imageWidth-neighbours):
            SIxx = Ixx[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
            SIxy = Ixy[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
            SIyy = Iyy[y-neighbours:y+neighbours+1, x-neighbours:x+neighbours+1]
            Sxx = SIxx.sum()
            Sxy = SIxy.sum()
            Syy = SIyy.sum()
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy            
            r = det - k*(trace**2)
#            r=det/trace
            r4[y][x]=r
    a,b,c,d=cv2.minMaxLoc(r4)
    threshold=b*.4
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
    for y in range(0, imageHeight):
            for x in range(0, imageWidth):
                b=r4[y][x]
                if b>threshold:
                        keypoint.append(cv2.KeyPoint(x,y,1,-1,0,0,-1))                    
    return keypoint

def main():
    nimg1="img1.ppm"
    nimg2="img2.ppm"
    image=cv2.imread("Resources/graf/"+nimg1,0)
    image2=cv2.imread("Resources/graf/"+nimg2,0)
    imageorig=cv2.imread("Resources/graf/"+    nimg1)
    image2orig=cv2.imread("Resources/graf/"  +  nimg2)
    window_size=5
    k=0.04
    threshold=0
    dx,dy,Ixx,Iyy,Ixy=gradientCalculation(image)
    print("Image 1 Step 0")


##    keypoint=[]
##    keypoint.append(cv2.KeyPoint(100,100,1,-1,0,0,-1))                    
##    keypoint.append(cv2.KeyPoint(50,50,1,-1,0,0,-1))                    
##    keypoint.append(cv2.KeyPoint(60,90,1,-1,0,0,-1))                    
##    keypoint.append(cv2.KeyPoint(300,300,1,-1,0,0,-1))                    
    bf = cv2.BFMatcher(crossCheck=True)
    keypoint=keypointcal(Ixx,Iyy,Ixy,window_size,k,threshold)
    print("Image 1 Step 1")
    img2 =cv2.drawKeypoints(imageorig,keypoint,image,color=(0,255,0), flags=0)
    print("Image 1 Step 2")
    binarray=discriptors(dx,dy,keypoint)
    print("Image 1 Step 3")
    dx1,dy1,Ixx1,Iyy1,Ixy1=gradientCalculation(image2)
    print("Image 2 Step 0")
    keypoints=keypointcal(Ixx1,Iyy1,Ixy1,window_size,k,threshold)
    print("Image 2 Step 1")
    img3 =cv2.drawKeypoints(image2orig,keypoints,image2,color=(0,255,0), flags=0)
    print("Image 2 Step 2")
    binarray2=discriptors(dx1,dy1,keypoints)
    print("Image 2 Step 3")
    matche=match(binarray,binarray2)
    print("Image 3 Step 0")
    imgx2 = cv2.drawMatches(imageorig,keypoint,image2orig,keypoints,matche, image2,flags=2)
    print("Image 3 Step 1")
    cv2.imshow("Image1",img2)
    cv2.imshow("Image2",img3)
    cv2.imshow("Image4",imgx2)
    cv2.imwrite("Image1.jpg",img2)
    cv2.imwrite("Image2.jpg",img3)
    cv2.imwrite("Image4.jpg",imgx2)
main()
