from scipy.linalg import svd
import numpy as np
np.seterr(over='ignore')
import cv2 as cv
import matplotlib.pyplot as plt
from math import atan2,asin,sqrt

def FundamentalMat(x1, x2):
    x = 0
    y = 1
    A  = np.zeros((x1.shape[0], 9))
    for i in range(x1.shape[0]):
        A[i, 0] = x1[i, x]*x2[i, x]
        A[i, 1] = x1[i, x]*x2[i, y]
        A[i, 2] = x1[i, x]
        A[i, 3] = x1[i, y]*x2[i, x]
        A[i, 4] = x1[i, y]*x2[i, y]
        A[i, 5] = x1[i, y]
        A[i, 6] = x2[i, x]
        A[i, 7] = x2[i, y]
        A[i, 8] = 1

    # print(A[:, -1])
    # print(A.shape)


    [U1, S1, V1] = svd(A)

    H = V1[-1, :]
    # print("H ", H)
    h =np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], H[8]]])
    # print("h", h)

    [U2, S2, V2] = svd(h)

    S2[2] = 0
    # print(V2)
    F = np.dot(U2, np.dot(np.diag(S2), V2))
    F = F/np.linalg.norm(F, 2)

    return F

def EssentialMatfromFundamentalMat(K, F):
    E = np.dot(np.transpose(K), np.dot(F, K))
    # print("E before decomposition ", E)
    [U, S, V] = svd(E)
    S[2] = 0
    
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V))
    # print(E)
    # print(np.linalg.norm(E, 2), np.linalg.matrix_rank(E))
    E = E/np.linalg.norm(E, 2)
    
    return E


def campose(E):
    [U, S, V] = svd(E)
    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    
    R = np.dot(U, np.dot(W, V))
    
    tx = np.dot(U, np.dot(Z, np.transpose(U)))
    # print(tx)
    t = np.array([tx[2, 1], tx[0, 2], tx[1, 0]])
    # C = -np.dot(np.linalg.inv(R), t)
    return R, t


img = cv.imread('database_img.png',cv.IMREAD_GRAYSCALE) 
width, height = img.shape[1],img.shape[0]
#print(width, height)
f=100
y1,x1=int(round(3000)), int(round(2500))
z1=5000
res=50
capture_resolution=[((z1)/res)*1000*(1.0/f),((z1)/res)*1000*(1.0/f)]
img1 = img[int(round(y1-capture_resolution[0]/2)):int(round(y1+capture_resolution[0]/2)),int(round( x1-capture_resolution[1]/2)):int(round(x1+capture_resolution[1]/2))]
img1=cv.resize(img1,(1000,1000),interpolation = cv.INTER_CUBIC)
#print(frame1.shape[1],frame1.shape[0])


z2=6000
y2,x2=  int(round(3200)), int(round(2700))                 #int(round(960)), int(round(440))
capture_resolution=[((z2)/res)*1000*(1.0/f),((z2)/res)*1000*(1.0/f)]
img2 = img[int(round(y2-capture_resolution[0]/2)):int(round(y2+capture_resolution[0]/2)),int(round( x2-capture_resolution[1]/2)):int(round(x2+capture_resolution[1]/2))]
img2=cv.resize(img2,(1000,1000),interpolation = cv.INTER_CUBIC)


fx=f
fy=f
K=[[fx,0,500],[0,fy,500],[0,0,1]]
F=FundamentalMat(img1, img2)
#print(frame1, frame2)
E=EssentialMatfromFundamentalMat(K, F)
[R,t]=campose(E)
roll = atan2(-R[2][1], R[2][2])
pitch = asin(R[2][0])
yaw = atan2(-R[1][0], R[0][0])
print(roll,pitch,yaw,t)




