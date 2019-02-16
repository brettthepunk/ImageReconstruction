import cv2

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy import misc
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

##### Parameters: image_location == directory    size = (x,y) vector containing dimensions of keyboard

def Checkerboard(image_location, size, save = False,units=None, savename=None):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((size[1]*size[0],3), np.float32)
    
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2) * units
    if units:
        objp[:,:2] = objp[:,:2]* units # code to translate calibration into real world units 
    # depending measurements of single side of chessboard, should be in 'mm'
    
# Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


# set image path here !!!! images should be in alone in unique directory 
    for fname in image_location:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (size[1],size[0]),None)

    # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
         # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (size[0],size[1]), corners2,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
# rendering the chessboard images here, commented out at the moment.
    #cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
# this function calculates calibration error by projecting point vectors back onto original image 
    mean_error = 0
    tot_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    error_term=mean_error/len(objpoints)
    
    
    # refine this save function..also might be nice to define some class functions for this project (CHRIS?)
    if save: 
        save_vector = ret
        save_vector.append(mtx)
        save_vector.append(dist)
        save_vector.append(rvecs)
        save_vector.append(tvecs)
        save_vector.append(error_term)
        numpy.savetxt(savename,save_vector, delimiter=",")
    return ret, mtx, dist, rvecs, tvecs, error_term


    ####### mtx,dist are camera matrices and distortion from above function, rvecs and tvecs are rotation and translation vectors ######
    ###### error_term is mean error calculated in final lines of code ######
    
##### Parameters: image_loc == directory information    size is (x,y) array containing # of checkerboards in pattern
##### mtx is camera matrix found above, dist is distortion coeffecient found above    
##### Should compute these variables first using above program, then pass the variables into this function
    
def trans_rotation(image_loc,size,mtx,dist):
 
    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((size[1]*size[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[1],0:size[0]].T.reshape(-1,2)

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


# adjust directory here into directory of local frames from video.
    chdir(image_loc)
    for fname in os.listdir('.'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (size[1],size[0]),None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
    rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    R_T = np.hstack(rvecs,tvecs)
    return R_T, inliers

###### Parameters:    K1, K2 are 
###### R_T is computed above 

def compute_RT(rvecs):
    rotation_mat = np.zeros(shape=(3, 3))
    R_b = cv2.Rodrigues(rvecs_b, rotation_mat)[0]
    return R_b

def compute_projections(K1,K2,R_T):
    p1 = K1 * np.identity(3)
    zero_string = [0,0,0]
    zero_string = np.array(zero_string)
    p1 = np.column_stack((p1,zero_string))
    p2 = np.matmul(K_2,  R_T)
    
    return p1,p2



####### Parameters: p1, p2 are transforms found above

###### pp1, pp2 are coordinate pairs to be transformed

def triangulation(p1,p2,pp1,pp2):
    # maybe create 3xN vector of length points to fill or actually 4xN length...proj
    true_points = cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2)
    return true_points

def translate(p1,p2,handle_1_coords,handle_2_coords):
    tupad = []
    for i in range(0,len(handle_1_coords)):
        tupac.append(triangulation(p1,p2,handle_1_coords[i,:],handle_2_coords[i]))
    
    coordinates = tupac / tupac[3] # Normalize returned coordinates
    coordinates = np.array(coordinates)
    true_gate = []
    for i in range(0,434):
        true_gate.append(coordinates[i,:,0])
    true_gate = np.array(true_gate)
    return true_gate

def load_data(DLC_array):
    data = pd.read_hdf(DLC_array)
    return data


def scatter(data,title,save_id=None):
    df = pd.DataFrame(data, columns=list('XYZ1'))
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)
    plt.xlabel('Reformed Coordinates in X')
    plt.ylabel('Reformed Coordinates in Y')
    ax.set_zlabel('Reformed Coordinates in Z')
    ax.set_ylim(-250,200)
    ax.set_xlim(-250,200)
    ax.set_zlim(-240,240)
    plt.title(title)
    if save_id:
        plt.savefig(save_id)

    plt.show()
