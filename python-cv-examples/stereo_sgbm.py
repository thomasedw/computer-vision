#####################################################################

# Example : stereo vision from 2 connected cameras using Semi-Global
# Block Matching. For usage: python3 ./stereo_sgbm.py -h

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015-18 Engineering & Computer Science, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgements:

# http://opencv-python-tutroals.readthedocs.org/en/latest/ \
# py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html

# http://docs.ros.org/electric/api/cob_camera_calibration/html/calibrator_8py_source.html
# OpenCV 3.0 example - stereo_match.py

# Andy Pound, Durham University, 2016 - calibration save/load approach

#####################################################################

# TODO:

# add sliders for some stereo parameters


#####################################################################

import cv2
import sys
import numpy as np
import os
import argparse
import time

#####################################################################
# define target framerates in fps (may not be achieved)

calibration_capture_framerate = 2
disparity_processing_framerate = 25

#####################################################################
# wrap different kinds of stereo camera - standard (v4l/vfw), ximea, ZED

class StereoCamera:
    def __init__(self, args):

        self.xiema = args.ximea
        self.zed = args.zed
        self.cameras_opened =  False

        if args.ximea:

            # ximea requires specific API offsets in the open commands

            self.camL = cv2.VideoCapture()
            self.camR = cv2.VideoCapture()

            if not((self.camL.open(cv2.CAP_XIAPI)) and (self.camR.open(cv2.CAP_XIAPI + 1))):
                print("Cannot open pair of Ximea cameras connected.")
            exit()

        elif args.zed:

            # ZED is a single camera interface with L/R returned as 1 image

            self.camZED = cv2.VideoCapture()
            if not(self.camZED.open(args.camera_to_use)):
                print("Cannot open connected ZED stereo camera as camera #: ", args.camera_to_use)
                exit()

            # report resolution currently in use for ZED (as various options exist)
            # can use .get()/.set() to read/change also

            _, frame = self.camZED.read()
            height,width, channels = frame.shape
            print()
            print("ZED left/right resolution: ", int(width/2), " x ",  int(height))
            print()

        else:

            # by default two standard system connected cams from the default video backend

            self.camL = cv2.VideoCapture()
            self.camR = cv2.VideoCapture()
            if not((self.camL.open(args.camera_to_use)) and (self.camR.open(args.camera_to_use + 1))):
                print("Cannot open pair of system cameras connected starting at camera #:", args.camera_to_use)
                exit()

        cameras_opened = True

    def swap_cameras(self):
        if not(self.zed):
            # swap the cameras - for all but ZED camera
            tmp = self.camL
            self.camL = self.camR
            self.camR = tmp

    def get_frames(self): # return left, right
        if self.zed:

            # grab single frame from camera (read = grab/retrieve)
            # and split into Left and Right

            _, frame = self.camZED.read()
            height,width, channels = frame.shape
            frameL= frame[:,0:int(width/2),:]
            frameR = frame[:,int(width/2):width,:]
        else:
            # grab frames from camera (to ensure best time sync.)

            self.camL.grab()
            self.camR.grab()

            # then retrieve the images in slow(er) time
            # (do not be tempted to use read() !)

            _, frameL = self.camL.retrieve()
            _, frameR = self.camR.retrieve()
#
        return frameL, frameR

#####################################################################
# deal with optional arguments

parser = argparse.ArgumentParser(description='Perform full stereo calibration and SGBM matching.')
parser.add_argument("--ximea", help="use a pair of Ximea cameras", action="store_true")
parser.add_argument("--zed", help="use a Stereolabs ZED stereo camera", action="store_true")
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-cbx", "--chessboardx", type=int, help="specify number of internal chessboard squares in x-direction", default=6)
parser.add_argument("-cby", "--chessboardy", type=int, help="specify number of internal chessboard squares in y-direction", default=9)
parser.add_argument("-cbw", "--chessboardw", type=float, help="specify width/height of chessboard squares in mm", default=40.0)
parser.add_argument("-cp", "--calibration_path", type=str, help="specify path to calibration files to load", default=-1)

args = parser.parse_args()

#####################################################################

# flag values to enter processing loops - do not change

keep_processing = True
do_calibration = False

#####################################################################

# STAGE 1 - open 2 connected cameras

# define video capture object

stereo_camera = StereoCamera(args)

# define display window names

windowNameL = "LEFT Camera Input" # window name
windowNameR = "RIGHT Camera Input" # window name

# create window by name (as resizable)

cv2.namedWindow(windowNameL, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameR, cv2.WINDOW_NORMAL)

# set sizes and set windows

frameL, frameR = stereo_camera.get_frames()

height, width, channels = frameL.shape
cv2.resizeWindow(windowNameL, width, height)
height, width, channels = frameR.shape
cv2.resizeWindow(windowNameR, width, height)

# controls

print("s : swap cameras left and right")
print("e : export camera calibration to file")
print("l : load camera calibration from file")
print("x : exit")
print()
print("space : continue to next stage")
print()

while (keep_processing):

    # get frames from camera

    frameL, frameR = stereo_camera.get_frames()

    # display image

    cv2.imshow(windowNameL,frameL)
    cv2.imshow(windowNameR,frameR)

    # start the event loop - essential

    # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
    # It waits for specified milliseconds for any keyboard event.
    # If you press any key in that time, the program continues.
    # If 0 is passed, it waits indefinitely for a key stroke.
    # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

    key = cv2.waitKey(40) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    # It can also be set to detect specific key strokes by recording which key is pressed

    # e.g. if user presses "x" then exit

    if (key == ord(' ')):
        keep_processing = False
    elif (key == ord('x')):
        exit()
    elif (key == ord('s')):
        # swap the cameras if specified

        stereo_camera.swap_cameras()
    elif (key == ord('l')):

        if (args.calibration_path == -1):
            print("Error - no calibration path specified:")
            exit()

        # load calibration from file

        os.chdir(args.calibration_path)
        print('Using calibration files: ', args.calibration_path)
        mapL1 = np.load('mapL1.npy')
        mapL2 = np.load('mapL2.npy')
        mapR1 = np.load('mapR1.npy')
        mapR2 = np.load('mapR2.npy')

        keep_processing = False
        do_calibration = True # set to True to skip next loop

#####################################################################

# STAGE 2: perform intrinsic calibration (removal of image distortion in each image)

termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# set up a set of real-world "object points" for the chessboard pattern

patternX = args.chessboardx
patternY = args.chessboardy
square_size_in_mm = args.chessboardw

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((patternX*patternY,3), np.float32)
objp[:,:2] = np.mgrid[0:patternX,0:patternY].T.reshape(-1,2)
objp = objp * square_size_in_mm

# create arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsR = [] # 2d points in image plane.
imgpointsL = [] # 2d points in image plane.

# count number of chessboard detection (across both images)
chessboard_pattern_detections = 0

print()
print("--> hold up chessboard")
print("press space when ready to start calibration stage  ...")
print()

while (not(do_calibration)):

        # get frames from camera

        frameL, frameR = stereo_camera.get_frames()

        # convert to grayscale

        grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners in the image
        # (change flags to perhaps improve detection ?)

        retR, cornersL = cv2.findChessboardCorners(grayL, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        retL, cornersR = cv2.findChessboardCorners(grayR, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)

        if ((retR == True) and (retL == True)):

            chessboard_pattern_detections += 1

            # add object points to global list

            objpoints.append(objp)

            # refine corner locations to sub-pixel accuracy and then

            corners_sp_L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),termination_criteria_subpix)
            imgpointsL.append(corners_sp_L)
            corners_sp_R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),termination_criteria_subpix)
            imgpointsR.append(corners_sp_R)

            # Draw and display the corners

            drawboardL = cv2.drawChessboardCorners(frameL, (patternX,patternY), corners_sp_L,retL)
            drawboardR = cv2.drawChessboardCorners(frameR, (patternX,patternY), corners_sp_R,retR)

            text = 'detected: ' + str(chessboard_pattern_detections)
            cv2.putText(drawboardL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)

            cv2.imshow(windowNameL,drawboardL)
            cv2.imshow(windowNameR,drawboardR)
        else:
            text = 'detected: ' + str(chessboard_pattern_detections)
            cv2.putText(frameL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)

            cv2.imshow(windowNameL,frameL)
            cv2.imshow(windowNameR,frameR)

        # start the event loop

        key = cv2.waitKey(int(1000/calibration_capture_framerate)) & 0xFF # wait 500ms between frames - i.e. 2 fps
        if (key == ord(' ')):
            do_calibration = True
        elif (key == ord('x')):
            exit()

# perform calibration on both cameras - uses [Zhang, 2000]

if (chessboard_pattern_detections > 0): # i.e. if we did not load a calibration

    print("START - intrinsic calibration ...")

    ret, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1],None,None)
    ret, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1],None,None)

    print("FINISHED - intrinsic calibration")

    # perform undistortion of the images

    keep_processing = True

    print()
    print("-> dislaying undistortion")
    print("press space to continue to next stage ...")
    print()

while (keep_processing):

    # get frames from camera

    frameL, frameR = stereo_camera.get_frames()

    undistortedL = cv2.undistort(frameL, mtxL, distL, None, None)
    undistortedR = cv2.undistort(frameR, mtxR, distR, None, None)

    # display image

    cv2.imshow(windowNameL,undistortedL)
    cv2.imshow(windowNameR,undistortedR)

    # start the event loop - essential

    key = cv2.waitKey(int(1000/disparity_processing_framerate)) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    if (key == ord(' ')):
        keep_processing = False
    elif (key == ord('x')):
        exit()

# show mean re-projection error of the object points onto the image(s)

if (chessboard_pattern_detections > 0): # i.e. if we did not load a calibration

    tot_errorL = 0
    for i in range(len(objpoints)):
        imgpointsL2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
        errorL = cv2.norm(imgpointsL[i],imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
        tot_errorL += errorL

    print("LEFT: Re-projection error: ", tot_errorL/len(objpoints))

    tot_errorR = 0
    for i in range(len(objpoints)):
        imgpointsR2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
        errorR = cv2.norm(imgpointsR[i],imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
        tot_errorR += errorR

    print("RIGHT: Re-projection error: ", tot_errorR/len(objpoints))

#####################################################################

# STAGE 3: perform extrinsic calibration (recovery of relative camera positions)

# this takes the existing calibration parameters used to undistort the individual images as
# well as calculated the relative camera positions - represented via the fundamental matrix, F

# alter termination criteria to (perhaps) improve solution - ?

termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

if (chessboard_pattern_detections > 0): # i.e. if we did not load a calibration
    print()
    print("START - extrinsic calibration ...")
    (rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F) = \
    cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR,  grayL.shape[::-1], criteria=termination_criteria_extrinsics, flags=0)

    print("START - extrinsic calibration ...")

    print("STEREO: RMS left to  right re-projection error: ", rms_stereo)

#####################################################################

# STAGE 4: rectification of images (make scan lines align left <-> right

# N.B.  "alpha=0 means that the rectified images are zoomed and shifted so that
# only valid pixels are visible (no black areas after rectification). alpha=1 means
# that the rectified image is decimated and shifted so that all the pixels from the original images
# from the cameras are retained in the rectified images (no source image pixels are lost)." - ?

if (chessboard_pattern_detections > 0): # i.e. if we did not load a calibration
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r,  grayL.shape[::-1], R, T, alpha=-1)

# compute the pixel mappings to the rectified versions of the images

if (chessboard_pattern_detections > 0): # i.e. if we did not load a calibration
    mapL1, mapL2 = cv2.initUndistortRectifyMap(camera_matrix_l, dist_coeffs_l, RL, PL, grayL.shape[::-1], cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(camera_matrix_r, dist_coeffs_r, RR, PR, grayR.shape[::-1], cv2.CV_32FC1)

    print()
    print("-> displaying rectification")
    print("press space to continue to next stage ...")

    keep_processing = True

while (keep_processing):

    # get frames from camera

    frameL, frameR = stereo_camera.get_frames()

    # undistort and rectify based on the mappings (could improve interpolation and image border settings here)

    undistorted_rectifiedL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    undistorted_rectifiedR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    # display image

    cv2.imshow(windowNameL,undistorted_rectifiedL)
    cv2.imshow(windowNameR,undistorted_rectifiedR)

    # start the event loop - essential

    key = cv2.waitKey(int(1000/disparity_processing_framerate)) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    # It can also be set to detect specific key strokes by recording which key is pressed

    # e.g. if user presses "x" then exit

    if (key == ord(' ')):
        keep_processing = False
    elif (key == ord('x')):
        exit()

#####################################################################

# STAGE 5: calculate stereo depth information

# uses a modified H. Hirschmuller algorithm [HH08] that differs (see opencv manual)

# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21)

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

print()
print("-> display disparity image")
print("press x to exit")
print("press e to export calibration")
print("press c for false colour mapped disparity")
print("press f for fullscreen disparity")

print()

# set up defaults for disparity calculation

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

keep_processing = True
apply_colourmap = False

# set up disparity window to be correct size

windowNameD = "SGBM Stereo Disparity - Output" # window name
cv2.namedWindow(windowNameD, cv2.WINDOW_NORMAL)
height, width, channels = frameL.shape
cv2.resizeWindow(windowNameD, width, height)

while (keep_processing):

    # get frames from camera

    frameL, frameR = stereo_camera.get_frames()

    # remember to convert to grayscale (as the disparity matching works on grayscale)

    grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)

    # undistort and rectify based on the mappings (could improve interpolation and image border settings here)
    # N.B. mapping works independant of number of image channels

    undistorted_rectifiedL = cv2.remap(grayL, mapL1, mapL2, cv2.INTER_LINEAR)
    undistorted_rectifiedR = cv2.remap(grayR, mapR1, mapR2, cv2.INTER_LINEAR)

    # compute disparity image from undistorted and rectified versions
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(undistorted_rectifiedL,undistorted_rectifiedR)
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    # display image

    cv2.imshow(windowNameL,undistorted_rectifiedL)
    cv2.imshow(windowNameR,undistorted_rectifiedR)

    #display disparity - which ** for display purposes only ** we re-scale to 0 ->255

    if (apply_colourmap):

        disparity_colour_mapped = cv2.applyColorMap((disparity_scaled * (256. / max_disparity)).astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imshow(windowNameD, disparity_colour_mapped)
    else:
        cv2.imshow(windowNameD, (disparity_scaled * (256. / max_disparity)).astype(np.uint8))

    # start the event loop - essential

    key = cv2.waitKey(int(1000/disparity_processing_framerate)) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

    # It can also be set to detect specific key strokes by recording which key is pressed

    # e.g. if user presses "x" then exit

    if (key == ord(' ')):
        keep_processing = False
    elif (key == ord('c')):
        apply_colourmap = not(apply_colourmap)
    elif (key == ord('x')):
        exit()
    elif (key == ord('f')):
        cv2.setWindowProperty(windowNameD, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif (key == ord('e')):
        # export to named folder path as numpy data
        try:
            os.mkdir('calibration')
        except OSError:
            print("Exporting to existing calibration archive directory.")
        os.chdir('calibration')
        folderName = time.strftime('%d-%m-%y-%H-%M-rms-') + ('%.2f' % rms_stereo) + '-zed-' + str(int(args.zed)) + '-ximea-' + str(int(args.ximea))
        os.mkdir(folderName)
        os.chdir(folderName)
        np.save('mapL1', mapL1)
        np.save('mapL2', mapL2)
        np.save('mapR1', mapR1)
        np.save('mapR2', mapR2)
        cv_file = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("source", ' '.join(sys.argv[0:]))
        cv_file.write("description", "camera matrices K for left and right, distortion coefficients for left and right, 3D rotation matrix R, 3D translation vector T, Essential matrix E, Fundamental matrix F, disparity to depth projection matrix Q")
        cv_file.write("K_l", camera_matrix_l)
        cv_file.write("K_r", camera_matrix_r)
        cv_file.write("distort_l", dist_coeffs_l)
        cv_file.write("distort_r", dist_coeffs_r)
        cv_file.write("R", R)
        cv_file.write("T", T)
        cv_file.write("E", E)
        cv_file.write("F", F)
        cv_file.write("Q", Q)
        cv_file.release()
        print("Exported to path: ", folderName)

#####################################################################

# close all windows and cams.

cv2.destroyAllWindows()

#####################################################################
