# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:01:06 2020

@author: ASUS
"""


#loading Cam / video
#Load YOLO , Weights, Labels
#Results Centriod, B Box, Confidence(80 / 90), Probability(0 / 1/ 2)
#Fets Only Person Values From Object Detection
#Start Calculating Euc Dist
#Violations
#Draw Circles / Rect
# Output

from packages import social_distancing_configuration as config
from packages.Object_detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os

labelsPath = os.path.sep.join([config.MODEL_PATH,"coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
print("[INFO] loading YOLO from disk....")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
if config.USE_GPU:
    print("[INFO] setting preferable backend and target to CUDA....")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#initialise the video stream & pointer to ouput video file
print("[INFO] accessing video stream....")
vs = cv2.VideoCapture(r"pedestrian.mp4" if "pedestrian.mp4" else 0)
writer = None
while True:
    #read the next frame from the file
    (grabbed, frame) = vs.read()
    #if the frame was not grabbed, then we have reached the end
    #of the stream
    if not grabbed:
        break
    #resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    #initialize the set of indexes that violate the minimum social distancing
    violate = set()
    if len(results) >= 2:
    #extract all centroids from the results and compute the Euclidean distances between all pairs of the centroid
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
    #loop over the upper traingle of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
            #check to see if the distance between any two
            #centroid pairs is less then the configured number
            #of pixels
                if D[i,j] < config.MIN_DISTANCE:
                #update our violation set with the indexes of
                #the centroid pairs
                    violate.add(i)
                    violate.add(j)
#loop over the result
    for (i, (prob, bbox, centroid)) in enumerate(results):
    #extract the bounding box and centroid coordinates, then
    #initialise the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
    #if the index pair exists within the violation set,then update the color
        if i in violate:
            color = (0, 0, 225)
    #draw (1) a bounding box around the person and (2) the centroid coordinate of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
#draw the total number of social distancing violations on the output frame
    text = "Social Distance Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF
#if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
#if an output video file path has been supplied and the video 
#writer has not been initialized, do so now
    if r"social-distance-detector" != "" and writer is None:
    #initialise our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(r"output.mp4", fourcc, 25, (frame.shape[1], frame.shape[0]), True)
#if the video writer is not None, write the frame to the output video file
    if writer is not None:
        writer.write(frame)
        cv2.destroyAllWindows()
    