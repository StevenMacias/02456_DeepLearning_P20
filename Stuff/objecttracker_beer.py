#from centroidtracker import CentroidTracker
#import centroidtracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path)
from centroidtracker import CentroidTracker

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

ct = CentroidTracker()
(H, W) = (None, None)
# load our serialized model from disk
print("[INFO] loading model...")
net = torch.load(args["model"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
convert_tensor = transforms.ToTensor()
net.eval()
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
color_map = ["","red","green"]

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    img = convert_tensor(frame)
	# if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = img.shape[:2]
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
    with torch.no_grad():
        detections = net([img.to(device)])
        rects = []
    
    # loop over the detections
    """
    for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
			# draw a bounding box surrounding the object so we can
			# visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
    """

    predicted_img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # create rectangle image
    img1 = ImageDraw.Draw(predicted_img)  
    
    
    for pred in detections:
        boxes = pred["boxes"]
        labels = pred["labels"]
        index = 0
        for box in boxes: 
            shape = [(box[0], box[1]), (box[2], box[3])]
            img1.rectangle(shape, outline =color_map[labels[index].item()])
            index += 1 
            
    # update our centroid tracker using the computed set of bounding
	# box rectangles
    objects = ct.update(rects)
	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

#python objectracker_beer.py  --model modelname