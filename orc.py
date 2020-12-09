import sys
import cv2
import time
import imutils
import argparse
import numpy as np

from imutils.video import FPS
from imutils.video import VideoStream

vehicle_list = []		# vehicle bounding box metadata buffer

# Vehicle Class ##########     # vehicle_list[] object class; described by the vehicle's tracking id, the number of frames it is tracked for and the coordinates of its bounding boxes


class Vehicle:
    def __init__(self, vehicle_id, frames_list=[], x1_list=[], y1_list=[], x2_list=[], y2_list=[]):
        self.vehicle_id = vehicle_id
        self.frames_list = frames_list
        self.x1_list = x1_list
        self.y1_list = y1_list
        self.x2_list = x2_list
        self.y2_list = y2_list

########## Vehicle Class ##########


def IOU(x11, y11, x21, y21, x12, y12, x22, y22):		# intersection over union of two bounding boxes
    b1 = (x11, y11, x21, y21)
    b2 = (x12, y12, x22, y22)
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    b1b2 = (x1, y1, x2, y2)
    area_b1b2 = (b1b2[2] - b1b2[0]) * (b1b2[3] - b1b2[1])
    area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    iou = area_b1b2 / (area_b1+area_b2-area_b1b2)
    return iou


def check_overlap(x11, y11, x21, y21, x12, y12, x22, y22):		# is there an overlap?
    if (y12 > y21) or (y11 > y22):
        return False
    if (x12 > x21) or (x11 > x22):
        return False
    else:
        return True


# Constructing Argument Parse to input from Command Line
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help='Path to prototxt')
ap.add_argument("-m", "--model", required=True, help='Path to model weights')
ap.add_argument("-c", "--confidence", type=float, default=0.7)
args = vars(ap.parse_args())

# Initialize Objects and corresponding colors which the model can detect
labels = ["background", "aeroplane", "bicycle", "bird",
          "boat", "bottle", "bus", "car", "cat", "chair", "cow",
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tvmonitor"]
# ~ labels = ["car", "bicycle", "person", "roadsign"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# Initialize Video Stream
print('[Status] Starting Video Stream...')
# ~ vs = VideoStream(src=0).start()
vs = cv2.VideoCapture('video.h264')
# ~ vs = cv2.VideoCapture('vid301_cropped.264')

time.sleep(2.0)
fps = FPS().start()
frame_number = 0
tracking_id = 0

# Loop Video Stream
while True:

    # Resize Frame to 400 pixels
    _, frame = vs.read()
    if type(frame) == type(None):
        break
    frame = imutils.resize(frame, width=1280)
    (h, w) = frame.shape[:2]
    # ~ print(h, w)

    cv2.line(frame, (0, int(0.2 * h)),
             (w, int(0.2 * h)), (0, 255, 0), thickness=2)

    # Converting Frame to Blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):

        # Extracting the confidence of predictions
        confidence = detections[0, 0, i, 2]

        # Filtering out weak predictions
        if confidence > args["confidence"]:

            # Extracting the index of the labels from the detection
            # Computing the (x,y) - coordinates of the bounding box
            idx = int(detections[0, 0, i, 1])

            if idx == 7:
                # Extracting bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ~ area = int((endX - startX) * (endY - startY))
                # ~ height = endY - startY
                # ~ print('height', h)
                # ~ print('width', w)

                if startY >= (0.2 * h):
                    # Drawing the prediction and bounding box
                    # ~ label = "{}: {:.2f}%: {} {} {} {}".format(
                    # ~ labels[idx], confidence * 100, startX, startY, endX, endY)

                    # ~ print(label)
                    if not vehicle_list:
                        temp_frames_list = []
                        temp_frames_list.append(frame_number)
                        temp_x1_list = []
                        temp_x1_list.append(startX)
                        temp_y1_list = []
                        temp_y1_list.append(startY)
                        temp_x2_list = []
                        temp_x2_list.append(endX)
                        temp_y2_list = []
                        temp_y2_list.append(endY)
                        vehicle_list.append(Vehicle(tracking_id, temp_frames_list,
                                                    temp_x1_list, temp_y1_list, temp_x2_list, temp_y2_list))
                        temp_track = 'id: ' + str(tracking_id)
                        tracking_id += 1

                    else:
                        iou_list = []
                        for v in vehicle_list:
                            frame_difference = frame_number - v.frames_list[-1]
                            # ~ print('overlap? ', check_overlap(
                            # ~ v.x1_list[-1], v.y1_list[-1], v.x2_list[-1], v.y2_list[-1], startX, startY, endX, endY))
                            if check_overlap(v.x1_list[-1], v.y1_list[-1], v.x2_list[-1], v.y2_list[-1], startX, startY, endX, endY) and frame_difference < 5:
                                intersection_over_union = IOU(
                                    v.x1_list[-1], v.y1_list[-1], v.x2_list[-1], v.y2_list[-1], startX, startY, endX, endY)
                                # ~ print(
                                # ~ 'operands: ', v.x1_list[-1], v.y1_list[-1], v.x2_list[-1], v.y2_list[-1], startX, startY, endX, endY)
                                iou_list.append(intersection_over_union)
                                # ~ print(intersection_over_union)
                            else:
                                # ~ print(v.x1_list[-1], v.y1_list[-1], v.x2_list[-1],
                                # ~ v.y2_list[-1], ' does not overlap with any bounding box')
                                iou_list.append(0)

                        if not all([v == 0 for v in iou_list]):
                            max_value = max(iou_list)
                            max_index = iou_list.index(max_value)
                            # ~ print('iou list: ', iou_list)
                            # ~ print('frame number: ', frame_number, ' max iou: ', max_value, ' b_old: (', vehicle_list[max_index].x1_list[-1], ',', vehicle_list[
                            # ~ max_index].y1_list[-1], ',', vehicle_list[max_index].x2_list[-1], ',', vehicle_list[max_index].y2_list[-1], ') b: ', (startX, startY, endX, endY))
                            vehicle_list[max_index].frames_list.append(
                                frame_number)
                            vehicle_list[max_index].x1_list.append(startX)
                            vehicle_list[max_index].y1_list.append(startY)
                            vehicle_list[max_index].x2_list.append(endX)
                            vehicle_list[max_index].y2_list.append(endY)
                            temp_track = 'id: ' + \
                                str(vehicle_list[max_index].vehicle_id)
                        else:
                            temp_frames_list = []
                            temp_frames_list.append(frame_number)
                            temp_x1_list = []
                            temp_x1_list.append(startX)
                            temp_y1_list = []
                            temp_y1_list.append(startY)
                            temp_x2_list = []
                            temp_x2_list.append(endX)
                            temp_y2_list = []
                            temp_y2_list.append(endY)
                            vehicle_list.append(Vehicle(
                                tracking_id, temp_frames_list, temp_x1_list, temp_y1_list, temp_x2_list, temp_y2_list))
                            temp_track = 'id: ' + str(tracking_id)
                            tracking_id += 1

                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), colors[idx], 2)

                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    label = "{} {}: {:.2f}%: {} {} {} {}".format(
                        temp_track, labels[idx], confidence * 100, startX, startY, endX, endY)
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow("Frame", frame)
    # ~ cv2.imwrite("/home/ee/Caffe-SSD-Object-Detection/Object Detection Caffe/stream/frame_number=" +
    # ~ str(frame_number)+".jpg", frame)
    frame_number += 1

    for i, x in enumerate(vehicle_list):
        d = frame_number - x.frames_list[-1]
        l = len(x.frames_list)
        if (d > 50 and l < 9):
            print('vehicle '+str(vehicle_list[i].vehicle_id)+' deleted')
            del vehicle_list[i]
            break

        if frame_number == 400:
            y_min_list = []
            y_max_list = []
            for car in vehicle_list:
                y_min_list.append(min(car.y1_list))
                y_max_list.append(max(car.y1_list))
            y_min_list.sort()
            y_max_list.sort()
            print('y_min:', y_min_list, len(y_min_list), '\n')
            print('y_max:', y_max_list, len(y_max_list), '\n')
            print('Optimal Frame Range:')
            print('y:', min(y_max_list), max(y_min_list))
            sys.exit()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("[Status] Video Stream Manually Terminated")
        break

    fps.update()

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))
print("[Info] Total frames: ", frame_number)

cv2.destroyAllWindows()
vs.release()

# ~ vehicle_count = 0
# ~ for v in vehicle_list:
    # ~ if v.frames_list[-1] < (frame_number - 50):
        # ~ vehicle_count += 1
        # ~ print('\n vehicle id: ', v.vehicle_id)
        # ~ for i, l in enumerate(v.frames_list):
            # ~ print('frame:', v.frames_list[i], 'x1:', v.x1_list[i], 'y1:',
                  # ~ v.y1_list[i], 'x2:', v.x2_list[i], 'y2:', v.y2_list[i])
# ~ print('\n'+str(vehicle_count)+' vehicles detected')

# ~ sys.exit()
