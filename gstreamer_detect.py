from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
from gi.repository import GObject, Gst, GLib
import sys
import numpy
import gi
import cv2
gi.require_version('Gst', '1.0')

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


width = 0
height = 0

start_time = time.time()


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End of stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def gst_to_opencv(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    # ~ print(caps.get_structure(0).get_value('format'))
    # ~ print(caps.get_structure(0).get_value('height'))
    # ~ print(caps.get_structure(0).get_value('width'))

    # ~ print(buf.get_size())

    arr = numpy.ndarray((caps.get_structure(0).get_value('height'), caps.get_structure(
        0).get_value('width'), 3), buffer=buf.extract_dup(0, buf.get_size()), dtype=numpy.uint8)
    return arr


def new_buffer(sink, data):
    global image_arr
    sample = sink.emit("pull-sample")
    arr = gst_to_opencv(sample)
    image_arr = arr
    return Gst.FlowReturn.OK


Gst.init(None)
image_arr = None

print("Creating Pipeline \n")
pipeline = Gst.Pipeline()

if not pipeline:
    sys.stderr.write("Unable to create Pipeline \n")

print("Creating Source \n")
source = Gst.ElementFactory.make('filesrc', 'file-source')
if not source:
    sys.stderr.write("Unable to create Source \n")

print("Creating H264Parser \n")
h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
if not h264parser:
    sys.stderr.write("Unable to create h264 Parser \n")

print("Creating Decoder \n")
decoder = Gst.ElementFactory.make("avdec_h264", "vdecoder")
if not h264parser:
    sys.stderr.write("Unable to create Decoder \n")

print("Creating Video Converter \n")
converter = Gst.ElementFactory.make("videoconvert", "converter")
if not converter:
    sys.stderr.write("Unable to create Video Converter \n")

print("Creating Appsink \n")
sink = Gst.ElementFactory.make("appsink", "sink")
if not sink:
    sys.stderr.write("Unable to create Appsink \n")

print("Playing file video.h264")
source.set_property('location', 'video.h264')

sink.set_property('wait-on-eos', False)
sink.set_property('drop', True)
sink.set_property('max-buffers', 60)
sink.set_property('emit-signals', True)

caps = Gst.caps_from_string(
    "video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer, format=(string){rggb,bggr,grbg,gbrg}")
sink.set_property('caps', caps)
sink.connect("new-sample", new_buffer, sink)

print("Adding elements to Pipeline \n")
pipeline.add(source)
pipeline.add(h264parser)
pipeline.add(decoder)
pipeline.add(converter)
pipeline.add(sink)

print("Linking elements in the Pipeline \n")
source.link(h264parser)
h264parser.link(decoder)
decoder.link(converter)
converter.link(sink)

loop = GLib.MainLoop()

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)

print("Starting pipeline \n")

pipeline.set_state(Gst.State.PLAYING)

frame_number = 0

# ~ width = 1280
# ~ height = 720
# ~ dim = (width, height)

stop = 0

while True:
    message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
    if image_arr is not None:
        if stop:
            sys.exit()

        cv2.imshow("car_detection", image_arr)

        # ~ frame_number += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[Status] Video Stream Manually Terminated")
            break

    if message:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(("Error received from element %s: %s" %
                   (message.src.get_name(), err)))
            print(("Debugging information: %s" % debug))
            break
        elif message.type == Gst.MessageType.EOS:
            stop = 1
            print("End-Of-Stream reached.")
            # ~ sys.exit()
            break
        elif message.type.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                print(("Pipeline state changed from %s to %s." %
                       (old_state.value_nick, new_state.value_nick)))
        else:
            print("Unexpected message received.")

pipeline.set_state(Gst.State.NULL)
