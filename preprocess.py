# Converts an MP4 video into a sequence of jpg images, capturing
# a frame every 'FRAMERATE' seconds

import os
import cv2


# TODO: make framerate and filename optional cmd-line parameters

# Tunable parameters
FPS = 30
VIDEO_DURATION_IN_SEC = 3600

# Derived parameters
# TIME_PER_FRAME = 0.5 # it will capture image every 0.5 seconds
TIME_PER_FRAME = 1 / FPS # it will capture FPS images every second


# data_base_dir = '/work/pi_rsitaram_umass_edu/sohaib/datasets/traffic_nexus/4_Olive_NS_/'
# filename = 't20190306-104139-0800_4_Olive_NS.mp4'

data_base_dir = 'data/bellevue_traffic'
filename = 'Bellevue_Bellevue_NE8th__2017-09-10_18-08-23.mp4'
output_dir = f'data/preprocessed_{FPS}fps'

if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)

vidcap = cv2.VideoCapture(os.path.join(data_base_dir, filename))

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        image_filename = os.path.join(output_dir, f'image{str(count).zfill(6)}.jpg')
        cv2.imwrite(image_filename, image)
        print(f'Saving image at: {image_filename}')     # save frame as JPG file
    return hasFrames

sec = 0
count = 1
success = getFrame(sec)

while success and sec < VIDEO_DURATION_IN_SEC:
    print(f'count: {count}, second: {sec}')
    count = count + 1
    sec = sec + TIME_PER_FRAME
    # sec = round(sec, 5)
    success = getFrame(sec)
