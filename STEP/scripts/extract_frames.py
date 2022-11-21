import cv2
import argparse
import os
import shutil
from utils.progress_bar import progress_bar

parser = argparse.ArgumentParser()
parser.add_argument('-video_name', type=str, default='P02T02C06.mp4')
args = parser.parse_args()
vidcap = cv2.VideoCapture('./datasets/demo/'+ args.video_name)

## clean up the frames before performing new extraction
if (os.path.exists('./datasets/demo/frames')):
  shutil.rmtree('./datasets/demo/frames')
os.mkdir("./datasets/demo/frames")
success,image = vidcap.read()
count = 0
dir = "./datasets/demo/frames/" + str(args.video_name).replace(".mp4", "")
if (os.path.exists(dir) == False):
  os.mkdir(dir)

length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar_instance = progress_bar(length)
progress_bar_instance.display_bar()
while success:
  cv2.imwrite(str(dir)+"/frame%04d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  count += 1
  progress_bar_instance.update_bar()

print(args.video_name + " has been extracted succesfully.")