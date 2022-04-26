import sleap
import cv2
import math
import numpy as np

from collections import defaultdict

def get_frames(video_file):
  print("reading video file...")

  video = cv2.VideoCapture(video_file)

  success, frame = video.read()
  frames_array = list()

  fps = video.get(cv2.CAP_PROP_FPS)
  while success:
    frames_array.append(frame)
    success, frame = video.read()

  video.release()

  print("closing video file...")

  return frames_array, fps

def get_predictions(predictions_file, instances, expected_body_parts):
  print("reading sleap predictions...")

  return_dict = defaultdict(list)

  sleap_labels = sleap.load_file(predictions_file)
  frame_count = len(sleap_labels.labeled_frames)
  
  for i in range(frame_count):
    curr_frame = sleap_labels[i].numpy()
    
    for j in range(instances):
      try:
        instance_pose = curr_frame[j]
      except:
        instance_pose = np.zeros(expected_body_parts * 2)
    
      instance_pose[np.isnan(instance_pose)] = 0
      instance_pose = instance_pose.flatten()
      return_dict[j].append(instance_pose)

  return return_dict