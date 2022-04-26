import math
import pykalman
import sleap 
import sys

import numpy as np

from collections import defaultdict

def kalman_prediction(sleap_predictions, num_individuals, num_body_parts, max_frames, smooth = True, pred_frames=1):
  return_dict = defaultdict(lambda: defaultdict(list))

  filters = list()
  for i in range(num_individuals):
    filters.append(pykalman.UnscentedKalmanFilter(n_dim_obs=num_body_parts * 2))

  predictions_arr = []
  actual_pose_arr = []

  for i in range(max_frames):
    if i > len(sleap_predictions[0]):
      break
    sys.stdout.write("\rpredicting frame: " + str(i))
    sys.stdout.flush()
    for j in range(num_individuals):
      kalman_input = sleap_predictions[j][i]
      for k in range(pred_frames):
        if smooth:
          kalman_input = (filters[j].smooth(kalman_input)[0]).reshape(2 * num_body_parts,)
        else:
          kalman_input = (filters[j].filter(kalman_input)[0]).reshape(2 * num_body_parts,)
        return_dict[j][i].append(kalman_input)
    sys.stdout.write("\r")
    sys.stdout.flush()
    
  return return_dict