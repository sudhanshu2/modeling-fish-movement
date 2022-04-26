import cv2
import matplotlib.pyplot as plt

def visualize_points(frames_array, predictions_array, output_video_path, fps):
  height, width, channels = frames_array[0].shape 
  fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
  video_out = cv2.VideoWriter(output_video_path, fourcc, fps, (width,height))

  for i in range(len(predictions_array)):
    curr_points = predictions_array[i]
    curr_frame = frames_array[i]
    for point in curr_points:
        curr_frame = cv2.circle(curr_frame, (int(point[0]), int(point[1])), radius=0, color=(0, 255, 255), thickness=10)
      
    video_out.write(curr_frame)
  
  video_out.release()

def get_predictions(predictions_array):
  x = list()
  y = list()
  counter = 1
  for prediction in predictions_array:
    x.append(counter)
    y.append(prediction[0][1])
    counter += 1
  
  plt.plot(x, y)
  plt.show()