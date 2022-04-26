import sys

from src.data import get_frames, get_predictions
from src.predictions import kalman_prediction

_INPUT_VIDEO_FILE = "data/video/reduced_mega_cut.mp4"
_OUTPUT_VIDEO_FILE = "data/linear/output.mp4"
_PREDICTIONS_FILE = "data/linear/reduced_mega_cut.mp4.predictions.slp"

_NUM_INDIVIDUALS = 2
_NUM_BODY_PARTS = 6
_MAX_FRAMES = 10
_PRED_FRAMES = 2

def main():
    print("reading input video file...")

    video_frames, _ = get_frames(_INPUT_VIDEO_FILE)
    ground_truth = get_predictions(_PREDICTIONS_FILE, _NUM_INDIVIDUALS, _NUM_BODY_PARTS)
    predictions_dict = kalman_prediction(ground_truth, _NUM_INDIVIDUALS, _NUM_BODY_PARTS, _MAX_FRAMES, pred_frames=_PRED_FRAMES)
    
    print("ready for visualization of predictions...")

if __name__=="__main__":
    main()