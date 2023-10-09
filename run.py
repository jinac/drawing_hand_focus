import argparse
import sys

import cv2
import mediapipe as mp
import numpy as np


def get_hand_data(landmarks, w, h):
    hand_lms = np.array([(lm.x*w, lm.y*h, lm.z) for lm in landmarks])
    
    hand_box = np.array([hand_lms.min(axis=0), hand_lms.max(axis=0)])
    hand_box = hand_box[:2, :2]
    hand_box[:, 0] = np.clip(hand_box[:, 0], 0.0, w)
    hand_box[:, 1] = np.clip(hand_box[:, 1], 0.0, h)
    hand_box = np.rint(hand_box).astype('int')

    hand_position = hand_lms.mean(axis=0)
    
    return hand_box, hand_position

class HandTracker():
    def __init__(self, w, h, moving_avg_window_size=5):
        self.pos_arr = []
        self.max_arr_len = moving_avg_window_size
        self.last_pos = None
        self.last_box = None
        self.frame_w = w
        self.frame_h = h
        self.diff_frame = None
        
    def update(self, hands_results):
        hand_box, hand_position = get_hand_data(hands_results, self.frame_w, self.frame_h)
        if self.last_pos is not None:
            self.diff_frame = np.linalg.norm(hand_position - self.last_pos)
        self.last_box = hand_box
        self.last_pos = hand_position
        self.pos_arr.append(hand_position)
        while (len(self.pos_arr) > self.max_arr_len):
            _ = self.pos_arr.pop(0)
            
        return (self.diff_frame, self.last_pos, self.last_box, np.mean(self.pos_arr, axis=0))

class State():
    def __init__(self, start_threshold=5, loss_threshold=10):
        self.hand_crop_on = False
        self.start_threshold = start_threshold
        self.loss_threshold = loss_threshold
        self.num_detect_loss = 0
        self.num_frames_detected = 0
    
    def get_crop_state(self, new_detect_bool):
        if new_detect_bool:
            self.num_detect_loss = 0
            self.num_frames_detected += 1
            if self.num_frames_detected > self.start_threshold:
                self.hand_crop_on = True
        else:
            self.num_detect_loss += 1
            if self.num_detect_loss > self.loss_threshold:
                self.hand_crop_on = False
                self.num_frames_detected = 0

        return self.hand_crop_on

class FrameCropper():
    def __init__(self, frame_w, frame_h, crop_dec=0.25, offset=(0.5, 0.0)):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.crop_w, self.crop_h = self.calc_crop_frame(crop_dec, offset)
        self.hand_x, self.hand_y = offset

        # self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def calc_crop_frame(self, crop_dec, offset):
        crop_w = np.rint(crop_dec * self.frame_w).astype('int')
        crop_h = np.rint(crop_dec * self.frame_h).astype('int')
        return crop_w, crop_h
    
    def crop_frame(self, frame, hand_loc):
        x0 = int(max(min(hand_loc[0] - (self.hand_x * self.crop_w), self.frame_w - self.crop_w), 0))
        x1 = x0 + self.crop_w
        y0 = int(max(min(hand_loc[1] - (self.hand_y * self.crop_h), self.frame_h - self.crop_h), 0))
        y1 = y0 + self.crop_h
        # txt = ",".join([str(a) for a in [x0, x1, y0, y1]])

        new_frame = cv2.resize(frame[y0:y1, x0:x1, :], (self.frame_w, self.frame_h))
        # cv2.putText(new_frame, txt, (100,100), self.font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        return new_frame


def process_results(results, w, h):
    out = []
    
    if results.multi_hand_landmarks:
        for idx, handedness in enumerate(results.multi_handedness):
            entry = handedness.classification[0]
            if  entry.label == "Left" and entry.score > 0.9:
                ldmks = results.multi_hand_landmarks[idx].landmark
                _, loc = get_hand_data(ldmks, w, h)
                out.append((ldmks, loc, idx))
                
    return out

def filter_closest(cur_frame_hands_data, prev_hand_position):
    pos_arr = np.array([l[1] for l in cur_frame_hands_data])
    closest_idx = np.linalg.norm(pos_arr - prev_hand_position, axis=1).argmin()
    return cur_frame_hands_data[closest_idx]

def open_input_stream(cam_id, video_filepath):
    use_cam = cam_id is not None
    use_video = video_filepath is not None
    if use_cam == use_video:
        if use_cam:
            sys.exit("Cannot open both camera and video stream. Pick one.")
        else:
            sys.exit("No input camera or video specified")
    else:
        if use_cam:
            return cv2.VideoCapture(cam_id)
        else:
            return cv2.VideoCapture(video_filepath)

def main():
    parser = argparse.ArgumentParser(description='Create camera stream with automated hand croppinmg')
    parser.add_argument('-c', '--cam',
                        help='')
    parser.add_argument('-v', '--video',
                        help='')
    parser.add_argument('-o', '--out',
                        help='')
    parser.add_argument('-w', '--window', default=10, type=int,
                        help='moving window average size')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='')
    args = parser.parse_args()

    cam = args.cam
    video_filepath = args.video
    debug_flag = args.debug
    out_video_filepath = args.out
    moving_avg_window_size = args.window

    # Init camera input
    cap = open_input_stream(cam, video_filepath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Init camera output
    out_resolution = (w, h)
    if debug_flag:
        out_resolution = (w*2, h)
    out = cv2.VideoWriter(out_video_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_resolution)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tracker = HandTracker(w=w, h=h, moving_avg_window_size=moving_avg_window_size)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(model_complexity=0,
                          static_image_mode=False)
    cropper = FrameCropper(w, h, crop_dec = 0.3)
    crop_state = State(start_threshold=10, loss_threshold=20)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if debug_flag:
            disp_frame = frame.copy()
        cropped_frame = frame.copy()
        results = hands.process(rgb_frame)
        data = process_results(results, w, h)

        if data:
            crop_on = crop_state.get_crop_state(True)
            if len(data) > 1:
                data = filter_closest(data, tracker.last_pos)
            else:
                data = data[0]
            ldmks, hand_loc, idx = data
            crop_data = tracker.update(ldmks)
        else:
            crop_on = crop_state.get_crop_state(False)
        
        if crop_on:
            hand_box = crop_data[2]
            if debug_flag:
                cv2.rectangle(disp_frame, hand_box[0, :], hand_box[1, :], (255, 255, 0), 3)
                cv2.putText(disp_frame, str(idx), (100,100), font, 3, (255, 255, 0), 2, cv2.LINE_AA)

            hand_pos = crop_data[3]
            cropped_frame = cropper.crop_frame(frame, hand_pos)

        if debug_flag:
            out_frame = np.concatenate((disp_frame, cropped_frame), axis=1)
        else:
            out_frame = cropped_frame
        # cv2.imshow('frame', out_frame)
        out.write(out_frame)
            
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()