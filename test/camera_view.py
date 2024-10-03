import cv2

from mechlmm_py import VisionCore, DebugCore

import threading
import time
import random

from dotenv import load_dotenv
load_dotenv()

class CameraView:
    def __init__(self, _data_path = "../output"):
        self.vision_core = VisionCore(_data_path)
        self.debug_core = DebugCore()
        self.debug_core.verbose = 3

        self.cam = cv2.VideoCapture(0)

        self.lmm_result = None

        self.vision_core.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vision_core.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def draw_bounding_boxes(slef, frame, objects):
        # Predefined list of colors (you can extend this if needed)
        color_list = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]

        # Iterate over all detected objects
        for idx, obj in enumerate(objects):
            temp_bounding_box = obj["position"]
            object_name = obj["name"]

            # Choose a color for the bounding box. Cycle through the predefined colors or generate random ones.
            color = color_list[idx % len(color_list)]  # Cycle through the predefined colors

            # Draw the object's name
            cv2.putText(frame, object_name, (temp_bounding_box[0] + 10, temp_bounding_box[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Draw the bounding box
            cv2.rectangle(frame, 
                        (temp_bounding_box[0], temp_bounding_box[1]), 
                        (temp_bounding_box[2], temp_bounding_box[3]), 
                        color, 2)

        return frame

    def run(self):
        while True:
            ret, frame = self.cam.read()

            if not ret:
                self.debug_core.log_error("Error: Could not read frame.")
                break
            
            self.vision_core.video_saver(frame)

            with self.lock:
                self.latest_frame = frame.copy()

            # if(self.lmm_result):
            #     temp_bounding_box = self.lmm_result["objects"][0]["position"]
            #     cv2.putText(frame, self.lmm_result["objects"][0]["name"], (temp_bounding_box[0] + 10, temp_bounding_box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            #     cv2.rectangle(frame, (temp_bounding_box[0], temp_bounding_box[1]), (temp_bounding_box[2], temp_bounding_box[3]), (255, 0, 0), 2)
            # cv2.imshow('Live Camera', frame)

            # Inside your main loop where you're processing the video feed
            if(self.lmm_result):
                frame = self.draw_bounding_boxes(frame, self.lmm_result["objects"])
            cv2.imshow('Live Camera', frame)


            if cv2.waitKey(1) == ord('q'):
                break
    
    def process_frames(self):
        while True:
            with self.lock:
                if hasattr(self, 'latest_frame'):
                    frame = self.latest_frame
                else:
                    frame = None

            if frame is not None:
                self.lmm_result = self.vision_core.frame_analyzer(frame)
            
            time.sleep(0.1)

if __name__ == '__main__':
    camera_view = CameraView()
    camera_view.run()