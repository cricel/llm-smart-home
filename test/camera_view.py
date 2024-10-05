import cv2

from mechlmm_py import VisionCore, DebugCore
from PIL import Image
import threading
import time
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

    def extract_coordinates(text):
        try:
            # Try to parse the entire text as JSON
            data = json.loads(text)
            if isinstance(data, list):
                return [coord for coord in data if len(coord) == 4 and all(isinstance(c, (int, float)) for c in coord)]
        except json.JSONDecodeError:
            # If it's not valid JSON, fall back to regex matching
            import re
            regex = r'\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]'
            matches = re.findall(regex, text)
            return [json.loads(match) for match in matches]

    def run(self):
        while True:
            ret, frame = self.cam.read()

            if not ret:
                self.debug_core.log_error("Error: Could not read frame.")
                break
            
            self.vision_core.video_saver(frame)

            with self.lock:
                self.latest_frame = frame.copy()

            if(self.lmm_result):
                # self.debug_core.log_error(self.lmm_result)
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
                try:
                    # temp_bounding_box = list(map(int, self.lmm_result[0]["args"]["objects"][0]["position"]))
                    # cv2.putText(frame, self.lmm_result[0]["args"]["objects"][0]["name"], (temp_bounding_box[0] + 10, temp_bounding_box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    # cv2.rectangle(frame, (temp_bounding_box[0], temp_bounding_box[1]), (temp_bounding_box[2], temp_bounding_box[3]), (255, 0, 0), 2)
                    bounding_boxes = []

                    # Iterate over all the objects in the result
                    for obj in self.lmm_result[0]["args"]['objects']:
                        # Convert the bounding box coordinates to integers
                        temp_bounding_box = list(map(int, obj['position']))
                        # Append the bounding box and object name to the list
                        bounding_boxes.append((temp_bounding_box, obj['name']))

                    for i, (box, name) in enumerate(bounding_boxes):
                        ymin, xmin, ymax, xmax = [int(coord / 1000 * frame.shape[0 if j % 2 == 0 else 1]) for j, coord in enumerate(box)]
                        color = colors[i % len(colors)]
                        
                        # Draw the rectangle
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        
                        # Add the name of the object above the rectangle
                        cv2.putText(frame, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except:
                    pass
            cv2.imshow('Live Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    def runs(self):
        while True:
            ret, frame = self.cam.read()

            if not ret:
                self.debug_core.log_error("Error: Could not read frame.")
                break
            
            self.vision_core.video_saver(frame)

            with self.lock:
                self.latest_frame = frame.copy()

            if self.lmm_result:
                colors = ['r', 'g', 'b', 'y', 'm', 'c']  # Colors for bounding boxes
                try:
                    bounding_boxes = []

                    # Iterate over all the objects in the result
                    for obj in self.lmm_result[0]["args"]['objects']:
                        # Convert the bounding box coordinates to integers
                        temp_bounding_box = list(map(int, obj['position']))
                        # Append the bounding box and object name to the list
                        bounding_boxes.append((temp_bounding_box, obj['name']))

                    # Convert the OpenCV frame to a PIL Image
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Create the plot
                    fig, ax = plt.subplots(1)
                    ax.imshow(img)  # Show the image

                    for i, (box, name) in enumerate(bounding_boxes):
                        ymin, xmin, ymax, xmax = [coord / 1000 for coord in box]
                        width = xmax - xmin
                        height = ymax - ymin
                        rect = Rectangle((xmin * img.width, ymin * img.height), width * img.width, height * img.height,
                                        linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none')
                        ax.add_patch(rect)

                        # Add the name of the object above the rectangle
                        ax.text(xmin * img.width, (ymin * img.height) - 10, name, color=colors[i % len(colors)], fontsize=12, fontweight='bold')

                    # Display grid and axis ticks for better visualization
                    # plt.xticks(range(0, img.width + 1, 100))
                    # plt.yticks(range(0, img.height + 1, 100))
                    # plt.grid(True, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
                    plt.show()

                except Exception as e:
                    self.debug_core.log_error(f"Error: {str(e)}")
            
            # Exit the loop if 'q' is pressed
            if plt.waitforbuttonpress(timeout=0.1):
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