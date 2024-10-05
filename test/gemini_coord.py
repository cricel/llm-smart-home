import json
import cv2
import base64

class Coord:
    def file_to_generative_part(file_path):
        with open(file_path, "rb") as image_file:
            return {
                "inlineData": {
                    "data": base64.b64encode(image_file.read()).decode("utf-8"),
                    "mimeType": "image/jpeg"
                }
            }
        
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
    
    def display_image_with_bounding_boxes(img, coordinates):

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        for i, box in enumerate(coordinates):
            ymin, xmin, ymax, xmax = [int(coord / 1000 * img.shape[0 if j % 2 == 0 else 1]) for j, coord in enumerate(box)]
            color = colors[i % len(colors)]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        
        return img
        # cv2.imshow('Image with Bounding Boxes', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
