import base64
import cv2
import json
import numpy as np
import os
import requests

def get_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Please enter your Gemini API key: ")
        os.environ["GEMINI_API_KEY"] = api_key
    return api_key

def resize_and_compress_image(image, max_width=1000):
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_size = (max_width, int(height * ratio))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buffer.tobytes()

def file_to_generative_part(image):
    return {
        "inlineData": {
            "data": base64.b64encode(image).decode("utf-8"),
            "mimeType": "image/jpeg"
        }
    }

def process_image_and_prompt(api_key, model_name, image, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    compressed_image = resize_and_compress_image(image)
    image_part = file_to_generative_part(compressed_image)
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    image_part
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

def extract_coordinates(text):
    import re
    regex = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
    matches = re.findall(regex, text)
    return [json.loads(match) for match in matches]

def display_image_with_bounding_boxes(image, coordinates):
    height, width = image.shape[:2]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, box in enumerate(coordinates):
        ymin, xmin, ymax, xmax = [int(coord * width / 1000) for coord in box]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[i % len(colors)], 2)

    # # Draw grid
    # for i in range(0, width, 100):
    #     cv2.line(image, (i, 0), (i, height), (0, 0, 255), 1)
    # for i in range(0, height, 100):
    #     cv2.line(image, (0, i), (width, i), (0, 0, 255), 1)

    # # Add labels
    # for i in range(0, width + 1, 100):
    #     cv2.putText(image, str(i), (i, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # for i in range(0, height + 1, 100):
    #     cv2.putText(image, str(i), (10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    api_key = get_api_key()
    
    image_path = input("Enter the path to your image: ")
    model_name = input("Enter the model name (default: gemini-pro-vision): ") or "gemini-pro-vision"
    prompt = input("Enter your prompt: ")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    print("Processing image and prompt...")
    response_text = process_image_and_prompt(api_key, model_name, image, prompt)
    print("API Response:")
    print(response_text)

    coordinates = extract_coordinates(response_text)
    if coordinates:
        print("Displaying image with bounding boxes...")
        display_image_with_bounding_boxes(image, coordinates)
    else:
        print("No bounding box coordinates found in the response.")

if __name__ == "__main__":
    main()