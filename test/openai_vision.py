import argparse
import base64
import io
import json
import os
from PIL import Image
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key

def resize_and_compress_image(file_path, max_width=1000):
    with Image.open(file_path) as img:
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_and_prompt(api_key, model_name, image_path, prompt):
    chat_model = ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_key=api_key
    )

    base64_image = encode_image(image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"

    result = chat_model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )
        ]
    )

    return result.content

def extract_coordinates(text):
    import re
    regex = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
    matches = re.findall(regex, text)
    return [json.loads(match) for match in matches]

def display_image_with_bounding_boxes(image_path, coordinates):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i, box in enumerate(coordinates):
        ymin, xmin, ymax, xmax = [coord / 1000 for coord in box]
        width = xmax - xmin
        height = ymax - ymin
        rect = Rectangle((xmin * img.width, ymin * img.height), width * img.width, height * img.height,
                         linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none')
        ax.add_patch(rect)

    plt.xticks(range(0, img.width + 1, 100))
    plt.yticks(range(0, img.height + 1, 100))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process image and prompt using OpenAI API")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("prompt", help="Prompt for the OpenAI API")
    parser.add_argument("--model", default="gpt-4o", help="Model name (default: gpt-4o-mini)")
    args = parser.parse_args()

    api_key = get_api_key()
    
    print("Processing image and prompt...")
    response_text = process_image_and_prompt(api_key, args.model, args.image_path, args.prompt)
    print("API Response:")
    print(response_text)

    coordinates = extract_coordinates(response_text)
    if coordinates:
        print("Displaying image with bounding boxes...")
        display_image_with_bounding_boxes(args.image_path, coordinates)
    else:
        print("No bounding box coordinates found in the response.")

if __name__ == "__main__":
    main()