import base64
import requests
import os

# OpenAI API Key
api_key = os.environ.get("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

initial_route = "images/unprocessed_positives/"
images = os.listdir(initial_route)
prompt = """
    ACT AS A FACE DETECTOR MACHINE.
    I WILL PASS TO YOU A FACE AND YOU WILL RESPONSE ME IF
    THE PERSON'S JAW IS SYMMETRIC, POLYGONAL, HAS BEEN TRAINED, IT IS MUSCLE.

    PLEASE RETURN ONLY '1' WHEN THE PERSON'S JAW IS SYMMETRIC.
    PLEASE RETURN ONLY '0' WHEN THE PERSON'S JAW IS NOT SYMMETRIC OR NOT POLYGONAL OR NOT TRAINED.

    FOLLOW ABOVE INSTRUCTIONS CORRECTLY.
"""

def payload(base64_image): return {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": f"{prompt}"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

for image in images:
    image_complete_path = initial_route+image
    b64 = encode_image(image_complete_path)
    body = payload(b64)
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)

        result = response.json()["choices"][0]["message"]["content"]

        print(" - The result is: ", result)
    except:
        pass
