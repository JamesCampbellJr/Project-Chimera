# /core/perception.py

import mss
import logging
import ollama
from PIL import Image
import io
import json

import config

class Perception:
    """
    Handles capturing the screen and analyzing its content using a VLM.
    """
    def __init__(self):
        self.sct = mss.mss()
        logging.info("Perception module initialized.")

    def capture_screen(self) -> Image.Image:
        """
        Captures the primary monitor and returns it as a PIL Image.
        """
        with self.sct as sct:
            monitor = sct.monitors[1]  # 0 is all monitors, 1 is the primary
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return img

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Converts a PIL image to bytes."""
        with io.BytesIO() as bio:
            image.save(bio, format='PNG')
            return bio.getvalue()

    async def analyze_screen(self, image: Image.Image, user_prompt: str) -> dict:
        """
        Analyzes the screen image with a VLM (llava) based on the user's prompt.
        Aims to identify key elements and return a structured JSON object.
        """
        logging.info("Analyzing screen content with VLM...")
        image_bytes = self._image_to_bytes(image)
        
        # A carefully crafted prompt for the VLM to get structured output
        vlm_prompt = f"""
        User's command: "{user_prompt}"
        Analyze the attached screenshot of a computer screen. Based on the user's command, identify interactive elements.
        Your response MUST be a single JSON object. Do not include any other text or explanations.
        The JSON object should have two keys: "description" and "elements".
        - "description": A brief summary of what's on the screen (e.g., "A web browser showing the Google search page.").
        - "elements": A list of dictionaries, where each dictionary represents an interactive element relevant to the user's command.
        Each element dictionary should have these keys: "id" (a unique integer), "type" (e.g., "button", "input_field", "link"), "label" (the text on the element), and "description" (a short description of its purpose).
        If you cannot identify specific elements or coordinates, provide an empty list for "elements" but still provide the screen description.

        Example response:
        {{
          "description": "A file explorer window is open, showing the 'Documents' folder.",
          "elements": [
            {{ "id": 1, "type": "input_field", "label": "Search Documents", "description": "The search bar at the top right." }},
            {{ "id": 2, "type": "icon", "label": "MyProject.docx", "description": "A Word document file." }}
          ]
        }}
        """
        
        try:
            response = await ollama.AsyncClient().chat(
                model=config.VISION_MODEL,
                messages=[{
                    'role': 'user',
                    'content': vlm_prompt,
                    'images': [image_bytes]
                }],
                format='json'
            )
            # The 'format=json' parameter in ollama is a powerful hint
            # that often encourages the model to return valid JSON.
            # We still need to parse it defensively.
            response_text = response['message']['content']
            logging.debug(f"Raw VLM response: {response_text}")
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from VLM response. The model may have returned plain text.")
            logging.error(f"VLM raw output: {response_text}")
            # Fallback: Try to use the raw text as a description
            return {"description": response_text, "elements": []}
        except Exception as e:
            logging.error(f"An error occurred during screen analysis: {e}")
            return {"description": "Error during screen analysis.", "elements": []}