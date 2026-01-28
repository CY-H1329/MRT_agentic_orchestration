#!/usr/bin/env python3
"""Test simple pour voir la structure exacte de la réponse Gemini."""

import os
from PIL import Image
import google.genai as genai

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY manquant")
    exit(1)

def _clean(s: str) -> str:
    return s.strip().replace("\u2028", "").replace("\u2029", "").replace("\r", "").replace("\n", "")

client = genai.Client(api_key=_clean(api_key))

# Image factice
img = Image.new("RGB", (100, 100), color="red")

# Convertir en base64
from io import BytesIO
import base64
buf = BytesIO()
img.save(buf, format="PNG")
image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

print("Test avec gemini-flash-latest...")
try:
    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    {"text": "What color is this? Answer with one letter: R for red, B for blue."},
                ],
            }
        ],
        config={"max_output_tokens": 10, "temperature": 0.0},
    )
    
    print(f"\n✅ Réponse reçue")
    print(f"Type: {type(response)}")
    print(f"hasattr text: {hasattr(response, 'text')}")
    print(f"response.text: {response.text}")
    print(f"response.text type: {type(response.text)}")
    
    print(f"\nhasattr candidates: {hasattr(response, 'candidates')}")
    print(f"candidates type: {type(response.candidates)}")
    print(f"candidates length: {len(response.candidates) if hasattr(response, 'candidates') and response.candidates else 'N/A'}")
    
    if hasattr(response, "candidates") and response.candidates and len(response.candidates) > 0:
        cand = response.candidates[0]
        print(f"\ncandidate[0] type: {type(cand)}")
        print(f"candidate[0] has content: {hasattr(cand, 'content')}")
        if hasattr(cand, "content"):
            content = cand.content
            print(f"content type: {type(content)}")
            print(f"content has parts: {hasattr(content, 'parts')}")
            if hasattr(content, "parts") and content.parts:
                print(f"parts length: {len(content.parts)}")
                if len(content.parts) > 0:
                    part = content.parts[0]
                    print(f"part[0] type: {type(part)}")
                    print(f"part[0] has text: {hasattr(part, 'text')}")
                    if hasattr(part, "text"):
                        print(f"part[0].text: {part.text}")
                        print(f"✅ TEXTE TROUVÉ: {part.text}")
    
    print(f"\nhasattr parts: {hasattr(response, 'parts')}")
    if hasattr(response, "parts"):
        print(f"response.parts: {response.parts}")
        if response.parts:
            print(f"parts[0]: {response.parts[0] if len(response.parts) > 0 else 'N/A'}")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
