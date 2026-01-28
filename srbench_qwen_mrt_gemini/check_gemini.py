#!/usr/bin/env python3
"""Vérifier que Gemini est correctement installé et fonctionne."""

import os
import sys
from PIL import Image
from io import BytesIO
import base64

print("=" * 60)
print("Vérification de l'installation Gemini")
print("=" * 60)

# 1. Vérifier l'API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("\n❌ GOOGLE_API_KEY ou GEMINI_API_KEY manquant")
    print("   Ex: export GOOGLE_API_KEY='...'")
    sys.exit(1)
print(f"\n✅ API key trouvée (longueur: {len(api_key)})")

# 2. Vérifier l'import
try:
    import google.genai as genai
    print("✅ google.genai importé")
except ImportError:
    print("❌ google.genai non installé")
    print("   Installez avec: pip install google-genai")
    sys.exit(1)

# 3. Nettoyer la clé
def _clean(s: str) -> str:
    return s.strip().replace("\u2028", "").replace("\u2029", "").replace("\r", "").replace("\n", "")

# 4. Créer un client
try:
    client = genai.Client(api_key=_clean(api_key))
    print("✅ Client Gemini créé")
except Exception as e:
    print(f"❌ Erreur création client: {e}")
    sys.exit(1)

# 5. Test simple avec une image factice
print("\n🧪 Test de génération avec image factice...")
img = Image.new("RGB", (100, 100), color="red")
buf = BytesIO()
img.save(buf, format="PNG")
image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

try:
    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    {"text": "What color is this image? Answer with ONE letter: R for red, B for blue, G for green."},
                ],
            }
        ],
        config={"max_output_tokens": 10, "temperature": 0.0},
    )
    
    print("✅ Réponse reçue")
    
    # Extraire le texte (méthode robuste)
    text = None
    
    # Méthode 1: response.parts[0].text
    try:
        if hasattr(response, "parts") and response.parts and len(response.parts) > 0:
            text = response.parts[0].text
            print(f"✅ Texte extrait via response.parts[0].text: '{text}'")
    except Exception as e:
        print(f"⚠️  response.parts échoué: {e}")
    
    # Méthode 2: response.text
    if not text:
        try:
            text = response.text
            if text:
                print(f"✅ Texte extrait via response.text: '{text}'")
        except Exception as e:
            print(f"⚠️  response.text échoué: {e}")
    
    # Méthode 3: candidates[0].content.parts[0].text
    if not text:
        try:
            if hasattr(response, "candidates") and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    content = candidate.content
                    if hasattr(content, "parts") and content.parts and len(content.parts) > 0:
                        part = content.parts[0]
                        if hasattr(part, "text") and part.text:
                            text = part.text
                            print(f"✅ Texte extrait via candidates[0].content.parts[0].text: '{text}'")
        except Exception as e:
            print(f"⚠️  candidates échoué: {e}")
    
    if text:
        print(f"\n✅ SUCCESS! Réponse Gemini: '{text.strip()}'")
        if text.strip().upper() in ["R", "RED"]:
            print("✅ Gemini voit bien l'image (réponse correcte pour image rouge)")
        else:
            print(f"⚠️  Réponse inattendue (attendu 'R' pour rouge, reçu '{text}')")
    else:
        print("\n❌ Aucun texte extrait de la réponse")
        print(f"   Type réponse: {type(response)}")
        print(f"   Attributs: {[x for x in dir(response) if not x.startswith('_')][:10]}")
        sys.exit(1)
        
except Exception as e:
    error_str = str(e)
    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
        print(f"\n⚠️  QUOTA ÉPUISÉ (429)")
        print("   Le quota gratuit Gemini est limité à 20 requêtes/jour par modèle.")
        print("   Options:")
        print("   1. Attendre le reset quotidien")
        print("   2. Upgrader vers un plan payant")
        print("   3. Utiliser Qwen/GPT-4o à la place")
        sys.exit(1)
    else:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 60)
print("✅ Gemini est correctement installé et fonctionne!")
print("=" * 60)
