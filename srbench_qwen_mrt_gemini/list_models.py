#!/usr/bin/env python3
"""Liste les modèles Gemini disponibles avec l'API actuelle."""

import os

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY ou GEMINI_API_KEY manquant")
    exit(1)

def _clean(s: str) -> str:
    return s.strip().replace("\u2028", "").replace("\u2029", "").replace("\r", "").replace("\n", "")

print("=" * 60)
print("Liste des modèles Gemini disponibles")
print("=" * 60)

# Essayer la nouvelle API
try:
    import google.genai as genai
    print("\n📦 Utilisation de google.genai (nouvelle API)")
    client = genai.Client(api_key=_clean(api_key))
    
    try:
        models = client.models.list()
        print(f"\n✅ {len(models)} modèles trouvés:\n")
        for model in models:
            name = getattr(model, "name", str(model))
            print(f"  - {name}")
    except Exception as e:
        print(f"❌ Erreur lors de la liste: {e}")
        print("\nEssai avec l'ancienne API...")
        raise
except Exception:
    # Fallback vers l'ancienne API
    try:
        import google.generativeai as genai
        print("\n📦 Utilisation de google.generativeai (ancienne API)")
        genai.configure(api_key=_clean(api_key))
        
        models = genai.list_models()
        print(f"\n✅ {len(list(models))} modèles trouvés:\n")
        for model in models:
            name = getattr(model, "name", str(model))
            # Filtrer seulement les modèles qui supportent generateContent
            if "generateContent" in getattr(model, "supported_generation_methods", []):
                print(f"  - {name}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

print("\n" + "=" * 60)
