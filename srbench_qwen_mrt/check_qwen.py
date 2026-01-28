from __future__ import annotations

import argparse
import sys

import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description="Vérifier l'installation de Qwen2.5-VL")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    print("=" * 60)
    print("Vérification de l'installation Qwen2.5-VL")
    print("=" * 60)

    # 1. Vérifier transformers
    try:
        import transformers

        print(f"\n✅ transformers: {transformers.__version__}")
        if transformers.__version__.startswith("5."):
            print("   ⚠️  Version 5.x détectée. Qwen2.5-VL peut nécessiter la version main de GitHub.")
            print("   Si ça plante, essayez: pip install 'git+https://github.com/huggingface/transformers.git'")
    except ImportError:
        print("\n❌ transformers non installé")
        sys.exit(1)

    # 2. Vérifier qwen-vl-utils
    process_vision_info = None
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        print("✅ qwen-vl-utils: installé")
    except ImportError as e:
        # Essayer aussi l'import alternatif
        try:
            import qwen_vl_utils  # type: ignore
            process_vision_info = qwen_vl_utils.process_vision_info
            print("✅ qwen-vl-utils: installé (import alternatif)")
        except (ImportError, AttributeError):
            print("❌ qwen-vl-utils non installé ou import échoué")
            print(f"   Erreur: {e}")
            print("   Vérifiez avec: python -c 'from qwen_vl_utils import process_vision_info'")
            print("   Installez avec: pip install qwen-vl-utils")
            sys.exit(1)

    # 3. Vérifier torch
    print(f"\n✅ torch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  CUDA non disponible (CPU uniquement)")

    # 4. Essayer de charger le modèle
    print(f"\n📦 Chargement du modèle: {args.model_name}")
    print("   (cela peut prendre quelques minutes...)")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if args.device == "auto" else None,
            trust_remote_code=True,
        )
        print("✅ Modèle chargé")

        if args.device != "auto":
            model = model.to(args.device)

        # Vérifier .generate()
        if hasattr(model, "generate"):
            print("✅ Méthode .generate() disponible")
        else:
            print("❌ Méthode .generate() manquante")
            print("   Le modèle chargé n'est pas de type ForConditionalGeneration")
            sys.exit(1)

    except ImportError as e:
        print(f"❌ Impossible d'importer Qwen2_5_VLForConditionalGeneration")
        print(f"   Erreur: {e}")
        print("\n   Solution: installez transformers depuis GitHub:")
        print("   pip install -U 'git+https://github.com/huggingface/transformers.git' accelerate")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        sys.exit(1)

    # 5. Charger le processor
    print("\n📦 Chargement du processor...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
        print("✅ Processor chargé")
    except Exception as e:
        print(f"❌ Erreur processor: {e}")
        sys.exit(1)

    # 6. Test simple avec une image factice
    print("\n🧪 Test de génération (image factice)...")
    try:
        # Image factice
        dummy_img = Image.new("RGB", (100, 100), color="red")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text", "text": "What color is this image? Answer with one word."},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Déplacer sur le bon device
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        print("   Génération en cours...")
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        # Décoder
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = generated_ids[0][in_len:]
        output_text = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]

        print(f"✅ Génération réussie!")
        print(f"   Réponse: {output_text.strip()[:100]}")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Tous les tests sont passés! Qwen2.5-VL est prêt.")
    print("=" * 60)


if __name__ == "__main__":
    main()
