from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


CHOICES = ["A", "B", "C", "D"]


@dataclass
class Example:
    idx: int
    split: str
    question: str
    answer: str
    image: Image.Image


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_choice(text: str) -> Optional[str]:
    """
    Parse robuste pour extraire A/B/C/D de la réponse Gemini.
    Utilise regex \b([ABCD])\b pour trouver la lettre n'importe où dans le texte.
    """
    if not text:
        return None
    
    # Nettoyer et mettre en majuscules
    t = text.strip().upper()
    
    # Méthode 1: \b([ABCD])\b (le plus robuste - trouve la lettre entourée de word boundaries)
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    
    # Méthode 2: (A) ou [A] ou "A"
    for pattern in [r"\(([ABCD])\)", r"\[([ABCD])\]", r'"([ABCD])"', r"'([ABCD])'"]:
        m = re.search(pattern, t)
        if m:
            return m.group(1)
    
    # Méthode 3: Lettre isolée au début ou à la fin
    m = re.match(r"^\s*([ABCD])\s*[.\-:)]?\s*$", t)
    if m:
        return m.group(1)
    
    m = re.search(r"([ABCD])\s*$", t)
    if m:
        return m.group(1)
    
    return None


def _build_prompt(question: str) -> str:
    # Prompt très strict pour forcer une réponse d'une seule lettre
    return (
        f"{question}\n\n"
        "IMPORTANT: Réponds avec EXACTEMENT UNE SEULE LETTRE: A, B, C, ou D.\n"
        "Ne mets rien d'autre. Juste la lettre."
    )


def iter_examples(
    ds,
    splits: List[str],
    max_samples: int,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Iterable[Example]:
    examples: List[Example] = []
    for i, row in enumerate(ds):
        sp = row.get("split")
        if sp not in splits:
            continue
        img = row.get("image")
        if isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = Image.open(img["path"]).convert("RGB")  # type: ignore[index]
        examples.append(
            Example(
                idx=i,
                split=str(sp),
                question=str(row.get("question")),
                answer=str(row.get("answer")).strip().upper(),
                image=pil_img.convert("RGB"),
            )
        )

    if shuffle:
        if seed is None:
            seed = int(time.time() * 1_000_000) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        random.shuffle(examples)

    count = 0
    for ex in examples:
        yield ex
        count += 1
        if max_samples != -1 and count >= max_samples:
            return


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, List[bool]] = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(bool(r["correct"]))
    return {
        "n": len(rows),
        "accuracy": float(np.mean([r["correct"] for r in rows])) if rows else 0.0,
        "by_split": {sp: {"n": len(v), "accuracy": float(np.mean(v)) if v else 0.0} for sp, v in by_split.items()},
    }


def call_gemini(image: Image.Image, question: str, model_name: str, max_output_tokens: int, debug_response: bool = False) -> str:
    # Utiliser la nouvelle API google.genai (l'ancienne google.generativeai ne fonctionne plus)
    try:
        import google.genai as genai
        use_new_api = True
    except ImportError:
        # Fallback vers l'ancienne API si google.genai n'est pas installé
        import google.generativeai as genai
        use_new_api = False

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY ou GEMINI_API_KEY manquant. Ex: export GOOGLE_API_KEY='...'.")

    # Nettoyer la clé API
    def _clean(s: str) -> str:
        return s.strip().replace("\u2028", "").replace("\u2029", "").replace("\r", "").replace("\n", "")

    prompt = _build_prompt(question)

    # Convertir l'image en bytes
    buf = BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    try:
        if use_new_api:
            # Nouvelle API google.genai
            client = genai.Client(api_key=_clean(api_key))
            
            # Mapping vers les noms corrects de l'API
            model_map = {
                "gemini-1.5-flash": "gemini-flash-latest",
                "gemini-1.5-pro": "gemini-pro-latest",
                "gemini-pro": "gemini-pro-latest",
                "gemini-2.5-flash": "gemini-2.5-flash",
                "gemini-2.0-flash": "gemini-2.0-flash",
            }
            
            # Normaliser le nom (enlever "models/" si présent)
            clean_name = model_name.replace("models/", "").strip()
            api_model_name = model_map.get(clean_name, clean_name)
            
            # Ajouter "models/" si nécessaire
            if not api_model_name.startswith("models/"):
                api_model_name = f"models/{api_model_name}"
            
            # Retry avec backoff pour les erreurs 429 (quota)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=api_model_name,
                        contents=[
                            {
                                "role": "user",
                                "parts": [
                                    {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("ascii")}},
                                    {"text": prompt},
                                ],
                            }
                        ],
                        config={"max_output_tokens": max_output_tokens, "temperature": 0.0},
                    )
                    break  # Succès, sortir de la boucle
                except Exception as e:
                    error_str = str(e)
                    # Si c'est une erreur 429 (quota), attendre et réessayer
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                            if debug_response:
                                print(f"[RETRY] Quota error, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                            time.sleep(wait_time)
                            continue
                        else:
                            return f"ERROR: Quota exceeded (429). Free tier limit: 20 requests/day per model. Please wait or upgrade to paid plan. Original error: {error_str[:200]}"
                    # Autres erreurs: propager immédiatement
                    raise
            
            # Debug: afficher la structure complète si demandé
            if debug_response:
                import json
                try:
                    finish_reason = None
                    if hasattr(response, "candidates") and response.candidates and len(response.candidates) > 0:
                        finish_reason = getattr(response.candidates[0], "finish_reason", None)
                    
                    debug_info = {
                        "type": str(type(response)),
                        "has_text": hasattr(response, "text"),
                        "text_value": getattr(response, "text", None),
                        "has_parts": hasattr(response, "parts"),
                        "parts_len": len(response.parts) if hasattr(response, "parts") and response.parts else 0,
                        "has_candidates": hasattr(response, "candidates"),
                        "candidates_len": len(response.candidates) if hasattr(response, "candidates") and response.candidates else 0,
                        "finish_reason": str(finish_reason) if finish_reason else None,
                    }
                    print(f"[DEBUG RESPONSE] {json.dumps(debug_info, indent=2, default=str)}")
                except Exception as e:
                    print(f"[DEBUG RESPONSE] Error inspecting: {e}")
            
            # Fonction robuste pour extraire le texte (comme suggéré par l'utilisateur)
            def extract_text(resp) -> Optional[str]:
                """Extrait le texte de la réponse Gemini de manière robuste."""
                if resp is None:
                    return None
                
                # 1) response.text (propriété calculée, peut être None)
                try:
                    if hasattr(resp, "text") and resp.text:
                        return str(resp.text).strip()
                except (AttributeError, TypeError):
                    pass
                
                # 2) response.parts[0].text (le plus fiable d'après check_gemini.py)
                try:
                    if hasattr(resp, "parts") and resp.parts and len(resp.parts) > 0:
                        part = resp.parts[0]
                        if hasattr(part, "text") and part.text:
                            return str(part.text).strip()
                except (TypeError, AttributeError, IndexError):
                    pass
                
                # 3) candidates[0].content.parts (parcourir tous les candidates et parts)
                try:
                    candidates = getattr(resp, "candidates", None) or []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if content:
                            parts = getattr(content, "parts", None) or []
                            for p in parts:
                                t = getattr(p, "text", None)
                                if t:
                                    return str(t).strip()
                except (TypeError, AttributeError, IndexError):
                    pass
                
                return None
            
            # Extraire le texte
            text = extract_text(response)
            
            if text:
                return text
            else:
                # Debug: afficher plus d'info pour comprendre pourquoi (finish_reason, etc.)
                debug_parts = ["ERROR: No text found"]
                try:
                    debug_parts.append(f"type={type(response).__name__}")
                    debug_parts.append(f"has_parts={hasattr(response, 'parts')}")
                    if hasattr(response, "parts") and response.parts:
                        debug_parts.append(f"parts_len={len(response.parts)}")
                    debug_parts.append(f"has_candidates={hasattr(response, 'candidates')}")
                    if hasattr(response, "candidates") and response.candidates and len(response.candidates) > 0:
                        cand = response.candidates[0]
                        finish_reason = getattr(cand, "finish_reason", None)
                        debug_parts.append(f"finish_reason={finish_reason}")
                        # Si finish_reason est MAX_TOKENS, c'est peut-être que max_output_tokens est trop petit
                        if finish_reason and "MAX_TOKENS" in str(finish_reason):
                            debug_parts.append("(MAX_TOKENS - peut-être max_output_tokens trop petit?)")
                except Exception as e:
                    debug_parts.append(f"debug_error={str(e)}")
                return ". ".join(debug_parts)
        else:
            # Ancienne API (fallback, probablement ne fonctionnera pas)
            genai.configure(api_key=_clean(api_key))
            model = genai.GenerativeModel(model_name.replace("models/", ""))
            response = model.generate_content(
                [image, prompt],
                generation_config={"max_output_tokens": max_output_tokens, "temperature": 0.0},
            )
            return (response.text or "").strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gemini-flash-latest", help="gemini-flash-latest, gemini-pro-latest, gemini-2.5-flash, gemini-2.0-flash, etc. (ou gemini-1.5-flash sera mappé vers gemini-flash-latest)")
    ap.add_argument("--dataset_name", default="stogian/srbench")
    ap.add_argument("--dataset_split", default="test")
    ap.add_argument("--splits", nargs="+", default=["mrt_easy", "mrt_hard"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_output_tokens", type=int, default=32, help="Augmenter si finish_reason=MAX_TOKENS (réponse tronquée)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--debug", action="store_true", help="Afficher les 5 premières réponses brutes pour debug")
    ap.add_argument("--debug_response", action="store_true", help="Afficher la structure complète de la réponse Gemini (très verbeux)")
    ap.add_argument("--verbose", action="store_true", help="Afficher toutes les réponses brutes (question, réponse attendue, réponse Gemini, prédiction)")
    args = ap.parse_args()

    if args.out_dir is None:
        safe_model = args.model_name.replace("/", "_")
        args.out_dir = f"runs/{safe_model}_{_now_id()}"
    else:
        args.out_dir = f"{args.out_dir.rstrip('/')}_{_now_id()}"

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []
    example_count = 0

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        iterator = iter_examples(ds, args.splits, args.max_samples, args.shuffle, args.seed)
        if args.verbose:
            iterator = list(iterator)  # Convertir en liste pour afficher sans tqdm
        else:
            iterator = tqdm(iterator, desc="Evaluating")
        
        for ex in iterator:
            raw = call_gemini(ex.image, ex.question, args.model_name, args.max_output_tokens, debug_response=args.debug_response)
            
            pred = _parse_choice(raw)
            pred = pred if pred in CHOICES else None
            correct = (pred == ex.answer)
            
            example_count += 1
            
            # Afficher les réponses (debug ou verbose)
            if args.verbose or (args.debug and example_count <= 5):
                print(f"\n[Exemple {example_count}]")
                print(f"  Question: {ex.question[:150]}...")
                print(f"  Réponse attendue: {ex.answer}")
                print(f"  Réponse brute Gemini: {raw}")
                print(f"  Prédiction parsée: {pred}")
                print(f"  Correct: {correct}")
                
                # Si verbose et que raw contient "ERROR", afficher plus de détails
                if args.verbose and "ERROR" in raw:
                    print(f"  ⚠️  Erreur détectée - vérifier la réponse Gemini")

            row = {"idx": ex.idx, "split": ex.split, "answer": ex.answer, "pred": pred, "correct": bool(correct), "raw": raw}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            detailed.append(
                {
                    "idx": ex.idx,
                    "split": ex.split,
                    "question": ex.question,
                    "ground_truth": ex.answer,
                    "model_prediction": pred,
                    "model_raw_output": raw,
                    "correct": bool(correct),
                    "image_size": list(ex.image.size),
                }
            )

    (out_dir / "detailed_results.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "dataset_split": args.dataset_split,
                "splits": args.splits,
                "max_samples": args.max_samples,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "total_examples": len(detailed),
                "results": detailed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = compute_metrics(rows)
    metrics.update(
        {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "splits": args.splits,
            "max_samples": args.max_samples,
            "shuffle": args.shuffle,
            "seed": args.seed,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
