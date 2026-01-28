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
    
    # Si c'est le code d'erreur MAX_TOKENS, retourner None
    if text.strip().upper() == "MAX_TOKENS":
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
    # Prompt ultra-strict pour forcer EXACTEMENT une seule lettre
    # Format minimal pour éviter MAX_TOKENS
    # IMPORTANT: Ne pas mettre de point final ou saut de ligne - stop_sequences s'en chargera
    return (
        f"{question}\n\n"
        "Answer with exactly ONE character: A, B, or C"
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
    # Compter seulement les exemples avec une vraie réponse (pas les erreurs)
    valid_rows = [r for r in rows if r.get("pred") is not None]  # Seulement ceux avec une prédiction valide
    error_count = len(rows) - len(valid_rows)
    
    by_split: Dict[str, List[bool]] = {}
    by_split_errors: Dict[str, int] = {}
    for r in rows:
        split = r["split"]
        if r.get("pred") is not None:
            # Seulement compter les réponses valides
            by_split.setdefault(split, []).append(bool(r["correct"]))
        else:
            # Compter les erreurs séparément
            by_split_errors[split] = by_split_errors.get(split, 0) + 1
    
    return {
        "n": len(rows),
        "n_valid": len(valid_rows),  # Nombre de réponses valides
        "n_errors": error_count,  # Nombre d'erreurs (MAX_TOKENS, etc.)
        "accuracy": float(np.mean([r["correct"] for r in valid_rows])) if valid_rows else 0.0,  # Accuracy seulement sur les réponses valides
        "by_split": {
            sp: {
                "n": len(v),
                "n_errors": by_split_errors.get(sp, 0),
                "accuracy": float(np.mean(v)) if v else 0.0
            }
            for sp, v in by_split.items()
        },
    }


def call_gemini(image: Image.Image, question: str, model_name: str, max_output_tokens: int, debug_response: bool = False) -> str:
    # Essayer d'abord l'ancienne API google.generativeai (plus stable pour max_output_tokens)
    # Puis fallback vers la nouvelle API google.genai si nécessaire
    use_old_api = False
    use_new_api = False
    
    try:
        import google.generativeai as genai_old
        use_old_api = True
        if debug_response:
            print("[DEBUG] Ancienne API google.generativeai disponible")
    except ImportError:
        pass
    
    if not use_old_api:
        try:
            import google.genai as genai
            use_new_api = True
            if debug_response:
                print("[DEBUG] Nouvelle API google.genai disponible")
        except ImportError:
            raise RuntimeError("Ni google.generativeai ni google.genai n'est installé. Installez avec: pip install google-generativeai ou pip install google-genai")

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
        if use_old_api:
            # Ancienne API google.generativeai (plus stable pour max_output_tokens)
            genai_old.configure(api_key=_clean(api_key))
            model = genai_old.GenerativeModel(model_name.replace("models/", ""))
            
            # Utiliser generation_config - stop_sequences limité à 5 max
            # Simplifier : juste arrêter après saut de ligne ou point
            generation_config = {
                "max_output_tokens": max_output_tokens,
                "temperature": 0.0,
                "stop_sequences": ["\n", "."],  # Maximum 5, on en met 2 seulement
            }
            
            if debug_response:
                print(f"[DEBUG] Ancienne API: max_output_tokens={max_output_tokens}, model={model_name}")
            
            response = model.generate_content(
                [image, prompt],
                generation_config=generation_config,
            )
            
            # Extraire le texte
            text = (response.text or "").strip()
            if text:
                return text
            else:
                # Vérifier finish_reason
                if hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    finish_reason = getattr(cand, "finish_reason", None)
                    if finish_reason and "MAX_TOKENS" in str(finish_reason):
                        return "MAX_TOKENS"
                return "ERROR: No text found"
        
        elif use_new_api:
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
                    # Utiliser types.GenerateContentConfig() au lieu d'un dict simple
                    # C'est le format correct pour l'API google.genai
                    # Ajouter stop_sequences pour forcer l'arrêt après A/B/C
                    try:
                        from google.genai import types
                        # stop_sequences limité à 5 max - simplifier
                        config = types.GenerateContentConfig(
                            max_output_tokens=max_output_tokens,
                            temperature=0.0,
                            stop_sequences=["\n", "."],  # Maximum 5, on en met 2 seulement
                        )
                    except ImportError:
                        # Fallback vers dict si types n'est pas disponible
                        config = {
                            "max_output_tokens": max_output_tokens,
                            "temperature": 0.0,
                            "stop_sequences": ["\n", "."],
                        }
                    
                    # Log pour vérifier que max_output_tokens est bien passé
                    if debug_response:
                        print(f"[DEBUG] Appel API avec max_output_tokens={max_output_tokens}, model={api_model_name}")
                        print(f"[DEBUG] Config type: {type(config)}, max_output_tokens dans config: {getattr(config, 'max_output_tokens', config.get('max_output_tokens', 'N/A'))}")
                    
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
                        config=config,
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
                """Extrait le texte de la réponse Gemini de manière robuste.
                Même si finish_reason=MAX_TOKENS, on essaie de récupérer le texte partiel.
                Ordre de priorité basé sur check_gemini.py qui fonctionne.
                """
                if resp is None:
                    return None
                
                # Méthode 1: response.parts[0].text (le plus fiable d'après check_gemini.py)
                try:
                    if hasattr(resp, "parts") and resp.parts and len(resp.parts) > 0:
                        part = resp.parts[0]
                        if hasattr(part, "text") and part.text:
                            text = str(part.text).strip()
                            if text:  # Même si tronqué, on peut extraire la lettre
                                return text
                except (TypeError, AttributeError, IndexError):
                    pass
                
                # Méthode 2: response.text (propriété calculée)
                # IMPORTANT: Même si text_value=null dans le debug, essaie quand même
                try:
                    if hasattr(resp, "text"):
                        text_val = resp.text
                        if text_val:
                            text = str(text_val).strip()
                            if text:
                                return text
                except (AttributeError, TypeError):
                    pass
                
                # Méthode 3: candidates[0].content.parts[0].text (parcourir tous les candidates et parts)
                # IMPORTANT: Même si finish_reason=MAX_TOKENS et parts_len=0, vérifier quand même
                try:
                    if hasattr(resp, "candidates") and resp.candidates and len(resp.candidates) > 0:
                        candidate = resp.candidates[0]
                        if hasattr(candidate, "content") and candidate.content:
                            content = candidate.content
                            if hasattr(content, "parts") and content.parts and len(content.parts) > 0:
                                part = content.parts[0]
                                if hasattr(part, "text") and part.text:
                                    text = str(part.text).strip()
                                    if text:  # Même texte tronqué, on peut extraire la lettre
                                        return text
                except (TypeError, AttributeError, IndexError):
                    pass
                
                # Méthode 4: Parcourir TOUS les candidates et TOUS les parts (dernier recours)
                try:
                    candidates = getattr(resp, "candidates", None) or []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if content:
                            parts = getattr(content, "parts", None) or []
                            for p in parts:
                                t = getattr(p, "text", None)
                                if t:
                                    text = str(t).strip()
                                    if text:  # Même texte tronqué, on peut extraire la lettre
                                        return text
                except (TypeError, AttributeError, IndexError):
                    pass
                
                return None
            
            # Extraire le texte (cette fonction essaie déjà toutes les méthodes)
            text = extract_text(response)
            
            if text:
                return text
            
            # Si on arrive ici, aucune méthode n'a fonctionné
            # Vérifier si finish_reason=MAX_TOKENS - dans ce cas, il n'y a vraiment pas de texte
            # (le debug montre parts_len=0 et text_value=null)
            try:
                if hasattr(response, "candidates") and response.candidates and len(response.candidates) > 0:
                    cand = response.candidates[0]
                    finish_reason = getattr(cand, "finish_reason", None)
                    
                    if finish_reason and "MAX_TOKENS" in str(finish_reason):
                        # Si MAX_TOKENS et parts_len=0, il n'y a vraiment pas de texte généré
                        # Cela signifie que Gemini a généré des tokens "cachés" (thinking) qui ont consommé la limite
                        # avant même de générer la première lettre
                        # Dans ce cas, on ne peut rien faire - il faut augmenter max_output_tokens
                        # ou utiliser un prompt encore plus strict
                        return "MAX_TOKENS"  # Retourner un code d'erreur spécial pour le parser
            except Exception:
                pass
            
            # Si vraiment rien n'a fonctionné, faire un diagnostic approfondi
            debug_parts = ["ERROR: No text found"]
            try:
                debug_parts.append(f"type={type(response).__name__}")
                debug_parts.append(f"has_parts={hasattr(response, 'parts')}")
                
                # Diagnostic approfondi de response.parts
                if hasattr(response, "parts") and response.parts:
                    debug_parts.append(f"parts_len={len(response.parts)}")
                    for i, part in enumerate(response.parts[:3]):  # Limiter à 3
                        part_type = type(part).__name__
                        has_text = hasattr(part, "text")
                        text_val = getattr(part, "text", None)
                        debug_parts.append(f"parts[{i}]: type={part_type}, has_text={has_text}, text={repr(str(text_val)[:50]) if text_val else None}")
                
                debug_parts.append(f"has_candidates={hasattr(response, 'candidates')}")
                if hasattr(response, "candidates") and response.candidates and len(response.candidates) > 0:
                    cand = response.candidates[0]
                    finish_reason = getattr(cand, "finish_reason", None)
                    debug_parts.append(f"finish_reason={finish_reason}")
                    
                    # Diagnostic approfondi de candidates[0].content.parts
                    if finish_reason and "MAX_TOKENS" in str(finish_reason):
                        debug_parts.append("(MAX_TOKENS)")
                        content = getattr(cand, "content", None)
                        if content:
                            content_type = type(content).__name__
                            debug_parts.append(f"content_type={content_type}")
                            parts = getattr(content, "parts", None)
                            if parts:
                                debug_parts.append(f"content.parts_len={len(parts)}")
                                for i, p in enumerate(parts[:3]):  # Limiter à 3
                                    p_type = type(p).__name__
                                    p_text = getattr(p, "text", None)
                                    debug_parts.append(f"content.parts[{i}]: type={p_type}, text={repr(str(p_text)[:50]) if p_text else None}")
                            else:
                                debug_parts.append("content.parts=None ou vide")
                        else:
                            debug_parts.append("candidate.content=None")
                        
                        # Dernière tentative : essayer d'accéder directement à tous les attributs
                        if debug_response:
                            print(f"[DEBUG MAX_TOKENS] Structure complète:")
                            print(f"  response type: {type(response)}")
                            print(f"  response dir: {[x for x in dir(response) if not x.startswith('_')][:10]}")
                            if hasattr(response, "candidates") and response.candidates:
                                cand = response.candidates[0]
                                print(f"  candidate type: {type(cand)}")
                                print(f"  candidate dir: {[x for x in dir(cand) if not x.startswith('_')][:10]}")
                                if hasattr(cand, "content"):
                                    content = cand.content
                                    print(f"  content type: {type(content)}")
                                    print(f"  content dir: {[x for x in dir(content) if not x.startswith('_')][:10]}")
            except Exception as e:
                debug_parts.append(f"debug_error={str(e)}")
                if debug_response:
                    import traceback
                    traceback.print_exc()
            
            return ". ".join(debug_parts)
        else:
            # Si ni l'ancienne ni la nouvelle API n'est disponible, erreur
            raise RuntimeError("Aucune API Gemini disponible. Installez google-generativeai ou google-genai.")
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
    ap.add_argument("--max_output_tokens", type=int, default=128, help="Tokens max pour la réponse. Par défaut: 128. Augmenter si MAX_TOKENS persiste.")
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
                
                # Afficher le mapping des options depuis la question
                if "Available options:" in ex.question:
                    import re
                    options_match = re.search(r"Available options: (.+)", ex.question)
                    if options_match:
                        options_text = options_match.group(1)
                        print(f"  Options dans question: {options_text[:100]}")
                
                # Si verbose et que raw contient "ERROR" ou "MAX_TOKENS", afficher plus de détails
                if args.verbose and ("ERROR" in raw or "MAX_TOKENS" in raw):
                    print(f"  ⚠️  Erreur détectée - vérifier la réponse Gemini")
                    print(f"  max_output_tokens utilisé: {args.max_output_tokens}")

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
