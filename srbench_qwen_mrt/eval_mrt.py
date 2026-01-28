from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer


class _FallbackProcessor:
    """
    Fallback simple processor for older transformers where AutoProcessor may crash
    (notably in video_processing_auto.py).

    It combines an image processor + tokenizer into a dict compatible with model.generate.
    """

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, text: str, images: Image.Image, return_tensors: str = "pt") -> Dict[str, Any]:
        tok = self.tokenizer(text, return_tensors=return_tensors)
        img = self.image_processor(images=images, return_tensors=return_tensors)
        # Common key: pixel_values
        return {**tok, **img}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


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
    Extract a single choice in {A,B,C,D} from model output.
    Designed to be conservative (avoid false positives).
    """
    if not text:
        return None
    t = text.strip().upper()

    # Common formats: "A", "Answer: A", "The correct option is B.", "(C)"
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)

    m = re.search(r"\(([ABCD])\)", t)
    if m:
        return m.group(1)

    # As a fallback, look for leading letter
    m = re.match(r"^\s*([ABCD])\s*(?:[.\-:)]|\s|$)", t)
    if m:
        return m.group(1)

    return None


def _build_prompt(question: str) -> str:
    # Strongly constrain output.
    return (
        f"{question}\n\n"
        "Réponds avec UNE SEULE LETTRE parmi {A,B,C,D}.\n"
        "N'ajoute aucun autre texte."
    )


def iter_examples(ds, splits: List[str], max_samples: int) -> Iterable[Example]:
    kept = 0
    for i, row in enumerate(ds):
        sp = row.get("split")
        if sp not in splits:
            continue
        img = row.get("image")
        if isinstance(img, Image.Image):
            pil_img = img
        else:
            # datasets Image feature may return dict-like depending on version
            pil_img = Image.open(img["path"]).convert("RGB")  # type: ignore[index]

        ex = Example(
            idx=i,
            split=str(sp),
            question=str(row.get("question")),
            answer=str(row.get("answer")).strip().upper(),
            image=pil_img.convert("RGB"),
        )
        yield ex
        kept += 1
        if max_samples != -1 and kept >= max_samples:
            return


def load_qwen_vl(model_name: str, device: str, dtype: str):
    """
    Loads Qwen2.5-VL models in a way that works with recent transformers.

    Notes:
    - Some Qwen-VL checkpoints rely on AutoProcessor for multimodal handling.
    - We keep it flexible: model class is obtained from AutoModelForVision2Seq
      if available; otherwise fall back to AutoModel.
    """
    torch_dtype = getattr(torch, dtype)

    # IMPORTANT: for generation we need the *conditional generation* class.
    # With some transformers versions, AutoModel* may return a base model without `.generate`.
    model = None
    last_err: Optional[Exception] = None

    # Best path (per Qwen model card)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch_dtype,  # transformers>=5 uses dtype
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True,
            )
        except TypeError:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,  # transformers<=4 uses torch_dtype
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True,
            )
    except Exception as e:
        last_err = e

    # Fallbacks
    if model is None:
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True,
            )
        except Exception as e:
            last_err = e

    if model is None:
        try:
            from transformers import AutoModelForCausalLM  # type: ignore

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True,
            )
        except Exception as e:
            last_err = e

    if model is None:
        raise RuntimeError(f"Impossible de charger le modèle: {model_name}. Dernière erreur: {last_err}")

    # Validate generation availability early with a clear error
    if not hasattr(model, "generate"):
        raise RuntimeError(
            "Le modèle chargé n'expose pas `.generate()` (probablement une classe base). "
            "Pour Qwen2.5-VL, installez transformers depuis la branche main:\n"
            "  pip install -U 'git+https://github.com/huggingface/transformers.git' accelerate\n"
            "Puis relancez."
        )

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = None

    # Primary path: AutoProcessor (best for Qwen2.5-VL)
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except TypeError:
        # Workaround for older transformers bug in video_processing_auto.py:
        # TypeError: argument of type 'NoneType' is not iterable
        from transformers import AutoImageProcessor

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor = _FallbackProcessor(image_processor=image_processor, tokenizer=tokenizer)
    except Exception:
        # Last resort
        from transformers import AutoImageProcessor

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor = _FallbackProcessor(image_processor=image_processor, tokenizer=tokenizer)

    if device != "auto":
        model = model.to(device)

    model.eval()
    return model, processor, tokenizer


@torch.inference_mode()
def predict_one(
    model,
    processor,
    tokenizer,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
) -> str:
    """
    Generic multimodal generation wrapper.
    Tries a few common Qwen2.5-VL processor formats.
    """
    prompt = _build_prompt(question)

    # Many Qwen-VL processors accept (text, images) and return input tensors.
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Move tensors to model device
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    out_ids = model.generate(**inputs, **generate_kwargs)

    if tokenizer is not None:
        # Qwen examples trim the prompt tokens
        try:
            in_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            gen_ids = out_ids[0][in_len:] if in_len > 0 else out_ids[0]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    else:
        # AutoProcessor may include decode()
        try:
            in_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            gen_ids = out_ids[0][in_len:] if in_len > 0 else out_ids[0]
            text = processor.decode(gen_ids, skip_special_tokens=True)
        except Exception:
            text = str(out_ids)

    # Sometimes decodes include the prompt; keep last line-ish
    return text.strip()


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, List[bool]] = {}
    for r in rows:
        sp = r["split"]
        by_split.setdefault(sp, []).append(bool(r["correct"]))

    metrics = {
        "n": len(rows),
        "accuracy": float(np.mean([r["correct"] for r in rows])) if rows else 0.0,
        "by_split": {
            sp: {"n": len(v), "accuracy": float(np.mean(v)) if v else 0.0} for sp, v in by_split.items()
        },
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dataset_name", default="stogian/srbench")
    ap.add_argument("--dataset_split", default="test")
    ap.add_argument("--splits", nargs="+", default=["mrt_easy", "mrt_hard"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=1)  # reserved (we run 1-by-1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    if args.out_dir is None:
        safe_model = args.model_name.replace("/", "_")
        args.out_dir = f"runs/{safe_model}_{_now_id()}"

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # HF auth: datasets/models will pick up HF_TOKEN / HUGGINGFACE_HUB_TOKEN
    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    model, processor, tokenizer = load_qwen_vl(args.model_name, args.device, args.dtype)

    rows: List[Dict[str, Any]] = []
    pred_path = out_dir / "predictions.jsonl"

    with pred_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(iter_examples(ds, args.splits, args.max_samples), desc="Evaluating"):
            raw = predict_one(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=ex.image,
                question=ex.question,
                max_new_tokens=args.max_new_tokens,
            )
            pred = _parse_choice(raw)
            pred = pred if pred in CHOICES else None
            correct = (pred == ex.answer)

            row = {
                "idx": ex.idx,
                "split": ex.split,
                "answer": ex.answer,
                "pred": pred,
                "correct": bool(correct),
                "raw": raw,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = compute_metrics(rows)
    metrics.update(
        {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "splits": args.splits,
            "max_samples": args.max_samples,
        }
    )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

