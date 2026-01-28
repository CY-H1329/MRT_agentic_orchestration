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
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\(([ABCD])\)", t)
    if m:
        return m.group(1)
    m = re.match(r"^\s*([ABCD])\s*(?:[.\-:)]|\s|$)", t)
    if m:
        return m.group(1)
    return None


def _build_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Réponds avec UNE SEULE LETTRE parmi {A,B,C,D}.\n"
        "N'ajoute aucun autre texte."
    )


def _pil_to_data_url(img: Image.Image) -> str:
    # SRBench images are small; PNG keeps it simple.
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


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


def openai_client():
    # Supports both OpenAI and Azure-style envs if needed later.
    # For now we use the official OpenAI Python SDK v1.
    from openai import OpenAI

    return OpenAI()


def call_gpt4o(image: Image.Image, question: str, model_name: str, max_output_tokens: int) -> str:
    client = openai_client()

    data_url = _pil_to_data_url(image)
    prompt = _build_prompt(question)

    resp = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": data_url},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )

    # SDK returns a structured response; `.output_text` is the simplest.
    return (resp.output_text or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt-4o")
    ap.add_argument("--dataset_name", default="stogian/srbench")
    ap.add_argument("--dataset_split", default="test")
    ap.add_argument("--splits", nargs="+", default=["mrt_easy", "mrt_hard"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_output_tokens", type=int, default=16)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    # Avoid overwriting previous results
    if args.out_dir is None:
        safe_model = args.model_name.replace("/", "_")
        args.out_dir = f"runs/{safe_model}_{_now_id()}"
    else:
        args.out_dir = f"{args.out_dir.rstrip('/')}_{_now_id()}"

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Require API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY manquant. Ex: export OPENAI_API_KEY='...'.")

    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for ex in tqdm(iter_examples(ds, args.splits, args.max_samples, args.shuffle, args.seed), desc="Evaluating"):
            raw = call_gpt4o(ex.image, ex.question, args.model_name, args.max_output_tokens)
            pred = _parse_choice(raw)
            pred = pred if pred in CHOICES else None
            correct = (pred == ex.answer)

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

