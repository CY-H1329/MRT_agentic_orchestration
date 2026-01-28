from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# Reuse Qwen loading + inference from existing srbench evaluator
from srbench_qwen_mrt.eval_mrt import _ensure_dir, _now_id, _parse_choice, load_qwen_vl, predict_one  # type: ignore


CHOICES = ["A", "B", "C"]


@dataclass
class Example:
    idx: int
    split: str
    question: str
    answer: str
    image_path: Path


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_examples(jsonl_path: Path, splits: List[str], max_samples: int, shuffle: bool, seed: Optional[int]) -> Iterable[Example]:
    rows = [r for r in _iter_jsonl(jsonl_path) if r.get("split") in set(splits)]
    if shuffle:
        rng = random.Random(seed if seed is not None else int(time.time() * 1_000_000) % (2**32))
        rng.shuffle(rows)

    count = 0
    for i, r in enumerate(rows):
        ex = Example(
            idx=i,
            split=str(r["split"]),
            question=str(r["question"]),
            answer=str(r["answer"]).strip().upper(),
            image_path=Path(r["image"]),
        )
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data_jsonl", required=True, help="Chemin vers polycube MRT dataset.jsonl")
    ap.add_argument("--splits", nargs="+", default=["mrt_easy", "mrt_hard"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    data_jsonl = Path(args.data_jsonl).expanduser()
    if not data_jsonl.exists():
        raise FileNotFoundError(f"dataset.jsonl introuvable: {data_jsonl}")

    if args.out_dir is None:
        safe_model = args.model_name.replace("/", "_")
        args.out_dir = f"runs/{safe_model}_polycube_{_now_id()}"
    else:
        args.out_dir = f"{args.out_dir.rstrip('/')}_{_now_id()}"

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    model, processor, tokenizer = load_qwen_vl(args.model_name, args.device, args.dtype)

    rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []
    pred_path = out_dir / "predictions.jsonl"
    detailed_path = out_dir / "detailed_results.json"

    with pred_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(iter_examples(data_jsonl, args.splits, args.max_samples, args.shuffle, args.seed), desc="Evaluating"):
            img = Image.open(ex.image_path).convert("RGB")
            raw = predict_one(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=img,
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
                "image": str(ex.image_path),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            detailed_results.append(
                {
                    "idx": ex.idx,
                    "split": ex.split,
                    "question": ex.question,
                    "ground_truth": ex.answer,
                    "model_prediction": pred,
                    "model_raw_output": raw,
                    "correct": bool(correct),
                    "image_path": str(ex.image_path),
                    "image_size": list(img.size),
                }
            )

    detailed_output = {
        "model_name": args.model_name,
        "data_jsonl": str(data_jsonl),
        "splits": args.splits,
        "max_samples": args.max_samples,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "total_examples": len(detailed_results),
        "results": detailed_results,
    }
    detailed_path.write_text(json.dumps(detailed_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = compute_metrics(rows)
    metrics.update(
        {
            "model_name": args.model_name,
            "data_jsonl": str(data_jsonl),
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

