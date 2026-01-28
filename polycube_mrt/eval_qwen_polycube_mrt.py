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
from srbench_qwen_mrt.eval_mrt import _ensure_dir, _now_id, _parse_choice, load_qwen_vl  # type: ignore


CHOICES = ["A", "B", "C"]


@dataclass
class Example:
    idx: int
    split: str
    question: str
    answer: str
    image_path: Path
    depth_query_path: Optional[Path] = None
    depth_options: Optional[Dict[str, Path]] = None
    color_query_path: Optional[Path] = None
    color_options: Optional[Dict[str, Path]] = None


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
        depth_query = None
        depth_opts = None
        color_query = None
        color_opts = None
        meta = r.get("meta") or {}
        depth_meta = meta.get("depth") if isinstance(meta, dict) else None
        if isinstance(depth_meta, dict):
            qd = depth_meta.get("query_depth")
            od = depth_meta.get("options_depth")
            if isinstance(qd, str):
                depth_query = Path(qd)
            if isinstance(od, dict):
                tmp: Dict[str, Path] = {}
                for k in ["A", "B", "C"]:
                    if isinstance(od.get(k), str):
                        tmp[k] = Path(od[k])
                if len(tmp) == 3:
                    depth_opts = tmp

        color_meta = meta.get("color") if isinstance(meta, dict) else None
        if isinstance(color_meta, dict):
            qc = color_meta.get("query_color")
            oc = color_meta.get("options_color")
            if isinstance(qc, str):
                color_query = Path(qc)
            if isinstance(oc, dict):
                tmpc: Dict[str, Path] = {}
                for k in ["A", "B", "C"]:
                    if isinstance(oc.get(k), str):
                        tmpc[k] = Path(oc[k])
                if len(tmpc) == 3:
                    color_opts = tmpc

        ex = Example(
            idx=i,
            split=str(r["split"]),
            question=str(r["question"]),
            answer=str(r["answer"]).strip().upper(),
            image_path=Path(r["image"]),
            depth_query_path=depth_query,
            depth_options=depth_opts,
            color_query_path=color_query,
            color_options=color_opts,
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
    ap.add_argument("--use_depth_inputs", action="store_true", help="Si dataset contient meta.depth, ajouter query/options depth comme images supplémentaires au modèle.")
    ap.add_argument("--use_color_inputs", action="store_true", help="Si dataset contient meta.color, ajouter query/options color comme images supplémentaires au modèle.")
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
            # Load main composite (mono)
            img_main = Image.open(ex.image_path).convert("RGB")

            # Multi-image input (optional): composite + color and/or depth
            images: List[Image.Image] = [img_main]
            used_depth = False
            used_color = False

            if args.use_color_inputs and ex.color_query_path is not None and ex.color_options is not None:
                images.append(Image.open(ex.color_query_path).convert("RGB"))
                images.append(Image.open(ex.color_options["A"]).convert("RGB"))
                images.append(Image.open(ex.color_options["B"]).convert("RGB"))
                images.append(Image.open(ex.color_options["C"]).convert("RGB"))
                used_color = True

            if args.use_depth_inputs and ex.depth_query_path is not None and ex.depth_options is not None:
                images.append(Image.open(ex.depth_query_path).convert("RGB"))
                images.append(Image.open(ex.depth_options["A"]).convert("RGB"))
                images.append(Image.open(ex.depth_options["B"]).convert("RGB"))
                images.append(Image.open(ex.depth_options["C"]).convert("RGB"))
                used_depth = True

            raw = predict_one_multi(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                images=images,
                question=ex.question,
                max_new_tokens=args.max_new_tokens,
                use_depth=used_depth,
                use_color=used_color,
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
                    "image_size": list(img_main.size),
                    "use_depth_inputs": bool(args.use_depth_inputs),
                    "use_color_inputs": bool(args.use_color_inputs),
                    "n_images": len(images),
                    "depth_query_path": str(ex.depth_query_path) if ex.depth_query_path else None,
                    "color_query_path": str(ex.color_query_path) if ex.color_query_path else None,
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


def _build_prompt_polycube(question: str, use_depth: bool, use_color: bool) -> str:
    if not use_depth and not use_color:
        return (
            f"{question}\n\n"
            "Réponds avec UNE SEULE LETTRE parmi {A,B,C}.\n"
            "N'ajoute aucun autre texte."
        )
    lines = [
        f"{question}",
        "",
        "INSTRUCTIONS (important):",
        "- Focus on matching the 3D SHAPE under rotation (same polycube).",
        "- Use COLOR arrangement as a strong cue: compare which colored faces/cubes are adjacent and their relative positions.",
        "- Ignore background, lighting, and minor rendering differences.",
        "- Do NOT guess from text; decide by visual comparison.",
        "",
        "How to compare (recommended):",
        "1) Identify distinctive color patterns on the QUERY (e.g., which colors touch, stacked order, left-right relations).",
        "2) Mentally rotate the QUERY and check which option preserves BOTH geometry + color adjacency relations.",
        "3) Exactly one option matches.",
        "",
        "You are given images in order:",
    ]
    idx = 1
    lines.append(f"{idx}) Composite mono (Q at top-center, options A/B/C on bottom row)")
    idx += 1
    if use_color:
        lines.append(f"{idx}) Color-QUERY (original)"); idx += 1
        lines.append(f"{idx}) Color-A"); idx += 1
        lines.append(f"{idx}) Color-B"); idx += 1
        lines.append(f"{idx}) Color-C"); idx += 1
    if use_depth:
        lines.append(f"{idx}) Depth-QUERY (original)"); idx += 1
        lines.append(f"{idx}) Depth-A"); idx += 1
        lines.append(f"{idx}) Depth-B"); idx += 1
        lines.append(f"{idx}) Depth-C"); idx += 1
    lines.append("")
    lines.append("Answer with EXACTLY ONE LETTER: A, B, or C. No other text.")
    return "\n".join(lines)


def predict_one_multi(
    model,
    processor,
    tokenizer,
    images: List[Image.Image],
    question: str,
    max_new_tokens: int,
    use_depth: bool,
    use_color: bool,
) -> str:
    """
    Qwen2.5-VL multi-image generation wrapper (composite + optional depth images).
    Mirrors srbench_qwen_mrt.eval_mrt.predict_one logic but supports multiple images.
    """
    import torch

    prompt = _build_prompt_polycube(question, use_depth=use_depth, use_color=use_color)

    if hasattr(processor, "apply_chat_template"):
        from qwen_vl_utils import process_vision_info  # type: ignore

        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        # Best-effort fallback: some processors accept list of images
        inputs = processor(text=prompt, images=images, return_tensors="pt")

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # trim prompt tokens if possible
    try:
        in_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        gen_ids = out_ids[0][in_len:] if in_len > 0 else out_ids[0]
    except Exception:
        gen_ids = out_ids[0]

    if tokenizer is not None:
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # fallback
    try:
        return processor.decode(gen_ids, skip_special_tokens=True).strip()
    except Exception:
        return str(out_ids)


if __name__ == "__main__":
    main()

