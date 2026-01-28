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
from PIL import Image
from tqdm import tqdm


CHOICES = ["A", "B", "C"]


@dataclass
class Example:
    idx: int
    split: str
    question: str
    answer: str
    image_path: Path
    color_query_path: Optional[Path] = None
    color_options: Optional[Dict[str, Path]] = None
    depth_query_path: Optional[Path] = None
    depth_options: Optional[Dict[str, Path]] = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_choice(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"\b([ABC])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\(([ABC])\)", t)
    if m:
        return m.group(1)
    m = re.match(r"^\s*([ABC])\s*(?:[.\-:)]|\s|$)", t)
    if m:
        return m.group(1)
    return None


def _pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


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
        color_query = None
        color_opts = None
        depth_query = None
        depth_opts = None
        meta = r.get("meta") or {}
        
        # Parse color paths
        color_meta = meta.get("color") if isinstance(meta, dict) else None
        if isinstance(color_meta, dict):
            qc = color_meta.get("query_color")
            oc = color_meta.get("options_color")
            if isinstance(qc, str):
                color_query = Path(qc)
            if isinstance(oc, dict):
                tmp: Dict[str, Path] = {}
                for k in ["A", "B", "C"]:
                    if isinstance(oc.get(k), str):
                        tmp[k] = Path(oc[k])
                if len(tmp) == 3:
                    color_opts = tmp
        
        # Parse depth paths
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

        ex = Example(
            idx=i,
            split=str(r["split"]),
            question=str(r["question"]),
            answer=str(r["answer"]).strip().upper(),
            image_path=Path(r["image"]),
            color_query_path=color_query,
            color_options=color_opts,
            depth_query_path=depth_query,
            depth_options=depth_opts,
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


def openai_client():
    from openai import OpenAI

    def _clean_header_value(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        return s.strip().replace("\u2028", "").replace("\u2029", "").replace("\r", "").replace("\n", "")

    api_key = _clean_header_value(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        return OpenAI()

    organization = _clean_header_value(os.getenv("OPENAI_ORG_ID"))
    project = _clean_header_value(os.getenv("OPENAI_PROJECT_ID"))

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if organization:
        kwargs["organization"] = organization
    if project:
        kwargs["project"] = project
    return OpenAI(**kwargs)


def _build_prompt(question: str, use_color_inputs: bool, use_depth_inputs: bool) -> str:
    if not use_color_inputs and not use_depth_inputs:
        return (
            f"{question}\n\n"
            "Answer with EXACTLY ONE LETTER: A, B, or C. No other text."
        )
    
    parts = [f"{question}\n\n", "You will receive multiple images.\n"]
    
    if use_color_inputs and use_depth_inputs:
        parts.append(
            "Use the COLOR images to compare the arrangement of colored cubes/faces.\n"
            "Use the DEPTH images to compare the 3D geometry and structure.\n"
            "Combine both visual cues to pick the option that matches the QUERY under rotation (same shape + consistent color adjacency + matching depth structure).\n"
        )
    elif use_color_inputs:
        parts.append(
            "Use the COLOR images to compare the arrangement of colored cubes/faces.\n"
            "Pick the option that matches the QUERY under rotation (same shape + consistent color adjacency).\n"
        )
    elif use_depth_inputs:
        parts.append(
            "Use the DEPTH images to compare the 3D geometry and structure.\n"
            "Pick the option that matches the QUERY under rotation (same shape + matching depth structure).\n"
        )
    
    parts.append("\nAnswer with EXACTLY ONE LETTER: A, B, or C. No other text.")
    return "".join(parts)


def call_gpt4o(images: List[Image.Image], question: str, model_name: str, max_output_tokens: int, use_color_inputs: bool, use_depth_inputs: bool) -> str:
    client = openai_client()

    content: List[Dict[str, Any]] = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}})
    content.append({"type": "text", "text": _build_prompt(question, use_color_inputs=use_color_inputs, use_depth_inputs=use_depth_inputs)})

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_output_tokens,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt-4o")
    ap.add_argument("--data_jsonl", required=True, help="Chemin vers polycube MRT dataset.jsonl")
    ap.add_argument("--splits", nargs="+", default=["mrt_easy", "mrt_hard"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_output_tokens", type=int, default=8)
    ap.add_argument("--use_color_inputs", action="store_true", help="Envoyer aussi Color-QUERY + Color-A/B/C si présents dans meta.color.")
    ap.add_argument("--use_depth_inputs", action="store_true", help="Envoyer aussi Depth-QUERY + Depth-A/B/C si présents dans meta.depth.")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY manquant. Ex: export OPENAI_API_KEY='...'.")

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

    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for ex in tqdm(iter_examples(data_jsonl, args.splits, args.max_samples, args.shuffle, args.seed), desc="Evaluating"):
            # Images order:
            # 1) composite mono (always)
            # 2..5) color query + color A/B/C (if --use_color_inputs)
            # 6..9) depth query + depth A/B/C (if --use_depth_inputs)
            images: List[Image.Image] = [Image.open(ex.image_path).convert("RGB")]
            used_color = False
            used_depth = False
            
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

            raw = call_gpt4o(images, ex.question, args.model_name, args.max_output_tokens, use_color_inputs=used_color, use_depth_inputs=used_depth)
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

            detailed.append(
                {
                    "idx": ex.idx,
                    "split": ex.split,
                    "question": ex.question,
                    "ground_truth": ex.answer,
                    "model_prediction": pred,
                    "model_raw_output": raw,
                    "correct": bool(correct),
                    "image_path": str(ex.image_path),
                    "use_color_inputs": used_color,
                    "use_depth_inputs": used_depth,
                    "n_images": len(images),
                    "color_query_path": str(ex.color_query_path) if ex.color_query_path else None,
                    "depth_query_path": str(ex.depth_query_path) if ex.depth_query_path else None,
                }
            )

    (out_dir / "detailed_results.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "data_jsonl": str(data_jsonl),
                "splits": args.splits,
                "max_samples": args.max_samples,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "max_output_tokens": args.max_output_tokens,
                "use_color_inputs": args.use_color_inputs,
                "use_depth_inputs": args.use_depth_inputs,
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
            "data_jsonl": str(data_jsonl),
            "splits": args.splits,
            "max_samples": args.max_samples,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "max_output_tokens": args.max_output_tokens,
            "use_color_inputs": args.use_color_inputs,
            "use_depth_inputs": args.use_depth_inputs,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

