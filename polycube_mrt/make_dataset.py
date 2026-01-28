from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
from tqdm import tqdm


_RE_ORIG = re.compile(r"^polycube_(\d+)_original_mono\.png$")
_RE_ANGLE = re.compile(r"^polycube_(\d+)_angle(\d+)_mono\.png$")


@dataclass
class PolycubeGroup:
    poly_id: str
    original_mono: Path
    angle_mono: Dict[int, Path]  # angle -> path


def _load_groups(data_dir: Path) -> Dict[str, PolycubeGroup]:
    groups: Dict[str, PolycubeGroup] = {}
    for name in os.listdir(data_dir):
        p = data_dir / name
        if not p.is_file():
            continue

        m = _RE_ORIG.match(name)
        if m:
            pid = m.group(1)
            g = groups.get(pid)
            if g is None:
                g = PolycubeGroup(poly_id=pid, original_mono=p, angle_mono={})
                groups[pid] = g
            else:
                g.original_mono = p
            continue

        m = _RE_ANGLE.match(name)
        if m:
            pid = m.group(1)
            ang = int(m.group(2))
            g = groups.get(pid)
            if g is None:
                # placeholder original; will be validated later
                g = PolycubeGroup(poly_id=pid, original_mono=data_dir / f"polycube_{pid}_original_mono.png", angle_mono={})
                groups[pid] = g
            g.angle_mono[ang] = p
            continue

    # filter valid groups
    out: Dict[str, PolycubeGroup] = {}
    for pid, g in groups.items():
        if not g.original_mono.exists():
            continue
        if not g.angle_mono:
            continue
        out[pid] = g
    return out


def _resize_to(img: Image.Image, size: int) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.BICUBIC)


def _make_composite(
    query_path: Path,
    option_paths: List[Path],
    tile_size: int,
    out_path: Path,
    labels: Optional[List[str]] = None,
) -> None:
    """
    Layout (3x2 tiles):
      row0: [blank][QUERY][blank]
      row1: [A][B][C]
    """
    assert len(option_paths) == 3
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg = Image.new("RGB", (tile_size * 3, tile_size * 2), (255, 255, 255))
    draw = ImageDraw.Draw(bg)

    q = _resize_to(Image.open(query_path), tile_size)
    bg.paste(q, (tile_size * 1, 0))

    for i, p in enumerate(option_paths):
        im = _resize_to(Image.open(p), tile_size)
        bg.paste(im, (tile_size * i, tile_size))

    # Labels
    draw.rectangle([tile_size * 1 + 4, 4, tile_size * 1 + 28, 28], fill=(255, 255, 255))
    draw.text((tile_size * 1 + 8, 6), "Q", fill=(0, 0, 0))

    if labels is None:
        labels = ["A", "B", "C"]
    for i, lab in enumerate(labels):
        x0, y0 = tile_size * i + 4, tile_size + 4
        draw.rectangle([x0, y0, x0 + 24, y0 + 24], fill=(255, 255, 255))
        draw.text((x0 + 6, y0 + 4), lab, fill=(0, 0, 0))

    bg.save(out_path, format="PNG")


def _question_text() -> str:
    return (
        "This image shows a 3D polycube shape.\n"
        "The QUERY view is at the top-center (marked 'Q').\n"
        "Which option (A, B, or C) at the bottom row is simply the QUERY shape in a rotated orientation?\n"
        "Only one option is correct.\n"
        "Available options: A. Left, B. Center, C. Right"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Dossier contenant polycube_XXXXX_original_mono.png et polycube_XXXXX_angleY_mono.png")
    ap.add_argument("--out_dir", required=True, help="Dossier de sortie (jsonl + images composites)")
    ap.add_argument("--n_per_split", type=int, default=200, help="Nombre d'exemples à générer par split (mrt_easy/mrt_hard).")
    ap.add_argument("--splits", nargs="+", default=["mrt_easy", "mrt_hard"])
    ap.add_argument("--seed", type=int, default=123, help="Seed RNG pour reproductibilité.")
    ap.add_argument("--tile_size", type=int, default=256, help="Taille (px) d'un tile dans l'image composite.")
    ap.add_argument("--angles", nargs="*", type=int, default=None, help="Sous-ensemble d'angles autorisés (ex: 1 2). Par défaut: tous angles trouvés.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    groups = _load_groups(data_dir)
    if not groups:
        raise RuntimeError(f"Aucun polycube valide trouvé dans: {data_dir}")

    all_ids = sorted(groups.keys())
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "dataset.jsonl"
    meta_path = out_dir / "meta.json"

    counts = {sp: 0 for sp in args.splits}
    total_target = args.n_per_split * len(args.splits)

    def pick_angle(g: PolycubeGroup) -> Tuple[int, Path]:
        angs = sorted(g.angle_mono.keys())
        if args.angles is not None:
            angs = [a for a in angs if a in set(args.angles)]
        if not angs:
            raise RuntimeError(f"Polycube {g.poly_id} n'a aucun angle dans la liste demandée.")
        ang = rng.choice(angs)
        return ang, g.angle_mono[ang]

    with jsonl_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=total_target, desc="Generating MRT examples")
        while True:
            done = all(counts[sp] >= args.n_per_split for sp in args.splits)
            if done:
                break

            # choose split that still needs examples
            pending_splits = [sp for sp in args.splits if counts[sp] < args.n_per_split]
            split = rng.choice(pending_splits)

            # choose base polycube id
            pid = rng.choice(all_ids)
            g = groups[pid]

            # correct option: rotated view of SAME polycube
            ang, correct_path = pick_angle(g)

            # distractors: 2 different polycubes (original_mono)
            distract_ids = rng.sample([x for x in all_ids if x != pid], k=2)
            distract_paths = [groups[x].original_mono for x in distract_ids]

            # options order
            option_paths = [correct_path] + distract_paths
            rng.shuffle(option_paths)
            labels = ["A", "B", "C"]
            correct_idx = option_paths.index(correct_path)
            answer = labels[correct_idx]

            # composite image
            ex_id = f"{split}_pid{pid}_ang{ang}_seed{args.seed}_{counts[split]:04d}"
            composite_path = images_dir / f"{ex_id}.png"
            _make_composite(
                query_path=g.original_mono,
                option_paths=option_paths,
                tile_size=int(args.tile_size),
                out_path=composite_path,
                labels=labels,
            )

            row = {
                "id": ex_id,
                "split": split,
                "question": _question_text(),
                "answer": answer,
                "image": str(composite_path),
                "meta": {
                    "polycube_id": pid,
                    "angle": ang,
                    "query_original_mono": str(g.original_mono),
                    "correct_image": str(correct_path),
                    "options": {
                        "A": str(option_paths[0]),
                        "B": str(option_paths[1]),
                        "C": str(option_paths[2]),
                    },
                    "distractor_ids": distract_ids,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            pbar.update(1)

        pbar.close()

    meta = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "tile_size": args.tile_size,
        "n_per_split": args.n_per_split,
        "splits": args.splits,
        "counts": counts,
        "total": sum(counts.values()),
        "note": "Each example is a composite image: top-center query (Q), bottom row options A/B/C.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

