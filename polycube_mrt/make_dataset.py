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


_RE_ORIG_MONO = re.compile(r"^polycube_(\d+)_original_mono\.png$")
_RE_ANGLE_MONO = re.compile(r"^polycube_(\d+)_angle(\d+)_mono\.png$")
_RE_ORIG_DEPTH = re.compile(r"^polycube_(\d+)_original_depth\.png$")
_RE_ANGLE_DEPTH = re.compile(r"^polycube_(\d+)_angle(\d+)_depth\.png$")
_RE_ORIG_COLOR = re.compile(r"^polycube_(\d+)_original_color\.png$")
_RE_ANGLE_COLOR = re.compile(r"^polycube_(\d+)_angle(\d+)_color\.png$")


@dataclass
class PolycubeGroup:
    poly_id: str
    original_mono: Path
    angle_mono: Dict[int, Path]  # angle -> path
    original_depth: Optional[Path] = None
    angle_depth: Optional[Dict[int, Path]] = None  # angle -> path
    original_color: Optional[Path] = None
    angle_color: Optional[Dict[int, Path]] = None  # angle -> path


def _load_groups(data_dir: Path) -> Dict[str, PolycubeGroup]:
    groups: Dict[str, PolycubeGroup] = {}
    for name in os.listdir(data_dir):
        p = data_dir / name
        if not p.is_file():
            continue

        m = _RE_ORIG_MONO.match(name)
        if m:
            pid = m.group(1)
            g = groups.get(pid)
            if g is None:
                g = PolycubeGroup(
                    poly_id=pid,
                    original_mono=p,
                    angle_mono={},
                    original_depth=None,
                    angle_depth={},
                    original_color=None,
                    angle_color={},
                )
                groups[pid] = g
            else:
                g.original_mono = p
            continue

        m = _RE_ANGLE_MONO.match(name)
        if m:
            pid = m.group(1)
            ang = int(m.group(2))
            g = groups.get(pid)
            if g is None:
                # placeholder original; will be validated later
                g = PolycubeGroup(
                    poly_id=pid,
                    original_mono=data_dir / f"polycube_{pid}_original_mono.png",
                    angle_mono={},
                    original_depth=None,
                    angle_depth={},
                    original_color=None,
                    angle_color={},
                )
                groups[pid] = g
            g.angle_mono[ang] = p
            continue

        m = _RE_ORIG_DEPTH.match(name)
        if m:
            pid = m.group(1)
            g = groups.get(pid)
            if g is None:
                g = PolycubeGroup(
                    poly_id=pid,
                    original_mono=data_dir / f"polycube_{pid}_original_mono.png",
                    angle_mono={},
                    original_depth=p,
                    angle_depth={},
                    original_color=None,
                    angle_color={},
                )
                groups[pid] = g
            else:
                g.original_depth = p
                if g.angle_depth is None:
                    g.angle_depth = {}
            continue

        m = _RE_ANGLE_DEPTH.match(name)
        if m:
            pid = m.group(1)
            ang = int(m.group(2))
            g = groups.get(pid)
            if g is None:
                g = PolycubeGroup(
                    poly_id=pid,
                    original_mono=data_dir / f"polycube_{pid}_original_mono.png",
                    angle_mono={},
                    original_depth=None,
                    angle_depth={},
                    original_color=None,
                    angle_color={},
                )
                groups[pid] = g
            if g.angle_depth is None:
                g.angle_depth = {}
            g.angle_depth[ang] = p
            continue

        m = _RE_ORIG_COLOR.match(name)
        if m:
            pid = m.group(1)
            g = groups.get(pid)
            if g is None:
                g = PolycubeGroup(
                    poly_id=pid,
                    original_mono=data_dir / f"polycube_{pid}_original_mono.png",
                    angle_mono={},
                    original_depth=None,
                    angle_depth={},
                    original_color=p,
                    angle_color={},
                )
                groups[pid] = g
            else:
                g.original_color = p
                if g.angle_color is None:
                    g.angle_color = {}
            continue

        m = _RE_ANGLE_COLOR.match(name)
        if m:
            pid = m.group(1)
            ang = int(m.group(2))
            g = groups.get(pid)
            if g is None:
                g = PolycubeGroup(
                    poly_id=pid,
                    original_mono=data_dir / f"polycube_{pid}_original_mono.png",
                    angle_mono={},
                    original_depth=None,
                    angle_depth={},
                    original_color=None,
                    angle_color={},
                )
                groups[pid] = g
            if g.angle_color is None:
                g.angle_color = {}
            g.angle_color[ang] = p
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


def _question_text(include_depth: bool, include_color: bool) -> str:
    return (
        "This image shows a 3D polycube shape.\n"
        "The QUERY view is at the top-center (marked 'Q').\n"
        "Which option (A, B, or C) at the bottom row is simply the QUERY shape in a rotated orientation?\n"
        "Only one option is correct.\n"
        "Available options: A. Left, B. Center, C. Right\n"
        + (
            "\nAdditional inputs are provided as separate DEPTH images:\n"
            "- Depth-QUERY (original)\n"
            "- Depth-A, Depth-B, Depth-C (options)\n"
            "Use them to decide the correct rotation match."
            if include_depth
            else ""
        )
        + (
            "\nAdditional inputs are provided as separate COLOR images:\n"
            "- Color-QUERY (original)\n"
            "- Color-A, Color-B, Color-C (options)\n"
            "Use them to decide the correct rotation match."
            if include_color
            else ""
        )
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
    ap.add_argument("--include_depth", action="store_true", help="Inclure les chemins depth (query+options) dans dataset.jsonl (pour multi-image input).")
    ap.add_argument("--include_color", action="store_true", help="Inclure les chemins color (query+options) dans dataset.jsonl (pour multi-image input).")
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

    def pick_angle_depth(g: PolycubeGroup, ang: int) -> Optional[Path]:
        if g.angle_depth is None:
            return None
        return g.angle_depth.get(ang)

    def pick_angle_color(g: PolycubeGroup, ang: int) -> Optional[Path]:
        if g.angle_color is None:
            return None
        return g.angle_color.get(ang)

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
            correct_depth_path = pick_angle_depth(g, ang) if args.include_depth else None
            correct_color_path = pick_angle_color(g, ang) if args.include_color else None

            # distractors: 2 different polycubes (original_mono)
            distract_ids = rng.sample([x for x in all_ids if x != pid], k=2)
            distract_paths = [groups[x].original_mono for x in distract_ids]
            distract_depth_paths = [groups[x].original_depth for x in distract_ids] if args.include_depth else [None, None]
            distract_color_paths = [groups[x].original_color for x in distract_ids] if args.include_color else [None, None]

            # if depth required, ensure we have query depth + correct depth + distractor depths
            if args.include_depth:
                if g.original_depth is None or correct_depth_path is None:
                    continue
                if any(d is None for d in distract_depth_paths):
                    continue

            if args.include_color:
                if g.original_color is None or correct_color_path is None:
                    continue
                if any(c is None for c in distract_color_paths):
                    continue

            # options order
            option_paths = [correct_path] + distract_paths
            rng.shuffle(option_paths)
            labels = ["A", "B", "C"]
            correct_idx = option_paths.index(correct_path)
            answer = labels[correct_idx]

            option_depth_paths: Optional[List[Path]] = None
            if args.include_depth:
                # keep depth aligned with option_paths
                all_opt_pairs: List[Tuple[Path, Path]] = []
                all_opt_pairs.append((correct_path, correct_depth_path))  # type: ignore[arg-type]
                all_opt_pairs.append((distract_paths[0], distract_depth_paths[0]))  # type: ignore[arg-type]
                all_opt_pairs.append((distract_paths[1], distract_depth_paths[1]))  # type: ignore[arg-type]
                # reorder pairs to match option_paths
                depth_by_img = {str(img_p): depth_p for (img_p, depth_p) in all_opt_pairs}
                option_depth_paths = [Path(depth_by_img[str(p)]) for p in option_paths]  # type: ignore[index]

            option_color_paths: Optional[List[Path]] = None
            if args.include_color:
                all_opt_pairs_c: List[Tuple[Path, Path]] = []
                all_opt_pairs_c.append((correct_path, correct_color_path))  # type: ignore[arg-type]
                all_opt_pairs_c.append((distract_paths[0], distract_color_paths[0]))  # type: ignore[arg-type]
                all_opt_pairs_c.append((distract_paths[1], distract_color_paths[1]))  # type: ignore[arg-type]
                color_by_img = {str(img_p): color_p for (img_p, color_p) in all_opt_pairs_c}
                option_color_paths = [Path(color_by_img[str(p)]) for p in option_paths]  # type: ignore[index]

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

            depth_payload: Optional[Dict[str, Any]] = None
            if args.include_depth and option_depth_paths is not None:
                depth_payload = {
                    "query_depth": str(g.original_depth),
                    "options_depth": {
                        "A": str(option_depth_paths[0]),
                        "B": str(option_depth_paths[1]),
                        "C": str(option_depth_paths[2]),
                    },
                }

            color_payload: Optional[Dict[str, Any]] = None
            if args.include_color and option_color_paths is not None:
                color_payload = {
                    "query_color": str(g.original_color),
                    "options_color": {
                        "A": str(option_color_paths[0]),
                        "B": str(option_color_paths[1]),
                        "C": str(option_color_paths[2]),
                    },
                }

            row = {
                "id": ex_id,
                "split": split,
                "question": _question_text(include_depth=args.include_depth, include_color=args.include_color),
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
                    **({"depth": depth_payload} if depth_payload is not None else {}),
                    **({"color": color_payload} if color_payload is not None else {}),
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
        "include_depth": bool(args.include_depth),
        "include_color": bool(args.include_color),
        "note": "Each example is a composite image: top-center query (Q), bottom row options A/B/C. If include_depth/meta.depth or include_color/meta.color, extra paths are included for multi-image input.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

