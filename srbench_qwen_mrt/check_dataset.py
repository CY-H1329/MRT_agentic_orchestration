from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description="Vérifier le chargement du dataset SRBench")
    ap.add_argument("--dataset_name", default="stogian/srbench")
    ap.add_argument("--dataset_split", default="test")
    ap.add_argument("--splits", nargs="+", default=["mrt_easy", "mrt_hard"])
    ap.add_argument("--max_samples", type=int, default=10, help="Nombre d'exemples à afficher")
    args = ap.parse_args()

    print(f"Chargement du dataset: {args.dataset_name} (split={args.dataset_split})")
    print(f"Filtrage des splits: {args.splits}")
    print("-" * 60)

    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    print(f"Dataset chargé: {len(ds)} exemples au total")

    # Compter par split (tous les splits)
    split_counts = Counter()
    # Compter spécifiquement les splits demandés
    target_split_counts = Counter()
    kept = []
    
    for i, row in enumerate(ds):
        sp = row.get("split")
        split_counts[sp] += 1
        if sp in args.splits:
            target_split_counts[sp] += 1
            kept.append((i, row))
            if args.max_samples > 0 and len(kept) >= args.max_samples:
                break

    print("\nRépartition par split (tous les splits du dataset):")
    for sp, count in split_counts.most_common():
        print(f"  {sp}: {count}")

    print(f"\nRépartition des splits demandés ({args.splits}):")
    for sp in args.splits:
        count = target_split_counts[sp]
        print(f"  {sp}: {count}")
    
    total_target = sum(target_split_counts.values())
    print(f"\nTotal ({args.splits}): {total_target}")
    print(f"Exemples affichés (max_samples={args.max_samples}): {len(kept)}")
    print("-" * 60)

    # Afficher quelques exemples
    for idx, (orig_idx, row) in enumerate(kept[:5]):
        print(f"\n[Exemple {idx+1}] (index original: {orig_idx})")
        print(f"  Split: {row.get('split')}")
        print(f"  Question: {row.get('question', '')[:150]}...")
        print(f"  Réponse attendue: {row.get('answer')}")

        img = row.get("image")
        if isinstance(img, Image.Image):
            print(f"  Image: {img.size} (mode: {img.mode})")
        else:
            print(f"  Image: {type(img)} (à convertir)")

    print("\n✅ Dataset chargé avec succès!")


if __name__ == "__main__":
    main()
