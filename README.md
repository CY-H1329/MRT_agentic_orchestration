# SRBench MRT — Évaluation Qwen2.5-VL

Ce repo sert à **évaluer Qwen2.5-VL** sur `stogian/srbench` (split `test`) en se concentrant sur:

- `mrt_easy`
- `mrt_hard`

## Principe

Le dataset fournit une image par exemple et une question de type QCM.  
Le script interroge le modèle et **force une sortie parmi {A,B,C,D}**, puis calcule l’accuracy par split et au global.

## Installation (H100 / Linux recommandé)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Accès Hugging Face

Si le modèle est gated / nécessite un token:

```bash
export HF_TOKEN="..."
```

## Lancer l’évaluation

Exemple (7B par défaut):

```bash
python -m srbench_qwen_mrt.eval_mrt \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --splits mrt_easy mrt_hard \
  --max_samples -1 \
  --batch_size 1 \
  --out_dir runs/qwen2.5-vl-7b
```

Le script écrit:

- `runs/.../predictions.jsonl`
- `runs/.../metrics.json`

## Exécution via GitHub Actions sur H100 (runner self-hosted)

Ce repo inclut un workflow `.github/workflows/run_mrt.yml` conçu pour tourner sur un runner **self-hosted** (ex: machine H100).
Il:

1. installe les dépendances
2. exécute l’évaluation
3. commit & push les résultats dans `runs/`

Voir la section “Mise en place du runner H100” plus bas.

## Mise en place du runner H100 (résumé)

1) Sur GitHub (repo) → **Settings → Actions → Runners → New self-hosted runner**  
2) Sur le serveur H100, suivez les commandes fournies par GitHub pour enregistrer le runner.  
3) Donnez-lui un label (par ex `h100`) et vérifiez que le workflow cible bien ce label.

## Notes

- Le pipeline est volontairement simple pour un premier “smoke test” MRT. Ensuite on pourra ajouter:
  - décodage plus contraint (logits mask) sur A/B/C/D
  - ensemble de prompts / calibrations
  - mesures d’incertitude / abstention

### Dépannage (JupyterHub / H100)

Si vous voyez une erreur du type:

- `TypeError: argument of type 'NoneType' is not iterable` dans `transformers/models/auto/video_processing_auto.py`

Alors vous avez une **version trop ancienne** de `transformers`. Corrigez avec:

```bash
pip install -U "transformers>=4.48.0" "accelerate>=0.30.0" "datasets>=2.18.0"
```

