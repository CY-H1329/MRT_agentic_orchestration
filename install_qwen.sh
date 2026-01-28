#!/usr/bin/env bash
# Script d'installation pour Qwen2.5-VL sur H100

set -euo pipefail

echo "=========================================="
echo "Installation Qwen2.5-VL pour SRBench MRT"
echo "=========================================="
echo ""

# 1. Mise à jour pip
echo "📦 Mise à jour de pip..."
python -m pip install -U pip

# 2. Installation de base
echo ""
echo "📦 Installation des dépendances de base..."
pip install -r requirements.txt

# 3. Vérifier transformers
echo ""
echo "🔍 Vérification de transformers..."
TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "NOT_INSTALLED")

if [[ "$TRANSFORMERS_VERSION" == "NOT_INSTALLED" ]]; then
    echo "❌ transformers non installé"
    exit 1
fi

echo "   Version actuelle: $TRANSFORMERS_VERSION"

# 4. Essayer d'importer Qwen2_5_VLForConditionalGeneration
echo ""
echo "🔍 Vérification de la classe Qwen2.5-VL..."
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('✅ OK')" 2>/dev/null || {
    echo "❌ Qwen2_5_VLForConditionalGeneration non disponible"
    echo ""
    echo "⚠️  Il faut installer transformers depuis GitHub:"
    echo "   pip install -U 'git+https://github.com/huggingface/transformers.git' accelerate"
    echo ""
    read -p "Voulez-vous installer depuis GitHub maintenant? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installation depuis GitHub..."
        pip install -U "git+https://github.com/huggingface/transformers.git" accelerate
        echo "✅ Installation terminée"
    else
        echo "❌ Installation annulée. Installez manuellement plus tard."
        exit 1
    fi
}

# 5. Vérifier qwen-vl-utils
echo ""
echo "🔍 Vérification de qwen-vl-utils..."
python -c "import qwen_vl_utils; print('✅ OK')" 2>/dev/null || {
    echo "❌ qwen-vl-utils non installé"
    echo "📦 Installation de qwen-vl-utils..."
    pip install qwen-vl-utils
}

echo ""
echo "=========================================="
echo "✅ Installation terminée!"
echo "=========================================="
echo ""
echo "Vérifiez avec:"
echo "  python -m srbench_qwen_mrt.check_qwen"
echo ""
