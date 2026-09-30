#!/bin/bash
set -e
echo "🚀 [1/3] Clonando repositório oficial do Space MiniMax-H3..."
cd /content
rm -rf /content/minimax-h3
git clone https://huggingface.co/spaces/Pepe104/MiniMax-H3-Turbo-Lora-UNCENSORED /content/minimax-h3

echo "📦 [2/3] Instalando dependências e motor 8-bit torchao..."
cd /content/minimax-h3
pip install -q -r requirements.txt
pip install -q torchao

echo "✅ [3/3] Ambiente configurado com sucesso!"
