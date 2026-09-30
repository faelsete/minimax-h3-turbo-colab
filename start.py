import os
import sys

print("⚙️ Preparando MiniMax-H3 com quantização 8-bit e CPU Offload inteligente...")
os.chdir("/content/minimax-h3")
os.system("git checkout app.py 2>/dev/null || true")

with open("app.py", "r") as f:
    code = f.read()

# 1. Configurar launch com link público (share=True)
code = code.replace("app.launch(show_error=True, allowed_paths=[OUTPUT_DIR])", "")
code = code.replace("server_port=7860)", "server_port=7860, share=True)")

# 2. Injetar a quantização 8-bit do torchao logo após a fusão do LoRA com identação exata
target = "        pipe.transformer.set_attention_backend(ATTENTION)"
quant_code = """        try:
            from torchao.quantization import quantize_, int8_weight_only
            print("[gen] Quantizando transformer para 8-bit (66GB -> 33GB)...", flush=True)
            quantize_(pipe.transformer, int8_weight_only())
            print("[gen] 8-Bit ativo! Transformer reduzido pela metade.", flush=True)
        except Exception as q_err:
            print(f"[gen] Aviso quantização: {q_err}", flush=True)

        pipe.transformer.set_attention_backend(ATTENTION)"""

code = code.replace(target, quant_code)

# 3. Garantir limpeza de cache CUDA no início de cada geração
gen_target = "    active_lora = h3_lora.set_active(PIPE.transformer, lora)"
gen_clean = """    import torch
    torch.cuda.empty_cache()
    active_lora = h3_lora.set_active(PIPE.transformer, lora)"""
code = code.replace(gen_target, gen_clean)

with open("app.py", "w") as f:
    f.write(code)

print("🚀 Carregando modelos com H3_PLACEMENT='offload' (Transformer e VAE alternam na GPU sem estourar)...")
print("👉 Como os pesos já estão em cache no disco, o carregamento levará cerca de 1 minuto.")
print("👉 Clique no link público 'Running on public URL: https://....gradio.live' que vai surgir abaixo:\n", flush=True)

cmd = "export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' && export H3_PLACEMENT='offload' && python3 app.py"
os.system(cmd)
