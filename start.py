import os
import sys

print("⚙️ Preparando MiniMax-H3 com quantização 8-bit...")
os.chdir("/content/minimax-h3")
os.system("git checkout app.py 2>/dev/null || true")

with open("app.py", "r") as f:
    code = f.read()

# 1. Configurar launch com link público (share=True) e permissão de arquivos
code = code.replace("app.launch(show_error=True, allowed_paths=[OUTPUT_DIR])", "")
code = code.replace("server_port=7860)", "server_port=7860, share=True, allowed_paths=[OUTPUT_DIR])")

# 2. Injetar a quantização 8-bit do torchao logo após a fusão do LoRA com identação exata
target = "        pipe.transformer.set_attention_backend(ATTENTION)"
quant_code = """        try:
            from torchao.quantization import quantize_, int8_weight_only
            print("[gen] Quantizando transformer para 8-bit (38.5GB -> 19.2GB)...", flush=True)
            quantize_(pipe.transformer, int8_weight_only())
            print("[gen] 8-Bit ativo! Mais de 20 GB de VRAM liberados na A100.", flush=True)
        except Exception as q_err:
            print(f"[gen] Aviso quantização: {q_err}", flush=True)

        pipe.transformer.set_attention_backend(ATTENTION)"""

code = code.replace(target, quant_code)

with open("app.py", "w") as f:
    f.write(code)

print("🚀 Carregando modelos em 8-bit e iniciando o MiniMax-H3 Studio...")
print("👉 Como os pesos já estão em cache no disco, o carregamento levará cerca de 1 minuto.")
print("👉 Clique no link público 'Running on public URL: https://....gradio.live' que vai surgir abaixo:\n", flush=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["H3_PLACEMENT"] = "lazy"
os.system("export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' && export H3_PLACEMENT='lazy' && python3 app.py")
