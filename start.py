import os
import sys

print("⚙️ Preparando MiniMax-H3 com otimizações para GPU A100...")
os.chdir("/content/minimax-h3")
os.system("git checkout app.py 2>/dev/null || true")

with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Configurar launch com link público (share=True) e permissão explícita de download de vídeos
old_launch = """if __name__ == "__main__":
    # allowed_paths: the /gradio_api/file= route only serves whitelisted directories.
    app.launch(show_error=True, allowed_paths=[OUTPUT_DIR])
    app.launch(server_name="0.0.0.0", server_port=7860)"""

new_launch = """if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=[OUTPUT_DIR], show_error=True)"""

code = code.replace(old_launch, new_launch)

# 2. Injetar a quantização 8-bit do torchao logo após o carregamento do LoRA
old_att = "        pipe.transformer.set_attention_backend(ATTENTION)"
quant_code = """        try:
            from torchao.quantization import quantize_, int8_weight_only
            print("[gen] Quantizando transformer para 8-bit (66GB -> 33GB)...", flush=True)
            quantize_(pipe.transformer, int8_weight_only())
            print("[gen] 8-Bit ativo! Mais de 30 GB de VRAM liberados na A100.", flush=True)
        except Exception as q_err:
            print(f"[gen] Aviso quantização: {q_err}", flush=True)

        pipe.transformer.set_attention_backend(ATTENTION)"""

code = code.replace(old_att, quant_code)

# 3. Injetar cópia automática de cada vídeo gerado para /content/ para acesso instantâneo
old_encode = "encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)"
new_encode = """encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)
    try:
        import shutil
        shutil.copy(path, os.path.join("/content", os.path.basename(path)))
        print(f"[gen] 🎬 Vídeo copiado automaticamente para /content/{os.path.basename(path)}", flush=True)
    except Exception:
        pass"""

code = code.replace(old_encode, new_encode)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)

print("🚀 Carregando modelos em 8-bit e iniciando o MiniMax-H3 Studio...")
print("👉 Como os pesos já estão em cache no disco, o carregamento levará cerca de 1 minuto.")
print("👉 Clique no link público 'Running on public URL: https://....gradio.live' que vai surgir abaixo:\n", flush=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["H3_PLACEMENT"] = "offload"
os.system("export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' && export H3_PLACEMENT='offload' && python3 app.py")
