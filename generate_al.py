import os
import sys
import time
import subprocess
from PIL import Image, ImageOps
import torch

print("=" * 60)
print("🎬 MiniMax-H3 Turbo — Pipeline de Alta Fidelidade (10/10)")
print("🎬 Modelo: Adriana Lima (al 1.jpg) · 5 Segundos · 9:16 Full")
print("🎬 Context-IR Prompt Rewrite ATIVO · 8 Passos de Denoise")
print("=" * 60)

os.chdir("/content/minimax-h3")
sys.path.insert(0, "/content/minimax-h3")
sys.path.insert(0, "/content/repo")

import prompt_rewrite
from h3_split_blocks import MiniMaxH3ConditionerBlocks, MiniMaxH3GeneratorBlocks
from diffusers.utils import encode_video
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
import h3_lora

MiniMaxH3ModularPipeline.min_duration = property(lambda self: 2.0)

CANVAS = "768x1344 · 9:16 full"
HEIGHT, WIDTH = 1344, 768
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5

def snap_frames(seconds: float) -> int:
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames

def prepare_keyframe(src_path: str, dst_path: str) -> str:
    img = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
    target = WIDTH / HEIGHT # 768 / 1344
    if abs(img.width / img.height - target) > 1e-3:
        if img.width / img.height > target:
            new_w = int(img.height * target)
            left = (img.width - new_w) // 2
            img = img.crop((left, 0, left + new_w, img.height))
        else:
            new_h = int(img.width / target)
            top = (img.height - new_h) // 2
            img = img.crop((0, top, img.width, top + new_h))
    img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    img.save(dst_path, quality=95)
    print(f"📸 Keyframe preparado: {dst_path} ({WIDTH}x{HEIGHT})")
    return dst_path

def keyframe(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path else None

model_src = "/content/al_model.jpg"
if not os.path.exists(model_src):
    model_src = "/content/repo/al 1.jpg"

model_ready = "/content/al_keyframe_768x1344.jpg"
prepare_keyframe(model_src, model_ready)

raw_user_prompt = (
    "Cinematic vertical 9:16 video. The gorgeous supermodel woman in black evening gown and fur stole walks forward directly toward the camera, stepping closer and closer until her face fills the frame in an intimate, tight close-up shot right in front of the lens. Her expression is dynamic and lively: her lips immediately part and her mouth opens in active speech, her jaw and lips articulating each word expressively with visible teeth and lively lip sync as she speaks directly and warmly into the camera: 'Olá! Que prazer enorme estar bem pertinho de você esta noite!' The camera remains stationary in front of her as she approaches close to the lens. High-fashion glamour, cinematic lighting, photorealistic skin and hair."
)

num_frames = snap_frames(5) # 124 frames
print(f"🎯 Duração: 5 segundos ({num_frames} frames)")

# =============================================================
# ETAPA 1: CONDITIONING LOCAL COM PROMPT REWRITE (Qwen3-VL 33B)
# =============================================================
print("\n🔤 [1/3] Inicializando Conditioner Local (Qwen3-VL na A100)...")
cond_blocks = MiniMaxH3ConditionerBlocks()
cond_pipe = cond_blocks.init_pipeline("MiniMaxAI/MiniMax-H3")
cond_pipe.load_components(dtype=torch.bfloat16)
cond_pipe.to("cuda")
print("✅ Conditioner carregado na GPU A100!")

print("\n🧠 Executando Context-IR Prompt Rewrite (a mágica do Will Smith)...")
t_rw = time.time()
try:
    refined_prompt = prompt_rewrite.refine_keyframe_prompt(
        cond_pipe,
        raw_user_prompt,
        keyframe(model_ready),
        None,
        num_frames=num_frames,
        max_new_tokens=1000
    )
    print(f"✅ Prompt reescrito com sucesso em {time.time()-t_rw:.1f}s!")
except Exception as e:
    print(f"⚠️ Erro no rewrite ({e}), utilizando prompt com tags manuais...")
    refined_prompt = (
        f"[Shot 1] The elegant supermodel woman in black gown and fur stole walks forward directly toward the camera, stepping closer until her face fills the screen in a tight close-up. Her lips immediately part and her mouth moves actively with visible teeth and dynamic lip articulation as she speaks directly to the camera: <d>[Portuguese] Olá! Que prazer enorme estar bem pertinho de você esta noite!</d> Warm golden evening lighting, cinematic bokeh lights, 24fps realism."
    )

# Garantir câmera estática para aproximação e lábios em movimento ativo
refined_prompt = refined_prompt.replace("tracks backward with small amplitude at slow speed", "remains static as she steps closer into an intimate tight close-up")
refined_prompt = refined_prompt.replace("tracks backward with small amplitude", "remains static as she steps closer into an intimate tight close-up")
refined_prompt = refined_prompt.replace("tracks backward", "holds static while she walks directly toward the lens")
refined_prompt = refined_prompt.replace("maintaining warm, confident eye contact with a captivating smile", "stepping forward into an intimate tight close-up, her lips parting and mouth opening with expressive speech articulation and visible teeth")
refined_prompt = refined_prompt.replace("lips remain closed", "lips part and mouth articulates speech actively")
refined_prompt = refined_prompt.replace("pose preserved from the image", "facial features and elegant attire preserved from the image while her pose develops into active forward movement and animated speech")

if "<d>" in refined_prompt and "lips part" not in refined_prompt.lower():
    refined_prompt = refined_prompt.replace(
        "speaks directly to the camera:",
        "her lips parting and mouth opening in active, expressive speech articulation with visible teeth as she speaks directly to the camera:"
    )

print("\n--- PROMPT ESTRUTURADO FINAL (CONTEXT-IR) ---")
print(refined_prompt)
print("---------------------------------------------\n")

print("📝 Codificando embeddings com o prompt refinado...")
t_enc = time.time()
state = cond_pipe(
    prompt=refined_prompt,
    image=keyframe(model_ready),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
)
prompt_embeds = state.get("prompt_embeds").cpu()
text_token_tags = state.get("text_token_tags").cpu()
print(f"✅ Embeddings codificados em {time.time()-t_enc:.1f}s!")

# Liberar VRAM completamente
del cond_pipe, cond_blocks, state
torch.cuda.empty_cache()
print("🧹 VRAM liberada para a etapa de denoise!")

# =============================================================
# ETAPA 2: GENERATOR (Transformer 8-bit + Turbo LoRA Larry)
# =============================================================
print("\n🎬 [2/3] Carregando Generator MiniMax-H3 na A100...")
gen_blocks = MiniMaxH3GeneratorBlocks()
gen_pipe = gen_blocks.init_pipeline("MiniMaxAI/MiniMax-H3", collection="h3")
gen_pipe.load_components(dtype=torch.bfloat16)

# LoRA Turbo Larry
lora_status = h3_lora.apply_lora(gen_pipe.transformer)
print(f"[gen] {lora_status}")

# Quantização 8-bit torchao
try:
    from torchao.quantization import quantize_, int8_weight_only
    print("[gen] Quantizando transformer para 8-bit...")
    quantize_(gen_pipe.transformer, int8_weight_only())
    print("[gen] 8-Bit ativo com sucesso!")
except Exception as e:
    print(f"[gen] Aviso quantização: {e}")

gen_pipe.transformer.set_attention_backend("_native_cudnn")
gen_pipe.to("cuda")
print("✅ Generator pronto na A100!")

# =============================================================
# ETAPA 3: RENDERIZAR VÍDEO (8 PASSOS · 5 SEGUNDOS)
# =============================================================
print(f"\n🎬 [3/3] Renderizando Adriana Lima ({num_frames} frames · 8 passos)...")
t_gen = time.time()
state = gen_pipe(
    prompt_embeds=prompt_embeds.to("cuda"),
    text_token_tags=text_token_tags,
    image=keyframe(model_ready),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
    num_frames=num_frames,
    num_inference_steps=8,
    generator=torch.Generator("cpu").manual_seed(42),
)

frames = state.get("videos")[0]
audio = state.get("audio")[0].cpu()
sr = state.get("sampling_rate", 24000)

output_video = "/content/Adriana_Lima_Walking_Speech_5s.mp4"
encode_video(frames, fps=FPS, output_path=output_video, audio=audio, audio_sample_rate=sr)
print(f"✅ Vídeo gerado em {time.time()-t_gen:.1f}s: {output_video}")

# Verificação via ffprobe
subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size,bit_rate", "-of", "default=noprint_wrappers=1", output_video], check=True)

file_size = os.path.getsize(output_video) / (1024 * 1024)
print(f"\n🎉 SUCESSO TOTAL! Arquivo final: {output_video} ({file_size:.2f} MB)")
print("✅ CONCLUIDO_COM_SUCESSO")
