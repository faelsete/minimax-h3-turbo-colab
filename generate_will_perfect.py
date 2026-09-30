import os
import sys
import time
import subprocess
from PIL import Image, ImageOps
import torch

print("=" * 70)
print("🎬 MiniMax-H3 Turbo — Pipeline 10/10 Will Smith Matriz 2K")
print("🎬 Modo: i2va Ancorado na Matriz Real + Encadeamento Frame Bridge")
print("🎬 8 Passos · 9:16 Full (768x1344) · Cadência de Fala Cirúrgica")
print("=" * 70)

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
    target = WIDTH / HEIGHT
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
    print(f"📸 Keyframe 9:16 preparado: {dst_path} ({WIDTH}x{HEIGHT})")
    return dst_path

def keyframe(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path and os.path.exists(path) else None

model_src = "/content/will_model.jpg"
model_ready = "/content/will_keyframe_768x1344.jpg"
prepare_keyframe(model_src, model_ready)

num_frames = snap_frames(5) # 124 frames = 5.17s
print(f"🎯 Duração por Take: 5 segundos ({num_frames} frames)")

# Roteiros com cadência milimetricamente mensurada
prompt_take1_raw = (
    "Authentic UGC smartphone selfie video in vertical 9:16. The charismatic middle-aged man shown in <Picture 1> holds his smartphone at arm's length against the cloudy blue sky, wearing his black puffer jacket over a black shirt. He speaks passionately with active, lively mouth movement and clear lip sync, teeth visible, smiling warmly directly into the camera lens: 'Because there's just such the most amazing opportunity like I've never seen before.' Natural daylight, realistic skin texture and stubble, subtle handheld camera micro-movement, gentle breeze in the phone microphone."
)

prompt_take2_raw = (
    "Authentic UGC smartphone selfie video in vertical 9:16. The charismatic middle-aged man in black jacket continues holding his smartphone, walking forward with a subtle head glance to the side before re-engaging the camera with a confident smile. His lips part dynamically with active speech articulation and visible teeth as he speaks directly into the lens: 'You want to be the first millionaire in your family? Here are my three best tips.' Natural handheld camera motion, outdoor daylight, soft breeze and footsteps."
)

# =============================================================
# ETAPA 1: CONDITIONING LOCAL COM PROMPT REWRITE (Qwen3-VL 33B)
# =============================================================
print("\n🔤 [1/4] Inicializando Conditioner Local (Qwen3-VL na A100)...")
cond_blocks = MiniMaxH3ConditionerBlocks()
cond_pipe = cond_blocks.init_pipeline("MiniMaxAI/MiniMax-H3")
cond_pipe.load_components(dtype=torch.bfloat16)
cond_pipe.to("cuda")
print("✅ Conditioner carregado na GPU A100!")

print("\n🧠 Reescrevendo Take 1 (Context-IR i2va com Matriz 2K)...")
t0 = time.time()
try:
    refined_take1 = prompt_rewrite.refine_keyframe_prompt(
        cond_pipe,
        prompt_take1_raw,
        keyframe(model_ready),
        None,
        num_frames=num_frames,
        max_new_tokens=1000
    )
    print(f"✅ Prompt Take 1 reescrito em {time.time()-t0:.1f}s!")
except Exception as e:
    print(f"⚠️ Fallback Take 1: {e}")
    refined_take1 = (
        "[Shot 1] Authentic UGC selfie video in vertical 9:16. The man in <Picture 1> holds his smartphone against the cloudy sky. His lips part and mouth moves actively with visible teeth: <d>[English] Because there's just such the most amazing opportunity like I've never seen before.</d>\n\noverall_soundscape: Soft breeze blowing against phone microphone.\n\nnon_diegetic_music: N/A"
    )

# Sanitização de articulação labial e enquadramento
refined_take1 = refined_take1.replace("lips remain closed", "lips part and mouth articulates speech actively")
refined_take1 = refined_take1.replace("pose preserved from the image", "facial features, clothing and cloudy sky preserved from the image while his expression becomes animated with lively speech")
if "<d>" in refined_take1 and "lips part" not in refined_take1.lower():
    refined_take1 = refined_take1.replace(
        "speaks directly to the camera:",
        "his lips parting and mouth opening with active speech articulation and visible teeth as he speaks directly to the camera:"
    )

print("\n--- PROMPT TAKE 1 FINAL ---")
print(refined_take1)
print("---------------------------\n")

print("📝 Codificando embeddings do Take 1...")
state1 = cond_pipe(
    prompt=refined_take1,
    image=keyframe(model_ready),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
)
prompt_embeds1 = state1.get("prompt_embeds").cpu()
text_token_tags1 = state1.get("text_token_tags").cpu()
print("✅ Embeddings Take 1 codificados com sucesso!")

print("\n🧠 Reescrevendo Take 2 (Context-IR Chaining)...")
t0 = time.time()
try:
    refined_take2 = prompt_rewrite.refine_keyframe_prompt(
        cond_pipe,
        prompt_take2_raw,
        keyframe(model_ready),
        None,
        num_frames=num_frames,
        max_new_tokens=1000
    )
    print(f"✅ Prompt Take 2 reescrito em {time.time()-t0:.1f}s!")
except Exception as e:
    print(f"⚠️ Fallback Take 2: {e}")
    refined_take2 = (
        "[Shot 1] Authentic UGC selfie video in vertical 9:16. The man continues walking forward holding his phone. His lips part with active speech articulation and visible teeth: <d>[English] You want to be the first millionaire in your family? Here are my three best tips.</d>\n\noverall_soundscape: Soft breeze and distant footsteps.\n\nnon_diegetic_music: N/A"
    )

refined_take2 = refined_take2.replace("lips remain closed", "lips part and mouth articulates speech actively")
refined_take2 = refined_take2.replace("pose preserved from the image", "facial features, clothing and background preserved while his expression becomes animated with lively speech")
if "<d>" in refined_take2 and "lips part" not in refined_take2.lower():
    refined_take2 = refined_take2.replace(
        "speaks directly to the camera:",
        "his lips parting and mouth opening with active speech articulation and visible teeth as he speaks directly to the camera:"
    )

print("\n--- PROMPT TAKE 2 FINAL ---")
print(refined_take2)
print("---------------------------\n")

print("📝 Codificando embeddings do Take 2...")
state2 = cond_pipe(
    prompt=refined_take2,
    image=keyframe(model_ready),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
)
prompt_embeds2 = state2.get("prompt_embeds").cpu()
text_token_tags2 = state2.get("text_token_tags").cpu()
print("✅ Embeddings Take 2 codificados com sucesso!")

# Liberar VRAM completamente
del cond_pipe, cond_blocks, state1, state2
torch.cuda.empty_cache()
print("🧹 VRAM liberada para a etapa de denoise!")

# =============================================================
# ETAPA 2: GENERATOR (Transformer 8-bit + LoRA Larry)
# =============================================================
print("\n🎬 [2/4] Carregando Generator MiniMax-H3 na A100...")
gen_blocks = MiniMaxH3GeneratorBlocks()
gen_pipe = gen_blocks.init_pipeline("MiniMaxAI/MiniMax-H3", collection="h3")
gen_pipe.load_components(dtype=torch.bfloat16)

lora_status = h3_lora.apply_lora(gen_pipe.transformer)
print(f"[gen] {lora_status}")

try:
    from torchao.quantization import quantize_, int8_weight_only
    print("[gen] Quantizando transformer para 8-bit...")
    quantize_(gen_pipe.transformer, int8_weight_only())
    print("[gen] 8-Bit ativo!")
except Exception as e:
    print(f"[gen] Aviso quant: {e}")

gen_pipe.transformer.set_attention_backend("_native_cudnn")
gen_pipe.to("cuda")
print("✅ Generator pronto na A100!")

# =============================================================
# ETAPA 3: RENDERIZAR TAKE 1 (Ancorado na Matriz 2K)
# =============================================================
print(f"\n🎬 [3/4] Renderizando Take 1 (Matriz 2K · 5s · 8 passos)...")
t0 = time.time()
state = gen_pipe(
    prompt_embeds=prompt_embeds1.to("cuda"),
    text_token_tags=text_token_tags1,
    image=keyframe(model_ready),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
    num_frames=num_frames,
    num_inference_steps=8,
    generator=torch.Generator("cpu").manual_seed(42),
)

frames1 = state.get("videos")[0]
audio1 = state.get("audio")[0].cpu()
sr1 = state.get("sampling_rate", 24000)

path1 = "/content/take1_will_perfect.mp4"
encode_video(frames1, fps=FPS, output_path=path1, audio=audio1, audio_sample_rate=sr1)
print(f"✅ Take 1 concluído em {time.time()-t0:.1f}s: {path1}")

# Extração de frame ponte para o Take 2
bridge_frame = "/content/bridge_will_perfect.png"
subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", path1, "-update", "1", "-q:v", "1", bridge_frame], check=True)
print(f"🌉 Frame ponte extraído para ancoragem visual: {bridge_frame}")

# =============================================================
# ETAPA 4: RENDERIZAR TAKE 2 (Chained i2va via Bridge Frame)
# =============================================================
print(f"\n🎬 [4/4] Renderizando Take 2 (Ancorado no Bridge Frame · 5s · 8 passos)...")
t0 = time.time()
state = gen_pipe(
    prompt_embeds=prompt_embeds2.to("cuda"),
    text_token_tags=text_token_tags2,
    image=keyframe(bridge_frame),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
    num_frames=num_frames,
    num_inference_steps=8,
    generator=torch.Generator("cpu").manual_seed(43),
)

frames2 = state.get("videos")[0]
audio2 = state.get("audio")[0].cpu()
sr2 = state.get("sampling_rate", 24000)

path2 = "/content/take2_will_perfect.mp4"
encode_video(frames2, fps=FPS, output_path=path2, audio=audio2, audio_sample_rate=sr2)
print(f"✅ Take 2 concluído em {time.time()-t0:.1f}s: {path2}")

# =============================================================
# ETAPA 5: MASTERIZAÇÃO FINAL (Concat sem re-encode)
# =============================================================
final_video = "/content/Will_Smith_UGC_Perfeito_10s.mp4"
concat_txt = "/content/concat_will_perfect.txt"
with open(concat_txt, "w") as f:
    f.write(f"file '{path1}'\nfile '{path2}'\n")

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", "-movflags", "+faststart", final_video], check=True)
print(f"\n🎉 VÍDEO UGC FINAL MASTERIZADO: {final_video}")

subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size,bit_rate", "-of", "default=noprint_wrappers=1", final_video], check=True)

file_size = os.path.getsize(final_video) / (1024 * 1024)
print(f"📊 Tamanho final: {file_size:.2f} MB")
print("✅ CONCLUIDO_COM_SUCESSO")
