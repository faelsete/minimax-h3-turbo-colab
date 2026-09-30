import os
import sys
import time
import subprocess
from PIL import Image, ImageOps
import torch

print("=" * 70)
print("🎬 MiniMax-H3 Turbo — UGC Video Engine (Multi-Take Zero-Matrix)")
print("🎬 Personagem: Will Smith · Cenário: Beverly Hills Fountain")
print("🎬 Formato: UGC Selfie (9:16 Full · 768x1344) · 8 Passos de Denoise")
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

def keyframe(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path and os.path.exists(path) else None

num_frames = snap_frames(5) # 124 frames = 5.17s
print(f"🎯 Duração por Take: 5 segundos ({num_frames} frames)")

# Roteiros UGC detalhados
prompt_take1_raw = (
    "Cinematic vertical 9:16 authentic UGC selfie video recorded on a smartphone front-facing camera. Charismatic middle-aged Black man resembling Will Smith with a clean buzz-cut fade haircut, well-groomed goatee beard, warm expressive brown eyes, visible skin pores, natural facial asymmetry and realistic laugh lines. He is wearing a black knit crewneck t-shirt with a small dollar sign ($) emblem on the left chest. Framing is waist-up, holding his smartphone with an extended arm with natural organic handheld micro-shake. He stands on the sunny sidewalk in front of the iconic Beverly Hills water fountain at Beverly Gardens Park on a bright sunny California day, green palm trees and blue sky behind him. He smiles warmly into the camera, his lips immediately parting and mouth opening in active, expressive speech articulation with visible teeth as he speaks passionately directly into the lens: 'Because there's just such the most amazing opportunity like I've never seen before. You want to be the first millionaire in your family?' Authentic ambient sound of water splashing from the Beverly Hills fountain, light California breeze, and distant street traffic."
)

prompt_take2_raw = (
    "Cinematic vertical 9:16 authentic UGC selfie video recorded on a smartphone front-facing camera. The charismatic middle-aged man in black knit t-shirt with dollar sign ($) emblem continues holding his smartphone with extended arm, walking forward along the sidewalk and looking to the left and right to check street traffic as he steps to cross the street. He turns his gaze back directly into the selfie camera with an engaging smile, his lips parting and mouth opening with lively, expressive speech articulation with visible teeth as he speaks directly into the camera: 'Here are my three best tips to help you achieve it. Number one: if you have a phone and internet, that's all you need.' Natural handheld camera movement, sunny outdoor lighting, footsteps on pavement, breeze and distant street traffic."
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

print("\n🧠 Reescrevendo Take 1 (Modo Zero-Matrix t2va)...")
t0 = time.time()
try:
    refined_take1 = prompt_rewrite.refine_keyframe_prompt(
        cond_pipe,
        prompt_take1_raw,
        None,
        None,
        num_frames=num_frames,
        max_new_tokens=1000
    )
    print(f"✅ Prompt Take 1 reescrito em {time.time()-t0:.1f}s!")
except Exception as e:
    print(f"⚠️ Fallback Take 1: {e}")
    refined_take1 = (
        "[Shot 1] Live-action, authentic UGC selfie video in 9:16 vertical. A charismatic middle-aged Black man resembling Will Smith in black knit t-shirt with a dollar sign emblem holds a smartphone recording himself in front of the sunny Beverly Hills fountain. His lips part and his mouth moves actively with visible teeth as he speaks directly to the camera: <d>[English] Because there's just such the most amazing opportunity like I've never seen before. You want to be the first millionaire in your family?</d>\n\noverall_soundscape: Water splashing softly from the fountain, gentle breeze in the phone microphone, distant street traffic.\n\nnon_diegetic_music: N/A"
    )

# Sanitização de articulação labial ativa
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
    image=None,
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
)
prompt_embeds1 = state1.get("prompt_embeds").cpu()
text_token_tags1 = state1.get("text_token_tags").cpu()
print("✅ Embeddings Take 1 codificados com sucesso!")

print("\n🧠 Reescrevendo Take 2 (Modo Chaining i2va)...")
t0 = time.time()
try:
    refined_take2 = prompt_rewrite.refine_keyframe_prompt(
        cond_pipe,
        prompt_take2_raw,
        None,
        None,
        num_frames=num_frames,
        max_new_tokens=1000
    )
    print(f"✅ Prompt Take 2 reescrito em {time.time()-t0:.1f}s!")
except Exception as e:
    print(f"⚠️ Fallback Take 2: {e}")
    refined_take2 = (
        "[Shot 1] Live-action, authentic UGC selfie video in 9:16 vertical. The charismatic middle-aged man in black knit t-shirt walks forward and glances to the sides to cross the street, then turns back to the camera, lips parting with active mouth movement and visible teeth: <d>[English] Here are my three best tips to help you achieve it. Number one: if you have a phone and internet, that's all you need.</d>\n\noverall_soundscape: Footsteps on concrete, breeze, and distant street traffic.\n\nnon_diegetic_music: N/A"
    )

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
    image=None,
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
# ETAPA 3: RENDERIZAR TAKE 1 (Zero-Matrix t2va)
# =============================================================
print(f"\n🎬 [3/4] Renderizando Take 1 (Sem Matriz · 5s · 8 passos)...")
t0 = time.time()
state = gen_pipe(
    prompt_embeds=prompt_embeds1.to("cuda"),
    text_token_tags=text_token_tags1,
    image=None,
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
    num_frames=num_frames,
    num_inference_steps=8,
    generator=torch.Generator("cpu").manual_seed(101),
)

frames1 = state.get("videos")[0]
audio1 = state.get("audio")[0].cpu()
sr1 = state.get("sampling_rate", 24000)

path1 = "/content/take1_will.mp4"
encode_video(frames1, fps=FPS, output_path=path1, audio=audio1, audio_sample_rate=sr1)
print(f"✅ Take 1 concluído em {time.time()-t0:.1f}s: {path1}")

# Extração de frame ponte para o Take 2
bridge_frame = "/content/bridge_will_1.png"
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
    generator=torch.Generator("cpu").manual_seed(102),
)

frames2 = state.get("videos")[0]
audio2 = state.get("audio")[0].cpu()
sr2 = state.get("sampling_rate", 24000)

path2 = "/content/take2_will.mp4"
encode_video(frames2, fps=FPS, output_path=path2, audio=audio2, audio_sample_rate=sr2)
print(f"✅ Take 2 concluído em {time.time()-t0:.1f}s: {path2}")

# =============================================================
# ETAPA 5: MASTERIZAÇÃO FINAL (Concat sem re-encode)
# =============================================================
final_video = "/content/Will_Smith_UGC_Beverly_Hills_10s.mp4"
concat_txt = "/content/concat_will.txt"
with open(concat_txt, "w") as f:
    f.write(f"file '{path1}'\nfile '{path2}'\n")

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", "-movflags", "+faststart", final_video], check=True)
print(f"\n🎉 VÍDEO UGC FINAL MASTERIZADO: {final_video}")

subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size,bit_rate", "-of", "default=noprint_wrappers=1", final_video], check=True)

file_size = os.path.getsize(final_video) / (1024 * 1024)
print(f"📊 Tamanho final: {file_size:.2f} MB")
print("✅ CONCLUIDO_COM_SUCESSO")
