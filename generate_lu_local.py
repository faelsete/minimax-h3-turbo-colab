import os
import sys
import time
import subprocess
from PIL import Image, ImageOps
import torch

print("=" * 60)
print("🎬 MiniMax-H3 Turbo — Pipeline 100% Autônomo e Local na A100")
print("🎬 Campanha: Lú (Smart Digital) · 14 Segundos · 9:16 Vertical Full")
print("=" * 60)

os.chdir("/content/minimax-h3")
sys.path.insert(0, "/content/minimax-h3")

from h3_split_blocks import MiniMaxH3ConditionerBlocks, MiniMaxH3GeneratorBlocks
from diffusers.utils import encode_video
from diffusers import ComponentsManager
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
import h3_lora

# Permitir durações curtas / customizadas
MiniMaxH3ModularPipeline.min_duration = property(lambda self: 2.0)

CANVAS = "768x1344 · 9:16 full"
HEIGHT, WIDTH = 1344, 768
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5

def snap_frames(seconds: float) -> int:
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames

def _fit_keyframe(image_path: str) -> str:
    img = Image.open(image_path)
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
        img.save(image_path)
    return image_path

def keyframe(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path else None

model_img = "/content/lu_model.jpg"
if not os.path.exists(model_img):
    model_img = "/content/repo/assets/lu_model.jpg"
if not os.path.exists(model_img):
    model_img = "/content/Woman_in_business_attire_2K_20260930023453.jpg"

_fit_keyframe(model_img)
print(f"📸 Imagem modelo pronta: {model_img} ({WIDTH}x{HEIGHT})")

prompt_take1 = (
    'Documentary 8k vertical video 9:16. Elegant professional blonde woman named Lú wearing a stylish charcoal grey blazer and silk blouse, facing camera directly, warm confident smile, natural eye contact. She speaks with perfect lip sync: "Oi, oi, pessoal! Eu sou a Lú, sou uma inteligência artificial criada pelo Rafael Fernandes da Smart Digital." Natural subtle speaking head tilt, lifelike facial expressions, smooth breathing movement. Soft natural studio window lighting, 24fps cinematic realism, clean video.'
)

prompt_take2 = (
    'Documentary 8k vertical video 9:16. Elegant professional blonde woman named Lú in charcoal grey blazer, looking into camera, engaging and articulate. She speaks with seamless lip sync: "Soluções digitais para empresários que querem ter resultados de verdade!" Energetic confident smile, subtle expressive gesture, continuous natural movement from keyframe. Crisp 24fps cinematic lighting, clean video.'
)

# =============================================================
# ETAPA 1: CONDITIONING LOCAL (Qwen3-VL 33B na A100)
# =============================================================
print("\n🔤 [1/4] Inicializando Conditioner Local (Qwen3-VL)...")
cond_blocks = MiniMaxH3ConditionerBlocks()
cond_pipe = cond_blocks.init_pipeline("MiniMaxAI/MiniMax-H3")
cond_pipe.load_components(dtype=torch.bfloat16)
cond_pipe.to("cuda")
print("✅ Conditioner carregado na GPU A100!")

print("📝 Codificando embeddings do Take 1...")
t0 = time.time()
state1 = cond_pipe(
    prompt=prompt_take1,
    image=keyframe(model_img),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
)
prompt_embeds1 = state1.get("prompt_embeds").cpu()
text_token_tags1 = state1.get("text_token_tags").cpu()
print(f"✅ Take 1 codificado em {time.time()-t0:.1f}s!")

print("📝 Codificando embeddings do Take 2...")
t0 = time.time()
state2 = cond_pipe(
    prompt=prompt_take2,
    image=keyframe(model_img),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
)
prompt_embeds2 = state2.get("prompt_embeds").cpu()
text_token_tags2 = state2.get("text_token_tags").cpu()
print(f"✅ Take 2 codificado em {time.time()-t0:.1f}s!")

# Liberar VRAM completamente do Text Encoder
del cond_pipe, cond_blocks, state1, state2
torch.cuda.empty_cache()
print("🧹 Memória VRAM liberada para a etapa de denoise!")

# =============================================================
# ETAPA 2: GENERATOR (Transformer 8-bit + VAEs + Turbo LoRA)
# =============================================================
print("\n🎬 [2/4] Carregando Generator MiniMax-H3 na A100...")
gen_blocks = MiniMaxH3GeneratorBlocks()
gen_pipe = gen_blocks.init_pipeline("MiniMaxAI/MiniMax-H3", collection="h3")
gen_pipe.load_components(dtype=torch.bfloat16)

# Aplicar Turbo LoRA Larry
LORA_STATUS = h3_lora.apply_lora(gen_pipe.transformer)
print(f"[gen] {LORA_STATUS}")

# Quantização 8-bit torchao para economizar VRAM
try:
    from torchao.quantization import quantize_, int8_weight_only
    print("[gen] Quantizando transformer para 8-bit (66GB -> 33GB)...")
    quantize_(gen_pipe.transformer, int8_weight_only())
    print("[gen] 8-Bit ativo com sucesso!")
except Exception as e:
    print(f"[gen] Aviso quant: {e}")

gen_pipe.transformer.set_attention_backend("_native_cudnn")
gen_pipe.to("cuda")
print("✅ Generator pronto na A100!")

# =============================================================
# ETAPA 3: RENDERIZAR TAKE 1 (7s · 9:16 Full)
# =============================================================
num_frames = snap_frames(7)
print(f"\n🎬 [3/4] Renderizando Take 1 ({num_frames} frames · 7 segundos)...")
t0 = time.time()
state = gen_pipe(
    prompt_embeds=prompt_embeds1.to("cuda"),
    text_token_tags=text_token_tags1,
    image=keyframe(model_img),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
    num_frames=num_frames,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames1 = state.get("videos")[0]
audio1 = state.get("audio")[0].cpu()
sr1 = state.get("sampling_rate", 24000)

path1 = "/content/take1.mp4"
encode_video(frames1, fps=FPS, output_path=path1, audio=audio1, audio_sample_rate=sr1)
print(f"✅ Take 1 concluído em {time.time()-t0:.1f}s: {path1}")

# Bridge frame do Take 1
bridge_frame = "/content/bridge_frame.png"
subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", path1, "-update", "1", "-q:v", "1", bridge_frame], check=True)
print(f"🌉 Frame ponte extraído para continuidade perfeita: {bridge_frame}")

# Renderizar Take 2 (7s · 9:16 Full)
print(f"\n🎬 Renderizando Take 2 ({num_frames} frames · 7 segundos)...")
t0 = time.time()
state = gen_pipe(
    prompt_embeds=prompt_embeds2.to("cuda"),
    text_token_tags=text_token_tags2,
    image=keyframe(bridge_frame),
    last_image=None,
    height=HEIGHT,
    width=WIDTH,
    num_frames=num_frames,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(43),
)
frames2 = state.get("videos")[0]
audio2 = state.get("audio")[0].cpu()
sr2 = state.get("sampling_rate", 24000)

path2 = "/content/take2.mp4"
encode_video(frames2, fps=FPS, output_path=path2, audio=audio2, audio_sample_rate=sr2)
print(f"✅ Take 2 concluído em {time.time()-t0:.1f}s: {path2}")

# =============================================================
# ETAPA 4: MASTERIZAÇÃO FINAL (14s)
# =============================================================
print("\n🎬 [4/4] Masterizando vídeo final de 14 segundos via FFmpeg...")
final_video = "/content/Lu_Smart_Digital_Final_14s.mp4"
concat_txt = "/content/concat.txt"
with open(concat_txt, "w") as f:
    f.write(f"file '{path1}'\nfile '{path2}'\n")

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", "-movflags", "+faststart", final_video], check=True)
print(f"\n🎉 VÍDEO FINAL DE 14 SEGUNDOS PRONTO: {final_video}")

file_size = os.path.getsize(final_video) / (1024 * 1024)
print(f"📊 Tamanho do arquivo: {file_size:.2f} MB")
print("✅ CONCLUIDO_COM_SUCESSO")
