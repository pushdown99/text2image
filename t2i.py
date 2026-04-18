import sys
import torch
import math
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python t2i.py <prompt> [output] [ref_image] [strength]")
    print("  prompt     : 생성할 이미지 설명")
    print("  output     : 출력 파일명 (기본값: output.png)")
    print("  ref_image  : 레퍼런스 이미지 경로 (img2img 모드)")
    print("  strength   : 변형 강도 0.0~1.0 (기본값: 0.5)")
    print("")
    print("Examples:")
    print("  python t2i.py 'a cat on a rooftop'")
    print("  python t2i.py 'oil painting style' result.png ref.jpg 0.7")
    sys.exit(1)

prompt    = sys.argv[1]
output    = sys.argv[2] if len(sys.argv) > 2 else "output.png"
ref_image = sys.argv[3] if len(sys.argv) > 3 else None
strength  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

NUM_STEPS = 4

print(f"생성 중: {prompt}")

def prepare_image(path, base=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w = base
        new_h = round(h * base / w / 8) * 8
    else:
        new_h = base
        new_w = round(w * base / h / 8) * 8
    img = img.resize((new_w, new_h), Image.LANCZOS)
    print(f"원본: {w}x{h} → 리사이즈: {new_w}x{new_h}")
    return img, new_w, new_h

def safe_strength(strength, num_steps):
    min_strength = math.ceil(1 / num_steps * 100) / 100
    if strength < min_strength:
        print(f"⚠️  strength {strength} → {min_strength} 로 자동 보정")
        return min_strength
    return strength

if ref_image:
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("mps")
    init_image, w, h = prepare_image(ref_image)
    adjusted_strength = safe_strength(strength, NUM_STEPS)
    image = pipe(
        prompt,
        image=init_image,
        num_inference_steps=NUM_STEPS,
        guidance_scale=0.0,
        strength=adjusted_strength,
        width=w,
        height=h,
    ).images[0]
else:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("mps")
    image = pipe(
        prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=0.0,
        width=512,
        height=512,
    ).images[0]

image.save(output)
print(f"저장 완료: {output}")
print(f"열기: open {output}")
