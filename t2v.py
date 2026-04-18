import sys
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

if len(sys.argv) < 2:
    print("Usage: python t2v.py <prompt> [output]")
    print("  prompt : 생성할 영상 설명")
    print("  output : 출력 파일명 (기본값: output.mp4)")
    print("")
    print("Examples:")
    print("  python t2v.py 'a panda eating bamboo'")
    print("  python t2v.py 'waves on a beach' ocean.mp4")
    sys.exit(1)

prompt = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

print(f"생성 중: {prompt}")

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

frames = pipe(
    prompt,
    num_frames=17,
    width=512,
    height=384,
    num_inference_steps=20,
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, output, fps=8)
print(f"저장 완료: {output}")
print(f"재생: open {output}")
