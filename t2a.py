import sys
import torch
import scipy.io.wavfile
import numpy as np
from transformers import pipeline

if len(sys.argv) < 2:
    print("Usage: python t2a.py <prompt> [output] [duration]")
    print("  prompt   : 생성할 음악 설명")
    print("  output   : 출력 파일명 (기본값: output.wav)")
    print("  duration : 토큰 수, 256=약5초 (기본값: 256)")
    print("")
    print("Examples:")
    print("  python t2a.py 'relaxing lo-fi hip hop beats'")
    print("  python t2a.py 'epic orchestral music' epic.wav 512")
    sys.exit(1)

prompt   = sys.argv[1]
output   = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
duration = int(sys.argv[3]) if len(sys.argv) > 3 else 256

print(f"생성 중: {prompt}")

synthesiser = pipeline(
    "text-to-audio",
    "facebook/musicgen-small",
    device="mps" if torch.backends.mps.is_available() else "cpu"
)

music = synthesiser(
    prompt,
    forward_params={"do_sample": True, "max_new_tokens": duration}
)

audio = music["audio"]
if audio.ndim == 2:
    audio = audio[0]

scipy.io.wavfile.write(output, rate=music["sampling_rate"], data=audio)
print(f"저장 완료: {output}")
print(f"재생: afplay {output}")
