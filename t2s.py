import sys
import torch
import scipy.io.wavfile
import numpy as np
from transformers import VitsModel, VitsTokenizer
import soundfile as sf

if len(sys.argv) < 2:
    print("Usage: python t2s.py <text> [output] [lang]")
    print("  text   : 변환할 텍스트")
    print("  output : 출력 파일명 (기본값: output.wav)")
    print("  lang   : 언어 ko/en (기본값: ko)")
    print("")
    print("Examples:")
    print("  python t2s.py '안녕하세요'")
    print("  python t2s.py 'Hello world' hello.wav en")
    sys.exit(1)

text   = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
lang   = sys.argv[3] if len(sys.argv) > 3 else "ko"

model_id = "facebook/mms-tts-kor" if lang == "ko" else "facebook/mms-tts-eng"

print(f"언어: {lang} | 모델: {model_id}")
print(f"생성 중: {text}")

tokenizer = VitsTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id)

sentences = [s.strip() for s in text.replace(".", ".|").replace("!", "!|").replace("?", "?|").split("|") if s.strip()]

all_audio = []
for sent in sentences:
    inputs = tokenizer(sent, return_tensors="pt")
    with torch.no_grad():
        output_audio = model(**inputs).waveform
    all_audio.append(output_audio.squeeze().numpy())

audio = np.concatenate(all_audio)
audio_int16 = (audio * 32767).astype(np.int16)

scipy.io.wavfile.write(output, rate=model.config.sampling_rate, data=audio_int16)
print(f"저장 완료: {output}")
print(f"재생: afplay {output}")
