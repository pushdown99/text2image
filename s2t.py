import sys
import torch
import whisper

if len(sys.argv) < 2:
    print("Usage: python s2t.py <audio> [model] [output]")
    print("  audio  : 변환할 오디오 파일 (wav, mp3 등)")
    print("  model  : 모델 크기 tiny/base/small/medium/large (기본값: base)")
    print("  output : 출력 파일명 (기본값: 입력파일명.txt)")
    print("")
    print("Examples:")
    print("  python s2t.py hello.wav")
    print("  python s2t.py recording.mp3 small")
    print("  python s2t.py recording.mp3 medium result.txt")
    sys.exit(1)

audio_file  = sys.argv[1]
model_size  = sys.argv[2] if len(sys.argv) > 2 else "base"
output_file = sys.argv[3] if len(sys.argv) > 3 else audio_file.rsplit(".", 1)[0] + ".txt"

print(f"변환 중: {audio_file} (모델: {model_size})")
print(f"출력 파일: {output_file}")

model = whisper.load_model(model_size)

if torch.backends.mps.is_available():
    model = model.to("mps")
    print("MPS 가속 사용 중")
else:
    print("CPU 사용 중")

result = model.transcribe(audio_file, language="ko", fp16=False)

print(f"\n변환 결과:\n{result['text']}")

with open(output_file, "w") as f:
    f.write(result["text"])
print(f"\n저장 완료: {output_file}")
