# AI Tools on Mac mini M4 16GB
> 작성일: 2026-04-18
> 다른 Claude가 이어서 작업할 수 있도록 오늘 작업 내용 정리

---

## 환경 정보

| 항목 | 내용 |
|------|------|
| 기기 | Mac mini M4 |
| RAM | 16GB (RAM+VRAM 공유, 실사용 ~10~12GB) |
| Python | 3.11.15 (Homebrew) |
| 작업 경로 | ~/.openclaw/workspace/ai/ |
| 가상환경 | ~/.openclaw/workspace/ai/venv/ |

---

## 환경 세팅

```bash
# 1. 시스템 의존성
brew install python@3.11 pkg-config ffmpeg

# 2. 폴더 및 venv
mkdir -p ~/.openclaw/workspace/ai
cd ~/.openclaw/workspace/ai
python3.11 -m venv venv
source venv/bin/activate

# 3. 패키지
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate sentencepiece protobuf
pip install "imageio[ffmpeg]" scipy soundfile uroman
pip install openai-whisper

# 매번 터미널 열 때
cd ~/.openclaw/workspace/ai && source venv/bin/activate
```

> audiocraft는 torch 버전 충돌(요구 2.1.0 vs 설치 2.11.0)로 설치 실패.
> MusicGen은 transformers pipeline으로 대체.

---

## 스크립트 목록

| 파일 | 기능 | 모델 | 상태 |
|------|------|------|------|
| t2i.py | Text → Image (+img2img) | SDXL-Turbo | ✅ 동작 |
| t2v.py | Text → Video | Wan2.1 1.3B | ⚠️ 품질 한계 |
| t2a.py | Text → Audio/Music | MusicGen-small | ✅ 동작 |
| t2s.py | Text → Speech (TTS) | mms-tts-kor/eng | ⚠️ 어눌한 품질 |
| s2t.py | Speech → Text (STT) | Whisper | ✅ 동작 |

---

## 사용법

```bash
# t2i: 인수 없으면 Usage 출력
python t2i.py "a cat on a rooftop"
python t2i.py "a cat on a rooftop" output.png
python t2i.py "oil painting style" output.png ref.jpg 0.7  # img2img

# t2v
python t2v.py "a panda eating bamboo"
python t2v.py "waves on a beach" ocean.mp4

# t2a
python t2a.py "relaxing lo-fi hip hop beats"
python t2a.py "epic orchestral" epic.wav 512

# t2s
python t2s.py "안녕하세요"
python t2s.py "안녕하세요" hello.wav
python t2s.py "Hello world" hello.wav en

# s2t
python s2t.py hello.wav
python s2t.py hello.wav small
python s2t.py hello.wav medium result.txt
```

---

## 모델 용량

| 모델 | 용량 | 캐시 경로 |
|------|------|-----------|
| FLUX.1-schnell | 25GB | ❌ 삭제 권장 (M4 16GB에서 너무 느림) |
| SDXL-Turbo | 7GB | models--stabilityai--sdxl-turbo |
| Wan2.1 1.3B | 5GB | models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers |
| MusicGen-small | 2.4GB | models--facebook--musicgen-small |
| mms-tts-kor | 145MB | models--facebook--mms-tts-kor |
| Whisper base | 145MB | (whisper 자체 캐시) |

```bash
# 캐시 확인
du -sh ~/.cache/huggingface/hub/

# FLUX 삭제
rm -rf ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell
```

---

## 알려진 버그 및 해결책

### t2i img2img - latent shape 0 버그
- 증상: RuntimeError: cannot reshape tensor of 0 elements
- 원인: num_inference_steps * strength < 1 → denoising step이 0
- 해결: strength 최솟값 = 0.25 (4스텝 기준), 코드에서 자동 보정

### t2i img2img - 얼굴 찌그러짐
- 원인: 이미지를 512x512 고정으로 강제 리사이즈
- 해결: 원본 비율 유지, 짧은 쪽 기준 512px, 8의 배수, pipe에 width/height 명시

### t2v - 메모리 부족
- 증상: RuntimeError: Invalid buffer size: 9.19 GiB
- 원인: 832x480 + 49프레임 → 16GB 초과
- 해결: 512x384, 17프레임으로 축소

### t2s - uroman 에러
- 증상: Text contains non-Roman characters
- 해결: pip install uroman

### s2t - FP16 경고
- 증상: FP16 is not supported on CPU
- 해결: fp16=False 명시, model.to("mps") 추가

---

## 품질 한계 및 클라우드 대안

### 영상 (t2v)
| 방법 | 품질 | 비용 |
|------|------|------|
| 로컬 Wan2.1 1.3B | ⭐⭐ | 무료 |
| Replicate API | ⭐⭐⭐⭐ | 영상당 50~100원 |
| Kling AI / Hailuo | ⭐⭐⭐⭐⭐ | 종량제 |

### 음성 (t2s)
| 방법 | 품질 | 비용 |
|------|------|------|
| 로컬 mms-tts-kor | ⭐⭐ | 무료 |
| OpenAI TTS (nova) | ⭐⭐⭐⭐⭐ | 1000자당 ~15원 |
| ElevenLabs | ⭐⭐⭐⭐⭐ | 무료티어 월 10000자 |
| Naver CLOVA Voice | ⭐⭐⭐⭐⭐ | 한국어 최적화 |

---

## HuggingFace 로그인

```bash
hf auth login
# 토큰: https://huggingface.co/settings/tokens
# Add token as git credential? → N
```

FLUX 라이선스 동의 완료: https://huggingface.co/black-forest-labs/FLUX.1-schnell

---

## TODO

- [ ] t2v → Replicate API 연동
  - https://replicate.com
  - pip install replicate
  - 모델: wavespeed-ai/wan-2.1-t2v-480p
  - 영상 1개 약 50~100원

- [ ] t2s → OpenAI TTS 또는 ElevenLabs 교체
  - OpenAI: pip install openai, nova 목소리 추천
  - ElevenLabs: 무료티어 월 10000자, 화자 커스텀

---

## IMSenz AI Studio API

`server.py`는 Mac mini에서 돌아가는 FastAPI 기반 이미지 생성 서버입니다.

### 엔드포인트
- `GET /health` — 서버 상태, 로드된 엔진/백엔드 확인
- `GET /presets` — 품질 preset + 사용 가능한 backend 확인
- `POST /generate` — JSON 메타데이터 + 선택적 base64 응답
- `POST /generate/raw` — PNG 바이너리 직접 응답
- `GET /files/{file_id}` — 저장된 PNG 다운로드
- `POST /unload?engine=turbo|base` — 메모리에서 엔진 unload

### `POST /generate` 요청 예시

기본 backend는 `studio`입니다.

```json
{
  "backend": "studio",
  "prompt": "a simple teal gradient background",
  "negative": "blurry, low quality, text, watermark",
  "quality": "draft",
  "width": 512,
  "height": 512,
  "filename": "teal-gradient.png",
  "include_base64": false
}
```

Codex backend를 쓰고 싶으면 이렇게 호출합니다.

```json
{
  "backend": "codex",
  "prompt": "a simple blue gradient background",
  "quality": "draft",
  "width": 512,
  "height": 512,
  "filename": "codex-api-blue-gradient.png",
  "include_base64": false
}
```

### `POST /generate` 응답 예시

```json
{
  "ok": true,
  "elapsed_sec": 8.19,
  "meta": {
    "backend": "studio",
    "engine": "turbo",
    "steps": 4,
    "width": 512,
    "height": 512,
    "guidance": 0.0,
    "quality": "draft"
  },
  "file_id": "img_20260428_145243_f1b46d2c",
  "filename": "teal-gradient.png",
  "relative_path": "2026/04/28/img_20260428_145243_f1b46d2c__teal-gradient.png",
  "download_url": "http://<host>:7860/files/img_20260428_145243_f1b46d2c",
  "bytes": 252452
}
```

### 저장 정책
- 외부 API는 `save_path`를 받지 않음
- 호출자는 `filename`만 지정 가능
- 실제 파일은 서버가 내부적으로 유니크한 `file_id`를 붙여 저장
- 저장 경로 패턴:
  - `outputs/YYYY/MM/DD/<file_id>__<filename>.png`

### backend 정책
- `backend: "studio"` → 로컬 SDXL 엔진(`turbo`, `base`) 사용
- `backend: "codex"` → Codex CLI 이미지 생성 사용
- Codex backend는 생성 성공 후 최종 저장을 검증하고,
  필요하면 임시 `ig_*.png` 결과를 복구해 저장

### 운영 메모
- 생성은 MPS 안정성을 위해 직렬화됨
- 동시 요청이 겹치면 빠르게 `503 generator busy`를 반환
- `draft/turbo`는 빠른 초안용, `balanced/base` 이상은 시간이 오래 걸림
- `backend: "codex"` 는 일반적으로 더 느리지만 프롬프트 해석이 다를 수 있음

## Codex CLI 향후 기능 (under development)

- image_generation: 켜지면 CLI에서 이미지 생성
- memories: 켜지면 대화 간 기억 유지
- realtime_conversation: 실시간 음성 대화
