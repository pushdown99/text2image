"""
IMSenz AI Studio — minimal image generation API.
Single public endpoint:
  POST /image/generate
Success: image/png
Failure: JSON error
"""
import io
import os
import re
import time
import shutil
import threading
import subprocess
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image

app = FastAPI(title='IMSenz AI Studio', version='0.6.0')


class Quality(str, Enum):
    draft = 'draft'
    balanced = 'balanced'
    hq = 'hq'
    ultra = 'ultra'


class Backend(str, Enum):
    studio = 'studio'
    codex = 'codex'


PRESETS = {
    Quality.draft: {'engine': 'turbo', 'steps': 4, 'size': 512, 'guidance': 0.0},
    Quality.balanced: {'engine': 'base', 'steps': 20, 'size': 768, 'guidance': 6.5},
    Quality.hq: {'engine': 'base', 'steps': 25, 'size': 1024, 'guidance': 6.5},
    Quality.ultra: {'engine': 'base', 'steps': 40, 'size': 1024, 'guidance': 7.5},
}

ENGINE_IDS = {
    'base': 'stabilityai/stable-diffusion-xl-base-1.0',
    'turbo': 'stabilityai/sdxl-turbo',
}

DEFAULT_NEG = (
    'low quality, blurry, deformed eyes, asymmetric eyes, crossed eyes, '
    'cartoon, illustration, anime, painting, ugly, distorted face, '
    'bad anatomy, extra limbs, watermark, text, logo, signature'
)

AI_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = AI_DIR.parent
CODEX_BIN_CANDIDATES = ['codex', '/opt/homebrew/bin/codex']
CODEX_HOME_IMAGES_DIR = Path.home() / '.codex' / 'generated_images'

_pipes: dict[str, object] = {}
_pipe_lock = threading.Lock()
_gen_lock = threading.Lock()
GEN_LOCK_TIMEOUT_SEC = 0.25


class GenRequest(BaseModel):
    prompt: str
    negative: str | None = None
    quality: Quality = Quality.balanced
    backend: Backend = Backend.studio
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    guidance: float | None = None
    seed: int | None = None


def get_pipe(engine: str):
    if engine not in ENGINE_IDS:
        raise HTTPException(400, f'unknown engine: {engine}')
    if engine in _pipes:
        return _pipes[engine]
    with _pipe_lock:
        if engine in _pipes:
            return _pipes[engine]
        pipe = AutoPipelineForText2Image.from_pretrained(
            ENGINE_IDS[engine],
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
        ).to('mps')
        _pipes[engine] = pipe
        return pipe


def _find_codex_bin() -> str | None:
    for candidate in CODEX_BIN_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if Path(candidate).exists():
            return candidate
    return None


def _acquire_generation_lock():
    acquired = _gen_lock.acquire(timeout=GEN_LOCK_TIMEOUT_SEC)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail='generator busy: another image request is already running; retry shortly',
        )


def _slugify_filename(text: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('-._') or 'image'
    return f'{stem}.png'


def _collect_recent_codex_images(started_at: float):
    candidates = []

    if CODEX_HOME_IMAGES_DIR.exists():
        for path in CODEX_HOME_IMAGES_DIR.rglob('ig_*.png'):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if stat.st_mtime >= started_at - 5 and stat.st_size > 0:
                candidates.append((stat.st_mtime, path))

    var_folders = Path('/var/folders')
    if var_folders.exists():
        for path in var_folders.glob('*/*/*/T/ig_*.png'):
            try:
                stat = path.stat()
            except (FileNotFoundError, PermissionError):
                continue
            if stat.st_mtime >= started_at - 5 and stat.st_size > 0:
                candidates.append((stat.st_mtime, path))

    return sorted(candidates, key=lambda item: item[0])


def _build_codex_prompt(req: GenRequest) -> str:
    size_hint = ''
    if req.width and req.height:
        size_hint = f' Aim for approximately {req.width}x{req.height} pixels composition/aspect ratio.'
    negative_hint = f' Avoid: {req.negative}.' if req.negative else ''
    quality_hint = {
        Quality.draft: 'Prefer a quick draft output.',
        Quality.balanced: 'Prefer balanced photorealistic quality.',
        Quality.hq: 'Prefer high-detail photorealistic quality.',
        Quality.ultra: 'Prefer the highest practical detail and realism.',
    }[req.quality]
    return (
        f'Generate one image only. {req.prompt}. '
        f'{quality_hint}{size_hint} {negative_hint} '
        'Return a single raster image only, no collage, no grid, no multiple panels.'
    ).strip()


def _generate_via_codex(req: GenRequest):
    codex_bin = _find_codex_bin()
    if codex_bin is None:
        raise HTTPException(status_code=503, detail='codex CLI is not installed on this server')

    log_dir = AI_DIR / '.codex-api-logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_filename(req.prompt[:40]).replace('.png', '')
    log_path = log_dir / f'{int(time.time())}_{slug}.txt'
    prompt = _build_codex_prompt(req)
    codex_env = os.environ.copy()
    codex_env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:' + codex_env.get('PATH', '')

    _acquire_generation_lock()
    started_at = time.time()
    try:
        proc = subprocess.run(
            [
                codex_bin, 'exec',
                '--skip-git-repo-check',
                '-C', str(WORKSPACE_DIR),
                '--dangerously-bypass-approvals-and-sandbox',
                prompt,
                '-o', str(log_path),
            ],
            cwd=str(WORKSPACE_DIR),
            env=codex_env,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail='image generation timed out')
    finally:
        _gen_lock.release()

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or '').strip()[-1000:]
        raise HTTPException(status_code=500, detail=detail or 'image generation failed')

    candidates = _collect_recent_codex_images(started_at)
    if not candidates:
        raise HTTPException(status_code=500, detail='generated image could not be located')

    _, latest = candidates[-1]
    with Image.open(latest) as opened:
        return opened.copy()


def _generate_via_studio(req: GenRequest):
    preset = PRESETS[req.quality]
    engine = preset['engine']
    steps = req.steps if req.steps is not None else preset['steps']
    size = preset['size']
    width = req.width if req.width is not None else size
    height = req.height if req.height is not None else size
    guidance = req.guidance if req.guidance is not None else preset['guidance']
    negative = req.negative if req.negative is not None else (DEFAULT_NEG if engine == 'base' else None)

    pipe = get_pipe(engine)
    generator = None
    if req.seed is not None:
        generator = torch.Generator(device='mps').manual_seed(req.seed)

    _acquire_generation_lock()
    try:
        kwargs = {
            'prompt': req.prompt,
            'num_inference_steps': steps,
            'guidance_scale': guidance,
            'width': width,
            'height': height,
            'generator': generator,
        }
        if negative:
            kwargs['negative_prompt'] = negative
        out = pipe(**kwargs)
    finally:
        _gen_lock.release()

    return out.images[0]


def _generate(req: GenRequest):
    if req.backend == Backend.codex:
        return _generate_via_codex(req)
    return _generate_via_studio(req)


@app.post('/image/generate')
def image_generate(req: GenRequest):
    img = _generate(req)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return Response(content=buf.getvalue(), media_type='image/png')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)
