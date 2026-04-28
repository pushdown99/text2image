"""
IMSenz AI Studio — FastAPI service.
Mac mini M4 (MPS). 4 quality tiers, two engines (SDXL base + SDXL-Turbo).

Engines:
  - base   : stabilityai/stable-diffusion-xl-base-1.0   (high fidelity)
  - turbo  : stabilityai/sdxl-turbo                      (sub-second draft)

Presets (engine + steps + size + guidance):
  - draft    : turbo,  4 steps, 512,  guidance 0.0    ~3s
  - balanced : base,  20 steps, 768,  guidance 6.5    ~45s
  - hq       : base,  25 steps, 1024, guidance 6.5    ~3min
  - ultra    : base,  40 steps, 1024, guidance 7.5    ~5min
"""
import io
import os
import re
import time
import base64
import shutil
import secrets
import threading
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from enum import Enum
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image

app = FastAPI(title='IMSenz AI Studio', version='0.5.0')


class Quality(str, Enum):
    draft    = 'draft'
    balanced = 'balanced'
    hq       = 'hq'
    ultra    = 'ultra'


class Backend(str, Enum):
    studio = 'studio'
    codex = 'codex'


PRESETS = {
    Quality.draft:    {'engine': 'turbo', 'steps':  4, 'size':  512, 'guidance': 0.0},
    Quality.balanced: {'engine': 'base',  'steps': 20, 'size':  768, 'guidance': 6.5},
    Quality.hq:       {'engine': 'base',  'steps': 25, 'size': 1024, 'guidance': 6.5},
    Quality.ultra:    {'engine': 'base',  'steps': 40, 'size': 1024, 'guidance': 7.5},
}

ENGINE_IDS = {
    'base':  'stabilityai/stable-diffusion-xl-base-1.0',
    'turbo': 'stabilityai/sdxl-turbo',
}

DEFAULT_NEG = (
    'low quality, blurry, deformed eyes, asymmetric eyes, crossed eyes, '
    'cartoon, illustration, anime, painting, ugly, distorted face, '
    'bad anatomy, extra limbs, watermark, text, logo, signature'
)

_pipes: dict[str, object] = {}
_pipe_lock = threading.Lock()
_gen_lock  = threading.Lock()
GEN_LOCK_TIMEOUT_SEC = 0.25
AI_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = AI_DIR.parent
OUTPUTS_DIR = AI_DIR / 'outputs'
CODEX_LOGS_DIR = OUTPUTS_DIR / '_codex_logs'
CODEX_BIN_CANDIDATES = ['codex', '/opt/homebrew/bin/codex']


def get_pipe(engine: str):
    if engine not in ENGINE_IDS:
        raise HTTPException(400, f'unknown engine: {engine}')
    if engine in _pipes:
        return _pipes[engine]
    with _pipe_lock:
        if engine in _pipes:
            return _pipes[engine]
        t0 = time.time()
        pipe = AutoPipelineForText2Image.from_pretrained(
            ENGINE_IDS[engine],
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
        ).to('mps')
        _pipes[engine] = pipe
        print(f'[load] {engine} loaded in {time.time()-t0:.1f}s')
        return pipe


class GenRequest(BaseModel):
    prompt: str
    negative: str | None = None
    quality: Quality = Quality.balanced
    backend: Backend = Backend.studio
    # Optional overrides (only set if you want to override the preset)
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    guidance: float | None = None
    seed: int | None = None
    filename: str | None = None
    include_base64: bool = True


def _slugify_filename(filename: str | None) -> str:
    name = (filename or 'image.png').strip()
    name = Path(name).name
    stem = Path(name).stem or 'image'
    stem = re.sub(r'[^A-Za-z0-9._-]+', '-', stem).strip('-._') or 'image'
    return f'{stem}.png'


def _build_managed_output_path(filename: str | None):
    safe_name = _slugify_filename(filename)
    file_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
    relative = Path(datetime.now().strftime('%Y/%m/%d')) / f'{file_id}__{safe_name}'
    return file_id, safe_name, (OUTPUTS_DIR / relative), relative


def _save_png_atomic(img, filename: str | None = None):
    file_id, public_name, target, relative = _build_managed_output_path(filename)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='tmp_', suffix='.png', dir=target.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
        img.save(tmp, format='PNG')
        tmp.flush()
        os.fsync(tmp.fileno())

    try:
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if not target.exists() or target.stat().st_size == 0:
        raise HTTPException(status_code=500, detail='image generated but final save verification failed')

    return {
        'file_id': file_id,
        'filename': public_name,
        'relative_path': str(relative),
        'absolute_path': str(target),
        'bytes': target.stat().st_size,
    }


def _find_file_by_id(file_id: str) -> Path:
    matches = sorted(OUTPUTS_DIR.rglob(f'{file_id}__*.png'))
    if not matches:
        raise HTTPException(status_code=404, detail='file not found')
    return matches[-1]


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
            detail='generator busy: another image request is already running; retry shortly'
        )


def _recover_codex_temp_output(target: Path, started_at: float) -> bool:
    candidates = []
    for path in Path('/var/folders').glob('*/*/*/T/ig_*.png'):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime >= started_at - 2 and stat.st_size > 0:
            candidates.append((stat.st_mtime, path))

    if not candidates:
        return False

    _, latest = max(candidates, key=lambda item: item[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, target)
    return target.exists() and target.stat().st_size > 0


def _build_codex_prompt(req: GenRequest, target: Path) -> str:
    size_hint = ''
    if req.width and req.height:
        size_hint = f' Aim for approximately {req.width}x{req.height} pixels composition/aspect ratio.'
    negative_hint = f' Avoid: {req.negative}.' if req.negative else ''
    quality_hint = {
        Quality.draft: 'Prefer a quick draft output.',
        Quality.balanced: 'Prefer balanced photorealistic quality.',
        Quality.hq: 'Prefer high-detail photorealistic quality.',
        Quality.ultra: 'Prefer the highest practical detail and realism.'
    }[req.quality]
    return (
        f'Generate one image only. {req.prompt}. '
        f'{quality_hint}{size_hint} {negative_hint} '
        'Return a single raster image only, no collage, no grid, no multiple panels. '
        f'Save the result to {target}.'
    ).strip()


def _generate_via_codex(req: GenRequest):
    codex_bin = _find_codex_bin()
    if codex_bin is None:
        raise HTTPException(status_code=503, detail='codex CLI is not installed on this server')

    file_id, public_name, target, relative = _build_managed_output_path(req.filename)
    CODEX_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CODEX_LOGS_DIR / f'{file_id}_last.txt'
    prompt = _build_codex_prompt(req, target)

    _acquire_generation_lock()
    started_at = time.time()
    codex_env = os.environ.copy()
    codex_env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:' + codex_env.get('PATH', '')

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
        raise HTTPException(status_code=504, detail='codex image generation timed out')
    finally:
        _gen_lock.release()

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or '').strip()[-1000:]
        raise HTTPException(status_code=500, detail=f'codex generation failed: {detail or "unknown error"}')

    if (not target.exists() or target.stat().st_size == 0) and not _recover_codex_temp_output(target, started_at):
        raise HTTPException(status_code=500, detail=f'codex generated output but final save could not be verified; inspect {log_path}')

    with Image.open(target) as opened:
        img = opened.copy()

    meta = {
        'backend': req.backend.value,
        'engine': 'codex',
        'quality': req.quality.value,
        'width': req.width,
        'height': req.height,
        'steps': None,
        'guidance': None,
        'log_path': str(log_path),
    }
    saved = {
        'file_id': file_id,
        'filename': public_name,
        'relative_path': str(relative),
        'absolute_path': str(target),
        'bytes': target.stat().st_size,
    }
    return img, time.time() - started_at, meta, saved


@app.get('/health')
def health():
    return {
        'ok': True,
        'engines_loaded': list(_pipes.keys()),
        'device': 'mps',
        'outputs_dir': str(OUTPUTS_DIR.resolve()),
        'backends': [b.value for b in Backend],
        'codex_available': _find_codex_bin() is not None,
    }


@app.get('/files/{file_id}')
def get_file(file_id: str):
    path = _find_file_by_id(file_id)
    filename = path.name.split('__', 1)[1] if '__' in path.name else path.name
    return FileResponse(path, media_type='image/png', filename=filename)


@app.get('/presets')
def list_presets():
    return {
        'qualities': {q.value: PRESETS[q] for q in Quality},
        'default': Quality.balanced.value,
        'engines': list(ENGINE_IDS.keys()),
        'backends': [b.value for b in Backend],
    }


def _generate_via_studio(req: GenRequest):
    preset = PRESETS[req.quality]
    engine  = preset['engine']
    steps   = req.steps    if req.steps    is not None else preset['steps']
    size    = preset['size']
    width   = req.width    if req.width    is not None else size
    height  = req.height   if req.height   is not None else size
    guidance= req.guidance if req.guidance is not None else preset['guidance']
    negative = req.negative if req.negative is not None else (DEFAULT_NEG if engine == 'base' else None)

    pipe = get_pipe(engine)
    generator = None
    if req.seed is not None:
        generator = torch.Generator(device='mps').manual_seed(req.seed)

    _acquire_generation_lock()
    try:
        t0 = time.time()
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
        elapsed = time.time() - t0
    finally:
        _gen_lock.release()

    meta = {
        'backend': req.backend.value,
        'engine': engine,
        'steps': steps,
        'width': width,
        'height': height,
        'guidance': guidance,
        'quality': req.quality.value,
    }
    saved = _save_png_atomic(out.images[0], filename=req.filename)
    return out.images[0], elapsed, meta, saved


def _generate(req: GenRequest):
    if req.backend == Backend.codex:
        return _generate_via_codex(req)
    return _generate_via_studio(req)


@app.post('/generate')
def generate(req: GenRequest, request: Request):
    img, elapsed, meta, saved = _generate(req)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    payload = {
        'ok': True,
        'elapsed_sec': round(elapsed, 2),
        'meta': meta,
        'file_id': saved['file_id'],
        'filename': saved['filename'],
        'relative_path': saved['relative_path'],
        'download_url': str(request.url_for('get_file', file_id=saved['file_id'])),
        'bytes': saved['bytes'],
    }
    if req.include_base64:
        payload['png_base64'] = base64.b64encode(buf.getvalue()).decode()
    return payload


@app.post('/generate/raw')
def generate_raw(req: GenRequest, request: Request):
    img, elapsed, meta, saved = _generate(req)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    headers = {
        'X-Elapsed-Sec': str(round(elapsed, 2)),
        'X-Backend': meta['backend'],
        'X-Engine': str(meta['engine']),
        'X-Steps': str(meta['steps']),
        'X-Size': f"{meta['width']}x{meta['height']}",
        'X-Quality': str(meta['quality']),
        'X-File-Id': saved['file_id'],
        'X-Filename': saved['filename'],
        'X-Relative-Path': saved['relative_path'],
        'X-Download-Url': str(request.url_for('get_file', file_id=saved['file_id'])),
    }
    return Response(content=buf.getvalue(), media_type='image/png', headers=headers)



@app.post("/unload")
def unload(engine: str):
    import gc, torch
    if engine not in _pipes:
        return {'ok': False, 'reason': f'engine {engine} not loaded'}
    with _pipe_lock:
        del _pipes[engine]
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return {'ok': True, 'unloaded': engine, 'now_loaded': list(_pipes.keys())}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)
