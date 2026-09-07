import os
import io
import re
import json
import shutil
import base64
import hashlib
import requests
import subprocess
import pdfkit
import threading
import concurrent.futures
import time
from PIL import Image
from pdf2image import convert_from_path
from app.core.config import settings
from app.services.job_manager import JobManager
from app.services.auth_manager import AuthManager
from app.db.prompt import default_system_prompt, default_user_prompt

# ==========================================
# Global Lock
# ==========================================
# Local LLM 사용 시에만 작동할 Lock (GPU 자원 보호)
local_gpu_lock = threading.Lock()

# 기본 모델 (app/core/config.py 에서 관리)
DEFAULT_MODEL = settings.DEFAULT_MODEL

# USD / 1M tokens. https://developers.openai.com/api/docs/pricing
# GPT-5.6 이후 모델은 캐시 쓰기(cache_write_tokens)에 입력 단가의 1.25배가 청구된다.
PRICING_TABLE = {
    "gpt-5.6-luna": {
        "input": 0.20,
        "cached": 0.02,
        "output": 1.20,
        "cache_write_mult": 1.25,
    },
    "gpt-5.6-terra": {
        "input": 2.00,
        "cached": 0.20,
        "output": 12.00,
        "cache_write_mult": 1.25,
    },
    "gpt-5.6-sol": {
        "input": 4.00,
        "cached": 0.40,
        "output": 20.00,
        "cache_write_mult": 1.25,
    },
    "gpt-5.2": {"input": 1.75, "cached": 0.175, "output": 14.00},
    "gpt-5-mini": {"input": 0.25, "cached": 0.025, "output": 2.00},
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
}

# GPT-5.x 는 reasoning 모델이라 기본 effort(medium)의 추론 토큰이 출력 요금으로 청구된다.
# 슬라이드 설명 작업에는 low 로 충분하며 출력 토큰을 크게 줄인다. (none/minimal/low/medium/high/xhigh/max)
REASONING_EFFORT = "low"

# ==========================================
# 토큰/비용 절감 관련 설정
# ==========================================
# Batch API 는 입력/출력 모두 50% 할인 (캐시 할인은 적용되지 않음)
BATCH_DISCOUNT = 0.5
# API 로 전송하는 이미지의 긴 변 최대 픽셀. 결과물(zip)에 들어가는 이미지는 원본 해상도 유지.
# GPT-5.6 계열은 이미지를 32px 패치로 토큰화 (ceil(w/32) * ceil(h/32) * 1.2, detail=auto 는 원본 크기 유지).
#   150DPI 16:9 슬라이드(약 2000x1125) ≈ 2,722 토큰  →  1536x864 ≈ 1,555 토큰 (약 43% 절감)
MAX_IMAGE_SIDE = 1536
# 실시간(OpenAI) 처리 시 동시 요청 수
PARALLEL_WORKERS = 3
# Batch API 폴링 간격(초), 진행 로그 heartbeat 간격(초), 최대 대기(초)
BATCH_POLL_INTERVAL = 15
BATCH_HEARTBEAT = 600
BATCH_MAX_WAIT = 24 * 3600 + 1800
# Batch 입력 파일 하나의 최대 크기 (OpenAI 제한 200MB 보다 여유 있게)
BATCH_MAX_FILE_BYTES = 150 * 1024 * 1024
# 사용자 프롬프트 템플릿의 {filename} 자리에 들어가는 고정 토큰 (캐시 prefix 유지용)
FILENAME_PLACEHOLDER = "[파일명]"

# ==========================================
# Helper Functions
# ==========================================


def get_headers(api_key=None, content_type="application/json"):
    headers = {"Authorization": f"Bearer {api_key or settings.CUSTOM_TOKEN}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def get_target_model(user_settings):
    pref = user_settings.get("preferred_model", DEFAULT_MODEL)
    user_key = user_settings.get("openai_api_key", "")

    system_prompt = user_settings.get("custom_prompt", "")
    user_prompt_template = user_settings.get("custom_user_prompt", "")

    config = {
        "model_id": "gpt-4o",
        "base_url": settings.CUSTOM_BASE_URL,
        "api_key": None,
        "provider": "local",
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "use_batch": False,
    }

    if pref.startswith("gpt"):
        if not user_key:
            raise ValueError(
                "OpenAI 모델이 선택되었으나 API Key가 설정되지 않았습니다."
            )
        config.update(
            {
                "provider": "openai",
                "model_id": pref,
                "base_url": "https://api.openai.com/v1",
                "api_key": user_key,
                "use_batch": bool(user_settings.get("use_batch_api", False)),
            }
        )
    else:
        try:
            url = f"{settings.CUSTOM_BASE_URL}/models"
            resp = requests.get(url, headers=get_headers(), timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    config["model_id"] = models[0]["id"]
        except Exception:
            pass

    return config


def image_to_data_url(path: str, max_side: int = MAX_IMAGE_SIDE) -> str:
    """
    이미지를 data URL 로 변환. 긴 변이 max_side 를 넘으면 메모리 상에서만 축소하여
    이미지 입력 토큰과 전송 페이로드를 줄인다. (디스크의 원본은 그대로 유지)
    """
    with Image.open(path) as img:
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            raw, mime = buf.getvalue(), "image/png"
        else:
            with open(path, "rb") as f:
                raw = f.read()
            ext = os.path.splitext(path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"

    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def resolve_prompts(model_config: dict):
    system_instruction = (model_config.get("system_prompt") or "").strip()
    if not system_instruction:
        system_instruction = default_system_prompt.strip()

    user_template = (model_config.get("user_prompt_template") or "").strip()
    if not user_template:
        user_template = default_user_prompt.strip()

    return system_instruction, user_template


def build_messages(image_path: str, model_config: dict):
    """
    Prompt Caching 친화적인 메시지 구성.

    OpenAI(및 vLLM 등 prefix cache 를 지원하는 서버)는 요청의 '앞부분'이 이전 요청과
    완전히 동일할 때만 캐시를 재사용한다. 따라서
      - 모든 슬라이드에서 동일한 system prompt + 사용자 지시문을 맨 앞에 두고
      - 슬라이드마다 달라지는 이미지와 파일명은 맨 뒤에 배치한다.
    사용자 템플릿의 {filename} 은 고정 토큰으로 치환하고, 실제 파일명은 마지막 텍스트
    블록으로 전달하여 지시문 부분이 매 요청 바이트 단위로 동일하도록 만든다.
    """
    filename = os.path.basename(image_path)
    system_instruction, user_template = resolve_prompts(model_config)

    has_filename = "{filename}" in user_template
    static_text = user_template.replace("{filename}", FILENAME_PLACEHOLDER)

    user_content = [
        {"type": "text", "text": static_text},
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
    ]
    if has_filename:
        user_content.append(
            {"type": "text", "text": f'{FILENAME_PLACEHOLDER} = "{filename}"'}
        )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def prompt_cache_key(model_config: dict) -> str:
    """
    동일한 프롬프트(모델 + system + 지시문)를 쓰는 요청들이 같은 캐시 서버로
    라우팅되도록 하는 키. 같은 사용자의 다른 작업에서도 캐시가 재사용된다.
    """
    system_instruction, user_template = resolve_prompts(model_config)
    digest = hashlib.sha256(
        f"{model_config['model_id']}\n{system_instruction}\n{user_template}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"lecai-{digest[:32]}"


def gpt_version(model_id: str):
    """'gpt-5.6-luna' -> (5, 6), 'gpt-5-mini' -> (5, 0), 'gpt-4o' -> (4, 0). 파싱 실패 시 None"""
    m = re.match(r"gpt-(\d+)(?:\.(\d+))?", model_id)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def is_gpt56_or_later(model_id: str) -> bool:
    v = gpt_version(model_id)
    return v is not None and v >= (5, 6)


def is_reasoning_model(model_id: str) -> bool:
    v = gpt_version(model_id)
    return v is not None and v >= (5, 0)


def supports_extended_cache(model_id: str) -> bool:
    """prompt_cache_retention="24h" 지원 모델 (GPT-5.6 미만의 gpt-5.x, gpt-4.1). GPT-5.6+ 에서는 deprecated."""
    if is_gpt56_or_later(model_id):
        return False
    return model_id.startswith(("gpt-5", "gpt-4.1"))


def build_payload(image_path: str, model_config: dict, for_batch: bool = False):
    payload = {
        "model": model_config["model_id"],
        "messages": build_messages(image_path, model_config),
        "max_completion_tokens": 10000,
    }

    if model_config["provider"] != "openai":
        return payload

    model_id = model_config["model_id"]

    # reasoning 모델: 추론 토큰(출력 요금)을 줄이기 위해 effort 를 낮춘다.
    if is_reasoning_model(model_id) and REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT

    # 캐시 옵션은 실시간 요청에만. (Batch 는 별도 요금 체계)
    if not for_batch:
        # 동일 프롬프트 요청이 같은 캐시로 라우팅되도록 하는 키 (모든 모델 공통)
        payload["prompt_cache_key"] = prompt_cache_key(model_config)
        if supports_extended_cache(model_id):
            # GPT-5.6 미만: 기본 캐시(5~10분) 대신 24시간 유지
            payload["prompt_cache_retention"] = "24h"
        # GPT-5.6 이상: prompt_cache_options.ttl 은 "30m" 이 유일값이자 기본값이고
        # mode 도 기본(implicit)이 슬라이드 단발 요청에 맞으므로 별도 옵션을 보내지 않는다.

    return payload


def empty_usage() -> dict:
    return {"prompt": 0, "cached": 0, "cache_write": 0, "completion": 0}


def parse_usage(result: dict) -> dict:
    usage_info = empty_usage()
    try:
        usage = result.get("usage", {}) or {}
        details = usage.get("prompt_tokens_details", {}) or {}
        usage_info["prompt"] = usage.get("prompt_tokens", 0) or 0
        usage_info["completion"] = usage.get("completion_tokens", 0) or 0
        usage_info["cached"] = details.get("cached_tokens", 0) or 0
        # GPT-5.6+: 캐시에 새로 기록된 토큰 수 (1.25배 청구)
        usage_info["cache_write"] = details.get("cache_write_tokens", 0) or 0
    except Exception:
        pass
    return usage_info


# 모델에 따라 지원 여부가 달라 400 이 나면 제거하고 재시도하는 선택 파라미터
OPTIONAL_PARAMS = ("prompt_cache_retention", "prompt_cache_options", "reasoning_effort")


def describe_image(image_path: str, model_config: dict):
    filename = os.path.basename(image_path)
    url = f"{model_config['base_url']}/chat/completions"
    headers = get_headers(model_config["api_key"])
    payload = build_payload(image_path, model_config)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)

            if resp.status_code == 200:
                result = resp.json()
                content = result["choices"][0]["message"].get("content", "")
                finish_reason = result["choices"][0].get("finish_reason")

                if not content or not content.strip():
                    print(
                        f"[Empty Response] {filename} returned empty content. Retrying... ({attempt + 1}/{max_retries})"
                    )
                    print(f"Reason: {finish_reason}")
                    time.sleep(2)
                    continue

                return content, parse_usage(result)

            elif resp.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(
                    f"[Rate Limit] 429 Error on {filename}. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
                continue

            elif resp.status_code == 400 and any(
                p in payload and p in resp.text for p in OPTIONAL_PARAMS
            ):
                # 모델이 지원하지 않는 선택 옵션이면 해당 옵션만 제거하고 재시도
                for p in OPTIONAL_PARAMS:
                    if p in payload and p in resp.text:
                        print(
                            f"[Param] {model_config['model_id']} rejected '{p}'. Retrying without it."
                        )
                        payload.pop(p, None)
                continue

            else:
                print(f"[API Error] {resp.status_code}: {resp.text}")
                if resp.status_code >= 500:
                    time.sleep(3)
                    continue
                raise RuntimeError(
                    f"OpenAI API Error: {resp.status_code} - {resp.text}"
                )

        except requests.exceptions.Timeout:
            print(f"[Timeout] {filename} timed out. Retrying...")
            time.sleep(3)
            continue

        except Exception as e:
            print(f"[Exception] {str(e)}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)

    raise RuntimeError(f"Failed to process {filename} after {max_retries} attempts.")


def convert_ppt_to_pdf(ppt_path: str, output_dir: str):
    subprocess.run(
        [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            ppt_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = os.path.splitext(os.path.basename(ppt_path))[0]
    return os.path.join(output_dir, f"{base}.pdf")


def calculate_total_cost(model_id, total_usage, exchange_rate=1400, batch=False):
    matched_model = next((m for m in PRICING_TABLE if m in model_id), "default")
    if matched_model == "default":
        return 0.0, 0

    rates = PRICING_TABLE[matched_model]
    cached = total_usage.get("cached", 0)
    cache_write = (
        total_usage.get("cache_write", 0) if "cache_write_mult" in rates else 0
    )
    regular_input = max(total_usage["prompt"] - cached - cache_write, 0)

    usd_cost = (
        (regular_input * rates["input"])
        + (cached * rates["cached"])
        + (cache_write * rates["input"] * rates.get("cache_write_mult", 1.0))
        + (total_usage["completion"] * rates["output"])
    ) / 1000000

    if batch:
        usd_cost *= BATCH_DISCOUNT

    return round(usd_cost, 4), int(usd_cost * exchange_rate)


# ==========================================
# 진행 상태 (실시간/배치 공용)
# ==========================================


class AnalysisState:
    """슬라이드 분석 진행 상태. 실시간 처리분과 배치 처리분의 사용량을 따로 집계한다."""

    def __init__(self, job_id: str, model_config: dict, total_pages: int):
        self.job_id = job_id
        self.model_config = model_config
        self.total_pages = total_pages
        self.results_map = {}
        self.usage = empty_usage()
        self.batch_usage = empty_usage()
        self.completed = 0
        self.lock = threading.Lock()

    def add_result(self, idx, filename, content, usage, from_batch=False):
        with self.lock:
            self.results_map[idx] = (filename, content)
            for k in self.usage:
                self.usage[k] += usage.get(k, 0)
                if from_batch:
                    self.batch_usage[k] += usage.get(k, 0)
            self.completed += 1

    def add_failure(self, idx, error):
        with self.lock:
            self.results_map[idx] = (
                "error.png",
                f"**[분석 실패]** 오류가 발생했습니다: {error}",
            )
            self.completed += 1

    @property
    def total_tokens(self):
        return self.usage["prompt"] + self.usage["completion"]

    def cost(self):
        """실시간 처리분은 정가, 배치 처리분은 50% 할인가로 합산"""
        realtime_usage = {k: self.usage[k] - self.batch_usage[k] for k in self.usage}
        usd_rt, _ = calculate_total_cost(self.model_config["model_id"], realtime_usage)
        usd_b, _ = calculate_total_cost(
            self.model_config["model_id"], self.batch_usage, batch=True
        )
        usd = round(usd_rt + usd_b, 4)
        return usd, int(usd * 1400)

    def log_progress(self, prefix="분석 중"):
        with self.lock:
            msg = f"{prefix} ({self.completed}/{self.total_pages}) | 누적 토큰: {self.total_tokens:,}"
            if self.usage["cached"]:
                msg += f" (캐시 적중 {self.usage['cached']:,})"
            if self.model_config["provider"] == "openai":
                usd_val, krw_val = self.cost()
                msg += f" | 예상 비용: ${usd_val:.3f} (₩{krw_val:,})"
            JobManager.update_progress(
                self.job_id, self.completed, self.total_pages, msg
            )


# ==========================================
# 실시간 처리 (OpenAI 병렬 / Local 순차)
# ==========================================


def _run_realtime(state: AnalysisState, items, max_workers: int, prefix="분석 중"):
    """
    items: [(idx, image_path), ...]
    OpenAI 병렬 처리 시 첫 슬라이드는 단독으로 먼저 보내 prompt cache 를 채운 뒤
    (cache warm-up) 나머지를 병렬로 보낸다. 동시에 출발하면 전부 캐시 미스가 난다.
    """
    items = list(items)
    if not items:
        return

    def run_one(idx, img_path):
        content, usage = describe_image(img_path, state.model_config)
        state.add_result(idx, os.path.basename(img_path), content, usage)

    def run_safely(idx, img_path):
        try:
            run_one(idx, img_path)
        except Exception as e:
            print(f"[FINAL ERROR] Slide {idx} processing failed: {e}")
            state.add_failure(idx, str(e))
        state.log_progress(prefix)

    warmup, rest = items[0], items[1:]
    run_safely(*warmup)

    if not rest:
        return

    if max_workers <= 1:
        for idx, img_path in rest:
            run_safely(idx, img_path)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_safely, idx, p) for idx, p in rest]
        concurrent.futures.wait(futures)


# ==========================================
# OpenAI Batch API (50% 할인)
# ==========================================


class BatchUnavailable(Exception):
    """배치 등록 자체가 불가능한 경우 (큐 한도 초과 등) → 실시간 처리로 대체"""


def _batch_upload_file(base_url, api_key, data: bytes, name: str) -> str:
    resp = requests.post(
        f"{base_url}/files",
        headers=get_headers(api_key, content_type=None),
        files={"file": (name, data, "application/jsonl")},
        data={"purpose": "batch"},
        timeout=600,
    )
    if resp.status_code != 200:
        raise BatchUnavailable(f"파일 업로드 실패 ({resp.status_code}): {resp.text}")
    return resp.json()["id"]


def _batch_create(base_url, api_key, file_id, job_id) -> dict:
    resp = requests.post(
        f"{base_url}/batches",
        headers=get_headers(api_key),
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"lecai_job_id": job_id},
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise BatchUnavailable(f"배치 생성 실패 ({resp.status_code}): {resp.text}")
    return resp.json()


def _batch_get(base_url, api_key, batch_id) -> dict:
    resp = requests.get(
        f"{base_url}/batches/{batch_id}", headers=get_headers(api_key), timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def _batch_cancel(base_url, api_key, batch_id):
    try:
        requests.post(
            f"{base_url}/batches/{batch_id}/cancel",
            headers=get_headers(api_key),
            timeout=30,
        )
    except Exception as e:
        print(f"[Batch] cancel failed for {batch_id}: {e}")


def _file_delete(base_url, api_key, file_id):
    if not file_id:
        return
    try:
        requests.delete(
            f"{base_url}/files/{file_id}", headers=get_headers(api_key), timeout=30
        )
    except Exception as e:
        print(f"[Batch] file delete failed for {file_id}: {e}")


def _file_content(base_url, api_key, file_id) -> str:
    resp = requests.get(
        f"{base_url}/files/{file_id}/content",
        headers=get_headers(api_key),
        timeout=600,
    )
    resp.raise_for_status()
    return resp.text


def _build_batch_chunks(items, model_config):
    """슬라이드별 요청을 JSONL 로 직렬화하고 파일 크기 한도에 맞춰 나눈다."""
    chunks, current, current_size = [], [], 0
    for idx, img_path in items:
        line = json.dumps(
            {
                "custom_id": f"slide-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_payload(img_path, model_config, for_batch=True),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        if current and current_size + len(line) + 1 > BATCH_MAX_FILE_BYTES:
            chunks.append(current)
            current, current_size = [], 0
        current.append(line)
        current_size += len(line) + 1
    if current:
        chunks.append(current)
    return chunks


BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}
BATCH_STATUS_KO = {
    "validating": "검증 중",
    "in_progress": "처리 중",
    "finalizing": "마무리 중",
    "completed": "완료",
    "failed": "실패",
    "expired": "만료",
    "cancelling": "취소 중",
    "cancelled": "취소됨",
}


def _run_batch(state: AnalysisState, items, image_by_idx):
    """
    OpenAI Batch API 로 슬라이드를 처리한다. 진행률은 배치의 request_counts 를 폴링해
    JobManager 에 반영하므로 대시보드의 진행 표시는 실시간 처리와 동일하게 유지된다.
    반환값: 배치로 처리되지 못한 (idx, path) 목록 → 호출측에서 실시간 처리
    """
    cfg = state.model_config
    base_url, api_key = cfg["base_url"], cfg["api_key"]
    job_id = state.job_id
    items = list(items)

    JobManager.update_progress(
        job_id,
        0,
        state.total_pages,
        f"배치 요청 파일 생성 중 ({len(items)}개 슬라이드)",
    )
    chunks = _build_batch_chunks(items, cfg)

    batches = []  # {"id", "input_file_id", "status", "counts"}
    try:
        for n, chunk in enumerate(chunks, 1):
            JobManager.update_progress(
                job_id, 0, state.total_pages, f"배치 업로드 중 ({n}/{len(chunks)})"
            )
            data = b"\n".join(chunk) + b"\n"
            file_id = _batch_upload_file(
                base_url, api_key, data, f"lecai-{job_id}-{n}.jsonl"
            )
            try:
                batch = _batch_create(base_url, api_key, file_id, job_id)
            except BatchUnavailable:
                _file_delete(base_url, api_key, file_id)
                raise
            batches.append(
                {
                    "id": batch["id"],
                    "input_file_id": file_id,
                    "status": batch.get("status", "validating"),
                    "counts": batch.get("request_counts", {}) or {},
                    "output_file_id": None,
                    "error_file_id": None,
                }
            )
    except BatchUnavailable:
        for b in batches:
            _batch_cancel(base_url, api_key, b["id"])
            _file_delete(base_url, api_key, b["input_file_id"])
        raise

    ids_text = ", ".join(b["id"] for b in batches)
    JobManager.update_progress(
        job_id,
        0,
        state.total_pages,
        f"배치 등록 완료 ({ids_text}) | OpenAI 대기열에서 처리 중 (최대 24시간, 50% 할인)",
    )

    # ---- 폴링: 진행률/상태 변화가 있을 때만 로그를 남기고, 그 외엔 heartbeat 만 ----
    started = time.time()
    last_signature, last_log_at = None, time.time()
    while True:
        for b in batches:
            if b["status"] in BATCH_TERMINAL:
                continue
            try:
                info = _batch_get(base_url, api_key, b["id"])
            except Exception as e:
                print(f"[Batch] poll failed for {b['id']}: {e}")
                continue
            b["status"] = info.get("status", b["status"])
            b["counts"] = info.get("request_counts", {}) or b["counts"]
            b["output_file_id"] = info.get("output_file_id")
            b["error_file_id"] = info.get("error_file_id")

        completed = sum(b["counts"].get("completed", 0) for b in batches)
        failed = sum(b["counts"].get("failed", 0) for b in batches)
        statuses = sorted({b["status"] for b in batches})
        status_label = " / ".join(BATCH_STATUS_KO.get(s, s) for s in statuses)
        done = completed + failed

        signature = (completed, failed, tuple(statuses))
        now = time.time()
        if signature != last_signature or now - last_log_at >= BATCH_HEARTBEAT:
            msg = f"배치 {status_label} ({done}/{state.total_pages})"
            if failed:
                msg += f" | 실패 {failed}개 (실시간으로 재처리 예정)"
            JobManager.update_progress(job_id, done, state.total_pages, msg)
            last_signature, last_log_at = signature, now
        else:
            JobManager.update_progress(job_id, done, state.total_pages)

        if all(b["status"] in BATCH_TERMINAL for b in batches):
            break

        if now - started > BATCH_MAX_WAIT:
            JobManager.update_progress(
                job_id,
                done,
                state.total_pages,
                "배치 대기 시간 초과 → 취소 후 실시간 처리",
            )
            for b in batches:
                if b["status"] not in BATCH_TERMINAL:
                    _batch_cancel(base_url, api_key, b["id"])
            break

        time.sleep(BATCH_POLL_INTERVAL)

    # ---- 결과 수집 (부분 완료/만료된 배치의 결과도 최대한 회수) ----
    JobManager.update_progress(job_id, done, state.total_pages, "배치 결과 수집 중...")
    for b in batches:
        if b["output_file_id"]:
            try:
                raw = _file_content(base_url, api_key, b["output_file_id"])
            except Exception as e:
                print(f"[Batch] output download failed for {b['id']}: {e}")
                raw = ""
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    idx = int(row["custom_id"].split("-", 1)[1])
                    response = row.get("response") or {}
                    body = response.get("body") or {}
                    if response.get("status_code") != 200 or row.get("error"):
                        continue
                    content = body["choices"][0]["message"].get("content", "")
                    if not content or not content.strip():
                        continue
                    state.add_result(
                        idx,
                        os.path.basename(image_by_idx[idx]),
                        content,
                        parse_usage(body),
                        from_batch=True,
                    )
                except Exception as e:
                    print(f"[Batch] bad result line skipped: {e}")

        # 업로드/결과 파일은 사용자 계정 스토리지에 남으므로 정리
        _file_delete(base_url, api_key, b["input_file_id"])
        _file_delete(base_url, api_key, b["output_file_id"])
        _file_delete(base_url, api_key, b["error_file_id"])

    state.log_progress("배치 분석 완료")

    remaining = [(idx, p) for idx, p in items if idx not in state.results_map]
    return remaining


# ==========================================
# Main Processing Logic
# ==========================================


def _process_job_internal(job_id: str, file_path: str, model_config: dict, owner: str):
    """
    실제 파일 처리 로직 (실시간 비용 로그 포함)
    """
    work_dir = os.path.join(settings.UPLOAD_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        JobManager.start_processing(job_id)

        # 1. 이미지 변환 (PDF/PPT -> Images)
        images = []
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            raw_images = convert_from_path(file_path, fmt="png", dpi=150)
            for i, img in enumerate(raw_images):
                p = os.path.join(work_dir, f"page_{i + 1:03d}.png")
                img.save(p)
                images.append(p)
        elif ext in [".ppt", ".pptx"]:
            pdf_path = convert_ppt_to_pdf(file_path, work_dir)
            raw_images = convert_from_path(pdf_path, fmt="png", dpi=150)
            for i, img in enumerate(raw_images):
                p = os.path.join(work_dir, f"page_{i + 1:03d}.png")
                img.save(p)
                images.append(p)

        result_base = os.path.join(settings.RESULT_DIR, job_id)
        result_images_dir = os.path.join(result_base, "images")
        os.makedirs(result_images_dir, exist_ok=True)

        for img_path in images:
            shutil.copy2(
                img_path, os.path.join(result_images_dir, os.path.basename(img_path))
            )

        total_pages = len(images)
        state = AnalysisState(job_id, model_config, total_pages)
        items = list(enumerate(images, 1))
        image_by_idx = dict(items)

        # 2. LLM 분석 (Batch / 병렬 / 순차)
        if model_config["provider"] == "openai" and model_config.get("use_batch"):
            try:
                remaining = _run_batch(state, items, image_by_idx)
            except BatchUnavailable as e:
                print(f"[Batch] unavailable, falling back to realtime: {e}")
                JobManager.update_progress(
                    job_id, 0, total_pages, f"배치 등록 실패 → 실시간 처리로 전환 ({e})"
                )
                remaining = items
            if remaining:
                JobManager.update_progress(
                    job_id,
                    state.completed,
                    total_pages,
                    f"배치 미처리 슬라이드 {len(remaining)}개 → 실시간 재처리",
                )
                _run_realtime(state, remaining, PARALLEL_WORKERS, prefix="재처리 중")

        elif model_config["provider"] == "openai":
            _run_realtime(state, items, PARALLEL_WORKERS)

        else:
            # Local LLM: 순차 처리 (GPU 보호)
            _run_realtime(state, items, 1)

        results_map = state.results_map

        # 3. 결과 조합 (인덱스 순서대로)
        md_content = ""
        for idx in sorted(results_map.keys()):
            fname, text = results_map[idx]
            md_content += (
                f"## Slide {idx}\n\n![{fname}](./images/{fname})\n\n{text}\n\n---\n\n"
            )

        # 4. 최종 완료 처리
        if model_config["provider"] == "openai":
            usd_val, krw_val = state.cost()
            job = JobManager.get_job(job_id)
            if job:
                AuthManager.update_user_cumulative_usage(job["owner"], usd_val)
            final_log = f"작업 완료! 총 비용: ${usd_val} (약 ₩{krw_val:,}) | 총 토큰: {state.total_tokens:,}"
            if state.usage["cached"]:
                final_log += f" | 캐시 적중 토큰: {state.usage['cached']:,}"
            if state.batch_usage["prompt"]:
                final_log += " | Batch 50% 할인 적용"
        else:
            final_log = f"작업 완료! 총 토큰: {state.total_tokens:,}"
        JobManager.update_progress(job_id, total_pages, total_pages, final_log)

        # Markdown 저장
        md_file = os.path.join(result_base, "result.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        # PDF 생성
        import markdown

        raw_html = markdown.markdown(md_content)

        # 절대 경로 변환 (wkhtmltopdf 에러 방지)
        abs_image_dir = os.path.abspath(os.path.join(result_base, "images")).replace(
            "\\", "/"
        )
        pdf_html_body = raw_html.replace("./images", f"file://{abs_image_dir}")

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; }}
                img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; }}
                h2 {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-top: 30px; page-break-before: always; }}
                h2:first-of-type {{ page-break-before: auto; }}
                blockquote {{ background: #f9f9f9; border-left: 10px solid #ccc; margin: 1.5em 10px; padding: 0.5em 10px; }}
            </style>
        </head>
        <body>
            {pdf_html_body}
        </body>
        </html>
        """

        pdf_options = {
            "quiet": "",
            "enable-local-file-access": "",  # 필수
            "encoding": "UTF-8",
            "no-outline": None,
        }

        pdfkit.from_string(
            full_html, os.path.join(result_base, "result.pdf"), options=pdf_options
        )

        # 압축 및 정리
        user_result_dir = os.path.join(settings.RESULT_DIR, owner)
        os.makedirs(user_result_dir, exist_ok=True)
        shutil.make_archive(os.path.join(user_result_dir, job_id), "zip", result_base)

        # [Cleanup] 압축 후 원본 폴더 삭제
        if os.path.exists(result_base):
            shutil.rmtree(result_base)

        JobManager.mark_completed(job_id, f"/static/results/{owner}/{job_id}.zip")

    except Exception as e:
        JobManager.mark_failed(job_id, str(e))
    finally:
        # [Cleanup] 임시 작업 폴더 및 업로드 원본 삭제
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        if os.path.exists(file_path):
            os.remove(file_path)


def process_file_task(job_id: str, file_path: str):
    """
    Celery나 BackgroundTasks에서 호출되는 진입점
    """
    job = JobManager.get_job(job_id)
    if not job:
        return

    owner = job.get("owner")
    user_settings = AuthManager.get_user_settings(owner)

    try:
        model_config = get_target_model(user_settings)
    except Exception as e:
        JobManager.mark_failed(job_id, f"설정 오류: {str(e)}")
        return

    # 모델 타입에 따라 Lock 사용 여부 결정
    if model_config["provider"] == "local":
        print(f"[Queue] Job {job_id} is waiting for GPU lock...")
        with local_gpu_lock:
            _process_job_internal(job_id, file_path, model_config, owner)
    else:
        mode = "Batch" if model_config.get("use_batch") else "API"
        print(f"[Queue] Job {job_id} is starting immediately ({mode} Mode).")
        _process_job_internal(job_id, file_path, model_config, owner)
