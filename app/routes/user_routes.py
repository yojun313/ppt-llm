# app/routes/user_routes.py
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional, Any
from pydantic import BaseModel
from app.services.auth_manager import AuthManager
from app.routes.deps import get_current_user
from app.db.prompt import default_system_prompt, default_user_prompt
import shutil
import os

router = APIRouter()


class ModelUpdateRequest(BaseModel):
    preferred_model: str


@router.post("/settings")
async def save_settings(
    model: str = Form(...),
    api_key: str = Form(""),
    audio_lang: str = Form("ko"),
    audio_model: str = Form("2"),
    custom_prompt: Optional[str] = Form(None),
    custom_user_prompt: Optional[str] = Form(None),
    use_batch: str = Form("0"),
    profile_img: Optional[UploadFile] = File(None),
    user: Any = Depends(get_current_user),
):
    use_batch_api = use_batch.strip().lower() in ("1", "true", "on", "yes")
    try:
        int_audio_model = int(audio_model)
    except:
        int_audio_model = 2

    profile_url = None
    if profile_img and profile_img.filename:
        profile_dir = "static/profiles"
        os.makedirs(profile_dir, exist_ok=True)

        ext = os.path.splitext(profile_img.filename)[1]
        file_path = os.path.join(profile_dir, f"{user}{ext}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(profile_img.file, buffer)
        profile_url = f"/{file_path}"

    success = AuthManager.update_user_settings(
        user,
        api_key,
        model,
        audio_lang,
        int_audio_model,
        custom_prompt,
        custom_user_prompt,
        profile_url,
        use_batch_api,
    )

    if success:
        return {"status": "success", "profile_url": profile_url}
    else:
        raise HTTPException(status_code=400, detail="설정 저장 실패")


@router.get("/settings/usage")
async def get_usage_info(user: str = Depends(get_current_user)):
    total_usd = AuthManager.get_user_usage(user)
    return {"total_spent_usd": round(total_usd, 4)}


@router.get("/settings/default_prompts")
async def get_default_prompts(user: str = Depends(get_current_user)):
    return {"system_prompt": default_system_prompt, "user_prompt": default_user_prompt}


@router.post("/user/profile-image")
async def upload_profile_image(
    file: UploadFile = File(...), user: str = Depends(get_current_user)
):
    os.makedirs("static/profiles", exist_ok=True)

    ext = os.path.splitext(file.filename)[1]
    file_path = f"static/profiles/{user}{ext}"
    profile_url = f"/{file_path}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from app.db import users_col

    users_col.update_one({"username": user}, {"$set": {"profile_img": profile_url}})

    return {"status": "success", "url": profile_url}


@router.post("/settings/model")
async def update_model_setting(
    req: ModelUpdateRequest, user: str = Depends(get_current_user)
):
    if req.preferred_model != "local":
        current_settings = AuthManager.get_user_settings(user)
        api_key = current_settings.get("openai_api_key")

        if not api_key or not api_key.strip():
            raise HTTPException(
                status_code=400,
                detail="OpenAI API Key가 설정되지 않았습니다. 설정 메뉴에서 먼저 키를 등록해주세요.",
            )

    success = AuthManager.update_preferred_model(user, req.preferred_model)

    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {"status": "success", "model": req.preferred_model}
