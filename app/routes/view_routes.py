# app/routes/view_routes.py
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from app.services.auth_manager import AuthManager
from app.db.prompt import default_system_prompt, default_user_prompt
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def index(request: Request):
    session_id = request.cookies.get("session_id")
    user = AuthManager.get_user_by_session(session_id)

    if not user:
        return RedirectResponse(url="/login")

    user_settings = AuthManager.get_user_settings(user)

    return templates.TemplateResponse(
        request, "dashboard.html", {"username": user, "settings": user_settings}
    )


@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@router.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html")


@router.get("/settings")
async def settings_page(request: Request):
    session_id = request.cookies.get("session_id")
    user = AuthManager.get_user_by_session(session_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    user_settings = AuthManager.get_user_settings(user)

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "username": user,
            "settings": user_settings,
            "default_system_prompt": default_system_prompt,
            "default_user_prompt": default_user_prompt,
        },
    )


@router.get("/viewer")
async def viewer_page(request: Request):
    session_id = request.cookies.get("session_id")
    user = AuthManager.get_user_by_session(session_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    return templates.TemplateResponse(request, "viewer.html", {"username": user})


@router.get("/guide/openai", response_class=HTMLResponse)
async def get_openai_guide(request: Request):
    return templates.TemplateResponse(request, "guide_openai.html")
