# app/services/auth_manager.py

import uuid
import hashlib
import bcrypt
import random
from app.services.email_service import send_verification_email
from app.db import users_col, sessions_col
from app.db.prompt import default_system_prompt, default_user_prompt
from app.core.config import settings

DEFAULT_MODEL = settings.DEFAULT_MODEL

verification_codes = {}


class AuthManager:
    @staticmethod
    def _pre_hash(password: str) -> bytes:
        """
        bcrypt의 72바이트 제한을 우회하기 위해
        SHA-256으로 먼저 해싱하여 64글자(bytes)로 고정합니다.
        """
        return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("utf-8")

    @staticmethod
    def request_signup(username, password, email):
        # 1. 사용자명 중복 체크 (MongoDB)
        if users_col.find_one({"username": username}):
            return "username_exists"

        # 2. 이메일 중복 체크 (MongoDB)
        if users_col.find_one({"email": email}):
            return "email_exists"

        # 인증 코드 생성
        code = str(random.randint(100000, 999999))

        # 메일 발송
        if send_verification_email(email, code):
            # 임시 저장 (검증용)
            verification_codes[email] = {
                "username": username,
                "password": password,
                "code": code,
            }
            return "success"
        else:
            return "mail_failed"

    @staticmethod
    def verify_and_create_user(email, code):
        data = verification_codes.get(email)
        if not data:
            return False  # 요청 내역 없음

        if data["code"] != code:
            return False  # 코드 불일치

        # 검증 완료 -> 실제 계정 생성 준비
        username = data["username"]
        password = data["password"]

        # 비밀번호 해싱
        safe_pw = AuthManager._pre_hash(password)
        hashed_pw = bcrypt.hashpw(safe_pw, bcrypt.gensalt()).decode("utf-8")

        # MongoDB에 저장할 문서 구조
        new_user = {
            "username": username,
            "password": hashed_pw,
            "email": email,
            "verified": True,
            "openai_api_key": "",
            "preferred_model": DEFAULT_MODEL,
            "use_batch_api": False,
            "audio_language": "auto",
            "audio_model_level": 2,
        }

        # DB 저장
        users_col.insert_one(new_user)

        # 임시 데이터 삭제
        del verification_codes[email]
        return True

    @staticmethod
    def create_user(username, password):
        if users_col.find_one({"username": username}):
            return False

        safe_pw = AuthManager._pre_hash(password)
        hashed_pw = bcrypt.hashpw(safe_pw, bcrypt.gensalt()).decode("utf-8")

        new_user = {
            "username": username,
            "password": hashed_pw,
            "email": "",
            "verified": False,
            "openai_api_key": "",
            "preferred_model": DEFAULT_MODEL,
            "use_batch_api": False,
            "audio_language": "auto",
            "audio_model_level": 2,
        }

        users_col.insert_one(new_user)
        return True

    @staticmethod
    def authenticate_user(username, password):
        # DB에서 유저 조회
        user = users_col.find_one({"username": username})
        if not user:
            return None

        stored_hash = user["password"].encode("utf-8")
        safe_pw = AuthManager._pre_hash(password)

        # 비밀번호 검증
        if bcrypt.checkpw(safe_pw, stored_hash):
            session_id = str(uuid.uuid4())

            # 세션 DB에 저장
            sessions_col.insert_one({"session_id": session_id, "username": username})
            return session_id
        return None

    @staticmethod
    def get_user_by_session(session_id):
        # DB에서 세션 조회
        session = sessions_col.find_one({"session_id": session_id})
        if session:
            return session["username"]
        return None

    @staticmethod
    def logout(session_id):
        # DB에서 세션 삭제
        sessions_col.delete_one({"session_id": session_id})

    @staticmethod
    def update_user_settings(
        username,
        api_key,
        model_choice,
        audio_lang="auto",
        audio_model=2,
        custom_prompt=None,
        custom_user_prompt=None,
        profile_url=None,
        use_batch_api=False,
    ):
        update_data = {
            "openai_api_key": api_key,
            "preferred_model": model_choice,
            "audio_language": audio_lang,
            "audio_model_level": int(audio_model),
            "use_batch_api": bool(use_batch_api),
        }

        if custom_prompt is not None:
            update_data["custom_prompt"] = custom_prompt

        if custom_user_prompt is not None:
            update_data["custom_user_prompt"] = custom_user_prompt

        if profile_url:
            update_data["profile_img"] = profile_url

        result = users_col.update_one({"username": username}, {"$set": update_data})
        return result.matched_count > 0

    @staticmethod
    def reset_prompts_to_default(known_defaults=()):
        """
        모든 사용자의 custom_prompt / custom_user_prompt 를 제거하여 현재 기본 프롬프트
        (app/db/prompt.py)를 사용하도록 한다. 필드를 지워 두면 이후 기본값이 바뀔 때도
        자동으로 따라간다.

        known_defaults: 과거 기본 프롬프트 문자열 목록. 저장된 값이 이 목록(또는 현재 기본값)과
        일치하면 그대로 제거하고, 사용자가 직접 수정한 프롬프트라면 custom_prompt_prev /
        custom_user_prompt_prev 에 백업한 뒤 제거한다.
        반환값: {"reset": [...], "backed_up": [...], "untouched": [...]}
        """
        import re

        def norm(text):
            return re.sub(r"[\s*`]+", "", text or "")

        defaults = {norm(default_system_prompt), norm(default_user_prompt)}
        defaults.update(norm(d) for d in known_defaults)

        report = {"reset": [], "backed_up": [], "untouched": []}
        for user in users_col.find(
            {}, {"username": 1, "custom_prompt": 1, "custom_user_prompt": 1}
        ):
            unset, backup = {}, {}
            for field in ("custom_prompt", "custom_user_prompt"):
                if field not in user:
                    continue
                value = user[field]
                unset[field] = ""
                if value and value.strip() and norm(value) not in defaults:
                    backup[f"{field}_prev"] = value

            if not unset:
                report["untouched"].append(user["username"])
                continue

            update = {"$unset": unset}
            if backup:
                update["$set"] = backup
            users_col.update_one({"_id": user["_id"]}, update)
            (report["backed_up"] if backup else report["reset"]).append(
                user["username"]
            )

        return report

    @staticmethod
    def update_preferred_model(username, model_choice):
        result = users_col.update_one(
            {"username": username}, {"$set": {"preferred_model": model_choice}}
        )
        return result.matched_count > 0

    @staticmethod
    def get_user_settings(username):
        user = users_col.find_one({"username": username})

        if user:
            return {
                "openai_api_key": user.get("openai_api_key", ""),
                "preferred_model": user.get("preferred_model", DEFAULT_MODEL),
                "use_batch_api": bool(user.get("use_batch_api", False)),
                "audio_language": user.get("audio_language", "auto"),
                "audio_model_level": user.get("audio_model_level", 2),
                "custom_prompt": user.get("custom_prompt", default_system_prompt),
                "custom_user_prompt": user.get(
                    "custom_user_prompt", default_user_prompt
                ),
                "profile_img": user.get("profile_img", "/static/default_avatar.png"),
            }

        return {
            "openai_api_key": "",
            "preferred_model": DEFAULT_MODEL,
            "use_batch_api": False,
            "audio_language": "auto",
            "audio_model_level": 2,
            "custom_prompt": default_system_prompt,
            "custom_user_prompt": default_user_prompt,
        }

    @staticmethod
    def update_user_cumulative_usage(username: str, cost_usd: float):
        users_col.update_one(
            {"username": username}, {"$inc": {"total_spent_usd": cost_usd}}
        )

    @staticmethod
    def get_user_usage(username: str):
        user = users_col.find_one({"username": username}, {"total_spent_usd": 1})
        if user and "total_spent_usd" in user:
            return user["total_spent_usd"]
        return 0.0
