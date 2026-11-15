"""
web_ui/auth.py - Web 認證與授權 (Role-Based Access Control)
提供 VIEWER / OPERATOR / ADMIN 三級角色、安全的 Session 管理與 CSRF 防護。
"""

import hmac
import secrets
import time
from typing import Any


class Role:
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"
    ADMIN = "ADMIN"


class AuthManager:
    """管理 Web UI 授權、Session 與角色權限"""

    def __init__(self, secret_key: str | None = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.csrf_tokens: dict[str, float] = {}

    def create_session(self, username: str, role: str = Role.VIEWER, ttl: int = 86400) -> str:
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "username": username,
            "role": role,
            "expires_at": time.time() + ttl,
            "csrf_token": secrets.token_hex(16),
        }
        return session_id

    def get_session(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id or session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        if time.time() > session["expires_at"]:
            del self.sessions[session_id]
            return None
        return session

    def validate_csrf(self, session_id: str, csrf_token: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        return hmac.compare_digest(session.get("csrf_token", ""), csrf_token)

    def check_permission(self, session_id: str | None, required_role: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            # 預設訪客僅擁有 VIEWER 唯讀權限
            return required_role == Role.VIEWER

        role = session.get("role", Role.VIEWER)
        role_hierarchy = {Role.VIEWER: 1, Role.OPERATOR: 2, Role.ADMIN: 3}
        return role_hierarchy.get(role, 0) >= role_hierarchy.get(required_role, 1)


auth_manager = AuthManager()
