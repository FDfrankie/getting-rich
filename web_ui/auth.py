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

    def __init__(self, secret_key: str | None = None, max_sessions: int = 1000):
        import threading
        self.secret_key = secret_key or secrets.token_hex(32)
        self._lock = threading.Lock()
        self._max_sessions = max(10, int(max_sessions))
        self.sessions: dict[str, dict[str, Any]] = {}

    def create_session(self, username: str, role: str = Role.VIEWER, ttl: int = 86400) -> str:
        now = time.time()
        session_id = secrets.token_urlsafe(32)
        with self._lock:
            expired = [sid for sid, s in self.sessions.items() if now > s.get("expires_at", 0)]
            for sid in expired:
                self.sessions.pop(sid, None)
            while len(self.sessions) >= self._max_sessions:
                oldest = next(iter(self.sessions))
                self.sessions.pop(oldest, None)
            self.sessions[session_id] = {
                "username": username,
                "role": role,
                "expires_at": now + ttl,
                "csrf_token": secrets.token_hex(16),
            }
        return session_id

    def get_session(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            if time.time() > session["expires_at"]:
                self.sessions.pop(session_id, None)
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
