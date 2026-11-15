"""
tests/test_web_authz.py - 測試 Web UI 授權、Session 與角色權限控制 (P12)
"""

from web_ui.auth import AuthManager, Role


def test_auth_manager_session_lifecycle():
    auth = AuthManager(secret_key="test-secret")
    session_id = auth.create_session("alice", Role.OPERATOR, ttl=3600)
    assert session_id is not None

    session = auth.get_session(session_id)
    assert session is not None
    assert session["username"] == "alice"
    assert session["role"] == Role.OPERATOR


def test_auth_manager_permissions():
    auth = AuthManager(secret_key="test-secret")
    viewer_id = auth.create_session("viewer_user", Role.VIEWER)
    operator_id = auth.create_session("operator_user", Role.OPERATOR)
    admin_id = auth.create_session("admin_user", Role.ADMIN)

    # 訪客 / 匿名
    assert auth.check_permission(None, Role.VIEWER) is True
    assert auth.check_permission(None, Role.OPERATOR) is False

    # VIEWER
    assert auth.check_permission(viewer_id, Role.VIEWER) is True
    assert auth.check_permission(viewer_id, Role.OPERATOR) is False

    # OPERATOR
    assert auth.check_permission(operator_id, Role.VIEWER) is True
    assert auth.check_permission(operator_id, Role.OPERATOR) is True
    assert auth.check_permission(operator_id, Role.ADMIN) is False

    # ADMIN
    assert auth.check_permission(admin_id, Role.ADMIN) is True


def test_csrf_token_validation():
    auth = AuthManager()
    session_id = auth.create_session("bob", Role.ADMIN)
    session = auth.get_session(session_id)
    valid_csrf = session["csrf_token"]

    assert auth.validate_csrf(session_id, valid_csrf) is True
    assert auth.validate_csrf(session_id, "invalid_csrf_token") is False
    assert auth.validate_csrf("non_existent_session", valid_csrf) is False
