import hmac
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

bearer = HTTPBearer(auto_error=False)


def authenticate(username: str, password: str) -> bool:
    expected = settings.users.get(username)
    return expected is not None and hmac.compare_digest(expected, password)


def make_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_hours)
    return jwt.encode({"sub": username, "exp": exp}, settings.jwt_secret, algorithm="HS256")


def current_user(cred: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    if not cred:
        raise HTTPException(401, "Not authenticated")
    try:
        return jwt.decode(cred.credentials, settings.jwt_secret, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
