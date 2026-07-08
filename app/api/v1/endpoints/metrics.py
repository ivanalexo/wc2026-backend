"""Métricas de descargas del calendario (.ics): navegador + geo por descarga."""
import logging

import requests
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models.download_event import DownloadEvent
from app.dependencies import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["Metrics"])


def _verify_secret(x_metrics_secret: str | None = Header(default=None)) -> None:
    """Valida X-Metrics-Secret contra SYNC_SECRET (mismo secreto que el sync)."""
    if not settings.sync_secret:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "SYNC_SECRET no configurado")
    if x_metrics_secret != settings.sync_secret:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "X-Metrics-Secret inválido")


def _parse_ua(ua: str) -> tuple[str, str]:
    """Extrae (navegador, SO) de forma aproximada, sin dependencias."""
    u = ua.lower()
    if "edg/" in u or "edga" in u:
        browser = "Edge"
    elif "crios" in u:
        browser = "Chrome"
    elif "fxios" in u:
        browser = "Firefox"
    elif "opr/" in u or "opera" in u:
        browser = "Opera"
    elif "chrome" in u or "chromium" in u:
        browser = "Chrome"
    elif "firefox" in u:
        browser = "Firefox"
    elif "safari" in u:
        browser = "Safari"
    else:
        browser = "Otro"

    if "iphone" in u or "ipad" in u or "ios" in u:
        os = "iOS"
    elif "android" in u:
        os = "Android"
    elif "windows" in u:
        os = "Windows"
    elif "mac os" in u or "macintosh" in u:
        os = "macOS"
    elif "linux" in u:
        os = "Linux"
    else:
        os = "Otro"
    return browser, os


def _resolve_geo(ip: str | None) -> tuple[str | None, str | None]:
    """País/ciudad desde la IP (best-effort, ip-api.com free). No bloquea al usuario."""
    if not ip or ip.startswith(("127.", "10.", "192.168.", "172.")) or ip in ("::1", "localhost"):
        return None, None
    try:
        r = requests.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,country,city"},
            timeout=3,
        )
        d = r.json()
        if d.get("status") == "success":
            return d.get("country"), d.get("city")
    except (requests.RequestException, ValueError):
        pass
    return None, None


class DownloadEventIn(BaseModel):
    round: str
    user_agent: str | None = None
    ip: str | None = None


@router.post("/download", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
def record_download(
    body: DownloadEventIn,
    db: Session = Depends(get_db),
    _: None = Depends(_verify_secret),
) -> Response:
    ua = (body.user_agent or "")[:300]
    browser, os = _parse_ua(ua) if ua else (None, None)
    country, city = _resolve_geo(body.ip)

    db.add(DownloadEvent(
        round=body.round[:30],
        browser=browser,
        os=os,
        user_agent=ua or None,
        country=country,
        city=city,
    ))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/downloads", include_in_schema=False)
def download_stats(
    db: Session = Depends(get_db),
    _: None = Depends(_verify_secret),
) -> dict:
    """Resumen agregado de descargas (para revisar con un curl)."""
    def counts(col):
        rows = db.query(col, func.count()).group_by(col).order_by(func.count().desc()).all()
        return {(k or "—"): n for k, n in rows}

    total = db.query(func.count(DownloadEvent.id)).scalar() or 0
    recent = (
        db.query(DownloadEvent)
        .order_by(DownloadEvent.created_at.desc())
        .limit(20)
        .all()
    )
    return {
        "total": total,
        "by_round": counts(DownloadEvent.round),
        "by_country": counts(DownloadEvent.country),
        "by_browser": counts(DownloadEvent.browser),
        "by_os": counts(DownloadEvent.os),
        "recent": [
            {
                "round": e.round, "browser": e.browser, "os": e.os,
                "country": e.country, "city": e.city,
                "at": e.created_at.isoformat(),
            }
            for e in recent
        ],
    }
