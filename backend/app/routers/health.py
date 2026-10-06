from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.routers.deps import ServicesDep

router = APIRouter(tags=["health"])


def _ok(check: Callable[[], None]) -> bool:
    try:
        check()
        return True
    except Exception:
        return False


@router.get("/health")
def health(services: ServicesDep):
    checks = {
        "postgres": _ok(services.ping_db),
        "s3": _ok(services.storage.ping),
        "chroma": _ok(services.index.ping),
        "model": services.encoder is not None,
    }
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
