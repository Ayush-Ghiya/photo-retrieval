import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import register_error_handlers
from app.routers import health, images, search, tags
from app.services.container import Services, build_services


def create_app(services_factory: Callable[[], Services] | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)
        services = services_factory() if services_factory else build_services(get_settings())
        services.startup()
        app.state.services = services
        yield

    app = FastAPI(title="Photo Retrieval", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    for module in (images, search, tags, health):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
