from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import settings, print_settings_summary
from app.services.loader import load_all_resources

app = FastAPI(title=settings.app_title, version=settings.app_version)
app.include_router(api_router)


@app.on_event("startup")
def startup_event():
    try:
        print_settings_summary()
        load_all_resources()
    except Exception as e:
        print(f"[Startup][ERROR] {e}")
        raise


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
