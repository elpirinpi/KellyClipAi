from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = Path(
    os.getenv("KELLYCLIP_ENV_FILE", str(ROOT_DIR / ".env"))
).expanduser()
load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class AppSettings:
    data_dir: Path
    cors_origins: tuple[str, ...]
    llm_provider: str
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    whisper_model: str
    supabase_url: str = ""
    supabase_anon_key: str = ""
    oauth_redirect_base_url: str = "http://127.0.0.1:8000"
    frontend_base_url: str = "http://127.0.0.1:5173"
    youtube_client_id: str = ""
    youtube_client_secret: str = ""
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    meta_app_id: str = ""
    meta_app_secret: str = ""
    zoom_client_id: str = ""
    zoom_client_secret: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    broll_provider: str = "Pexels"
    processing_concurrency: int = 1

    @property
    def database_path(self) -> Path:
        return self.data_dir / "kellyclipai.sqlite3"


def get_settings() -> AppSettings:
    data_dir = Path(os.getenv("KELLYCLIP_DATA_DIR", str(ROOT_DIR / "data"))).expanduser()
    if not data_dir.is_absolute():
        data_dir = (ROOT_DIR / data_dir).resolve()
    origins = tuple(
        origin.strip().rstrip("/")
        for origin in os.getenv(
            "KELLYCLIP_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )
    return AppSettings(
        data_dir=data_dir,
        cors_origins=origins,
        llm_provider=os.getenv("LLM_PROVIDER", "Groq"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        supabase_url=os.getenv("SUPABASE_URL", "").rstrip("/"),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
        oauth_redirect_base_url=os.getenv(
            "KELLYCLIP_OAUTH_REDIRECT_BASE_URL", "http://127.0.0.1:8000"
        ).rstrip("/"),
        frontend_base_url=os.getenv(
            "KELLYCLIP_FRONTEND_BASE_URL", "http://127.0.0.1:5173"
        ).rstrip("/"),
        youtube_client_id=os.getenv("YOUTUBE_CLIENT_ID", ""),
        youtube_client_secret=os.getenv("YOUTUBE_CLIENT_SECRET", ""),
        tiktok_client_key=os.getenv("TIKTOK_CLIENT_KEY", ""),
        tiktok_client_secret=os.getenv("TIKTOK_CLIENT_SECRET", ""),
        meta_app_id=os.getenv("META_APP_ID", ""),
        meta_app_secret=os.getenv("META_APP_SECRET", ""),
        zoom_client_id=os.getenv("ZOOM_CLIENT_ID", ""),
        zoom_client_secret=os.getenv("ZOOM_CLIENT_SECRET", ""),
        pexels_api_key=os.getenv("PEXELS_API_KEY", ""),
        pixabay_api_key=os.getenv("PIXABAY_API_KEY", ""),
        broll_provider=os.getenv("BROLL_PROVIDER", "Pexels"),
        processing_concurrency=int(os.getenv("PROCESSING_CONCURRENCY", "1")),
    )
