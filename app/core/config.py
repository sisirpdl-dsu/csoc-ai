from functools import lru_cache
from typing import List, Optional
from pydantic import BaseModel
try:  # pydantic v2
    from pydantic import ConfigDict  # type: ignore
except Exception:  # v1 fallback
    ConfigDict = dict  # type: ignore
from dotenv import load_dotenv  # type: ignore
try:  # Pydantic v2
    from pydantic import field_validator  # type: ignore
except ImportError:  # Fallback for v1
    def field_validator(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator
import os

# Load .env once at import so os.getenv sees variables (idempotent if already loaded)
load_dotenv(override=False)

if hasattr(BaseModel, "model_config"):  # Pydantic v2
    class Settings(BaseModel):
        model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)  # type: ignore[arg-type]
        # Azure / Sentinel
        tenant_id: Optional[str] = None
        client_id: Optional[str] = None
        client_secret: Optional[str] = None
        log_workspace_id: Optional[str] = None
        defender_enabled: bool = False  # toggle for Defender Advanced Hunting API

        # App
        environment: str = "dev"
        max_rows: int = 500
        model_stub_mode: bool = True  # toggle when real model integrated

        # Security (future IP allowlist)
        ip_allowlist: str = ""  # comma separated

        @field_validator("ip_allowlist", mode="before")  # type: ignore[misc]
        @classmethod
        def _normalize_ips(cls, v):  # type: ignore
            if not v:
                return ""
            return ",".join([i.strip() for i in str(v).split(",") if i.strip()])

        @property
        def ip_list(self) -> List[str]:
            return [i for i in self.ip_allowlist.split(",") if i] if self.ip_allowlist else []
else:  # Pydantic v1 fallback
    class _SettingsV1(BaseModel):
        class Config:  # type: ignore
            arbitrary_types_allowed = True

        tenant_id: Optional[str] = None
        client_id: Optional[str] = None
        client_secret: Optional[str] = None
        log_workspace_id: Optional[str] = None
        defender_enabled: bool = False
        environment: str = "dev"
        max_rows: int = 500
        model_stub_mode: bool = True
        ip_allowlist: str = ""

        @classmethod  # v1 validator style compatibility (no-op for simplicity)
        def _normalize_ips(cls, v):  # type: ignore
            if not v:
                return ""
            return ",".join([i.strip() for i in str(v).split(",") if i.strip()])

        @property
        def ip_list(self) -> List[str]:
            return [i for i in self.ip_allowlist.split(",") if i] if self.ip_allowlist else []

    Settings = _SettingsV1  # type: ignore

@lru_cache()
def get_settings() -> Settings:
    """Build Settings from environment variables (simple manual load).

    Environment variables supported:
      TENANT_ID, CLIENT_ID, CLIENT_SECRET, LOG_WORKSPACE_ID,
      ENVIRONMENT, MAX_ROWS, MODEL_STUB_MODE, IP_ALLOWLIST

    Boolean parsing for MODEL_STUB_MODE accepts: 1, true, yes, on (case-insensitive).
    """

    def _bool(name: str, default: str = "false") -> bool:
        raw = os.getenv(name, default).strip()
        # remove optional surrounding quotes
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()
        return raw.lower() in {"1", "true", "yes", "on"}

    def _int(name: str, default: str) -> int:
        val = os.getenv(name, default)
        try:
            return int(val)
        except ValueError:
            return int(default)

    return Settings(  # type: ignore[arg-type]
        tenant_id=os.getenv("TENANT_ID"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        log_workspace_id=os.getenv("LOG_WORKSPACE_ID"),
        defender_enabled=_bool("DEFENDER_ENABLED", "false"),
        environment=os.getenv("ENVIRONMENT", "dev"),
        max_rows=_int("MAX_ROWS", "500"),
        model_stub_mode=_bool("MODEL_STUB_MODE", "true"),
        ip_allowlist=os.getenv("IP_ALLOWLIST", ""),
    )
