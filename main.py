from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings, Settings
from app.models.schemas import (
    GenerateKQLRequest,
    GenerateKQLResponse,
    RunKQLRequest,
    RunKQLResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    SummaryModel,
    ErrorResponse,
)
from app.services.model_provider import ModelProvider
from app.services.log_query import SentinelLogService
from app.services.kql_guard import enforce_or_fix, validate_kql
from datetime import timedelta
import uvicorn

app = FastAPI(title="AI SOC Analyst Backend", version="0.1.0")

# Basic CORS (adjust origins later as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
settings = get_settings()
print(f"[startup] MODEL_STUB_MODE={settings.model_stub_mode}")
model_provider = ModelProvider(stub_mode=settings.model_stub_mode)
log_service = SentinelLogService()
print("[startup] Sentinel log service mode=", "enabled" if log_service.enabled else "stub")

def check_ip(request: Request, settings: Settings = Depends(get_settings)):
    if settings.ip_list:
        client_ip = request.client.host if request.client else "unknown"
        if client_ip not in settings.ip_list:
            raise HTTPException(status_code=403, detail="IP not allowed")
    return True

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Pydantic v2: use model_dump(); still works as .dict() fallback if downgraded
    err = ErrorResponse(detail=str(exc))
    # getattr used to satisfy static analyzer across pydantic versions
    content = getattr(err, "model_dump", None)
    if callable(content):  # pydantic v2
        content = content()
    else:  # pydantic v1
        content = err.dict()
    
    return JSONResponse(status_code=500, content=content)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_stub_mode": settings.model_stub_mode,
        "log_query_mode": "enabled" if log_service.enabled else "stub",
    }

@app.post("/generate-kql", response_model=GenerateKQLResponse, dependencies=[Depends(check_ip)])
async def generate_kql(req: GenerateKQLRequest):
    kql = model_provider.generate_kql(req.prompt, context=req.context)
    guarded = enforce_or_fix(kql)
    ok, issues = validate_kql(guarded)
    meta = {"guard_ok": ok, "guard_issues": issues}
    return GenerateKQLResponse(kql=guarded)

@app.post("/run-kql", response_model=RunKQLResponse, dependencies=[Depends(check_ip)])
async def run_kql(req: RunKQLRequest):
    result = log_service.query(req.kql, timespan=timedelta(hours=req.hours))
    return RunKQLResponse(**result)

@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(check_ip)])
async def analyze(req: AnalyzeRequest):
    kql_raw = model_provider.generate_kql(req.prompt, context=req.context)
    kql = enforce_or_fix(kql_raw, hours=req.hours)
    log_result = log_service.query(kql, timespan=timedelta(hours=req.hours))
    logs = log_result.get("rows", [])
    summary_dict = model_provider.summarize(logs, req.prompt)
    summary = SummaryModel(**summary_dict)
    metadata = {
        "log_query_mode": log_result.get("mode"),
        "statistics": log_result.get("statistics"),
    }
    ok, issues = validate_kql(kql)
    metadata.update({"guard_ok": ok, "guard_issues": issues})
    return AnalyzeResponse(kql=kql, logs=logs, summary=summary, metadata=metadata)

@app.post("/admin/reload-model")
async def reload_model(stub_mode: bool | None = None):
    """Hot-reload the model provider. Optional stub_mode override.

    If stub_mode is provided it will recreate the provider with that mode.
    Otherwise keeps current mode and re-initializes (useful after adding adapter or token).
    """
    global model_provider, settings
    if stub_mode is not None:
        settings.model_stub_mode = stub_mode  # type: ignore[attr-defined]
    # Recreate instance (lazy load honored)
    model_provider = ModelProvider(stub_mode=settings.model_stub_mode)
    mode = "stub" if settings.model_stub_mode else "real"
    return {"status": "reloaded", "mode": mode}

@app.get("/model-info")
async def model_info():
    return model_provider.get_info()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
