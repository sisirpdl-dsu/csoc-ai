from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import timedelta

class GenerateKQLRequest(BaseModel):
    prompt: str = Field(..., description="Natural language request from analyst")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional structured context/incident JSON")

class GenerateKQLResponse(BaseModel):
    kql: str

class RunKQLRequest(BaseModel):
    kql: str
    hours: int = Field(default=24, ge=1, le=168)

class RunKQLResponse(BaseModel):
    mode: str
    kql: str
    rows: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    error: Optional[str] = None

class AnalyzeRequest(BaseModel):
    prompt: str
    hours: int = Field(default=24, ge=1, le=168)
    context: Optional[Dict[str, Any]] = None

class SummaryModel(BaseModel):
    paragraph: str
    bullet_points: List[str]

class AnalyzeResponse(BaseModel):
    kql: str
    logs: List[Dict[str, Any]]
    summary: SummaryModel
    metadata: Dict[str, Any]

class ErrorResponse(BaseModel):
    detail: str
    code: str = Field(default="error")
