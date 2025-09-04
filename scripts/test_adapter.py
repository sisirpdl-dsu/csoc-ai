import requests
"""Minimal TinyLlama LoRA adapter smoke test (CPU friendly).

Usage (PowerShell):
  $env:MODEL_STUB_MODE='false'
  $env:MODEL_ADAPTER_PATH='models/tinyllama_lora_adapter'
  $env:BASE_MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
  # Optional if private / 401 errors:
  # $env:HF_TOKEN='hf_xxx'
  python scripts/test_adapter.py "failed logins last 1 hour"
"""

import os, sys, json
from pathlib import Path

# ---------------------------------------------------------------------------
# OPTIONAL HARD-CODED HF TOKEN (NOT RECOMMENDED)
# Put your Hugging Face token between the quotes below ONLY for a quick test.
# SECURITY: This will embed the secret in the file. REMOVE before committing.
# Prefer: set $env:HF_TOKEN in PowerShell instead of editing the file.
HF_TOKEN_OVERRIDE = ""  # e.g. "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ---------------------------------------------------------------------------

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
except ImportError as e:
    print("Missing dependencies:", e)
    print("Install: pip install transformers peft accelerate safetensors")
    sys.exit(1)

USE_BACKEND = True  # Set to False to use local model
BACKEND_URL = "http://localhost:8000/generate-kql"
prompt = " ".join(sys.argv[1:]) or "failed logins last 1 hour"

if USE_BACKEND:
    print(f"[DEBUG] Sending prompt to backend: {prompt}")
    try:
        resp = requests.post(BACKEND_URL, json={"prompt": prompt})
        resp.raise_for_status()
        kql = resp.json().get("kql")
        print("\n[BACKEND] Generated KQL:\n" + kql)
    except Exception as e:
        print(f"[ERROR] Failed to get KQL from backend: {e}")
        sys.exit(1)
else:
    # --- Local model logic (unchanged) ---
    raw_adapter = os.getenv("MODEL_ADAPTER_PATH", "models/tinyllama_lora_adapter")
    adapter_path = Path(raw_adapter)
    # If relative and missing, try resolving relative to project root (parent of this script directory)
    if not adapter_path.exists():
        if not adapter_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            candidate = project_root / adapter_path
            if candidate.exists():
                adapter_path = candidate
    # Lightweight auto-discovery if still not found
    if not adapter_path.exists():
        project_root = Path(__file__).resolve().parent.parent
        found = []
        for p in project_root.rglob('adapter_config.json'):
            if any(skip in p.parts for skip in ('.venv', 'venv', '__pycache__', '.git')):
                continue
            found.append(p.parent)
            if len(found) > 5:
                break
        if len(found) == 1:
            adapter_path = found[0]
            print(f"[info] Auto-discovered adapter at: {adapter_path}")
        elif len(found) > 1:
            print("Adapter path not found and multiple candidates detected:")
            for f in found:
                print("  -", f)
            print("Set MODEL_ADAPTER_PATH to the correct one.")
            sys.exit(1)
    if not adapter_path.exists():
        print(f"Adapter path not found: {raw_adapter}\nTried resolved: {adapter_path.resolve()}\nRun from project root or set an absolute MODEL_ADAPTER_PATH.")
        sys.exit(1)
    base_model = os.getenv("BASE_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    cfg = adapter_path / "adapter_config.json"
    if cfg.exists():
        try:
            cfg_data = json.loads(cfg.read_text())
            bm = cfg_data.get("base_model_name_or_path") or cfg_data.get("base_model")
            if bm:
                base_model = bm
        except Exception:
            pass
    HF_TOKEN_OVERRIDE=""  # (cleared) DO NOT commit real tokens
    token = HF_TOKEN_OVERRIDE or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if HF_TOKEN_OVERRIDE:
        print("[warn] Using hard-coded HF token override (remove before commit).")
    if token:
        print(f"[debug] HF token detected (length={len(token)}). Not printing for safety.")
    else:
        print("[debug] No HF token detected (env HF_TOKEN / HUGGING_FACE_HUB_TOKEN empty).")
    print(f"Base model: {base_model}\nAdapter:    {adapter_path}\nPrompt:     {prompt}")
    dtype = torch.float32  # CPU safe
    load_kwargs = {"torch_dtype": dtype}
    try:
        if token:
            # Some versions use 'token', older use 'use_auth_token'
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model, token=token, **load_kwargs)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(base_model, use_auth_token=token, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except Exception as e:
        print("Failed to load base model:", e)
        if not token:
            print("Hint: set HF_TOKEN if the model is gated or you get 401 errors.")
        sys.exit(2)
    try:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    except Exception as e:
        print("Failed to load adapter:", e)
        sys.exit(3)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prefix = (
        "You are an expert KQL assistant for Microsoft Sentinel. Your task is to convert a user's natural language request into a single, valid KQL query.\n\n"
        "Rules:\n"
        "1.  Output ONLY the raw KQL query. Do not include any commentary, explanations, or markdown backticks.\n"
        "2.  Always include a bounded time filter (e.g., `... | where TimeGenerated > ago(24h)`).\n"
        "3.  If the user provides a specific entity (like a username, IP, or filename), use it directly in the query.\n"
        "4.  If the user's request is generic (e.g., 'a user' or 'an IP'), use a realistic but clearly example entity like 'john.doe@example.com' or '198.51.100.99'.\n"
        "5.  Use the most relevant and common fields for the specified log table and task.\n\n"
        "---\nExample:\n\nUser Request: Show me both successful and failed logins for john.doe@trojans.dsu.edu in the last day.\n\n"
        "KQL Output:\nSigninLogs | where UserPrincipalName =~ 'john.doe@trojans.dsu.edu' and TimeGenerated > ago(1d) | summarize SuccessfulLogins = countif(ResultType == 0), FailedLogins = countif(ResultType != 0) | project TimeGenerated, IPAddress, Location, AppDisplayName, ResultType, ResultDescription\n---\n"
    )
    # Optional few-shot examples (enable by setting KQL_FEWSHOT=1)
    fewshot_enabled = os.getenv("KQL_FEWSHOT") in {"1", "true", "True"}
    fewshots = []
    if fewshot_enabled:
        fewshots = [
            (
                "failed sign-ins last 1 hour",
                "SigninLogs\n| where TimeGenerated >= ago(1h)\n| where ResultType != 0\n| project TimeGenerated, UserPrincipalName, AppDisplayName, IPAddress, ResultType"
            ),
            (
                "successful sign-ins from new country last 24 hours",
                "let recent = SigninLogs\n  | where TimeGenerated >= ago(24h)\n  | summarize Countries = make_set(LocationDetails.countryOrRegion) by UserPrincipalName;\nSigninLogs\n| where TimeGenerated >= ago(24h)\n| summarize SeenCountries = make_set(LocationDetails.countryOrRegion) by UserPrincipalName\n| join kind=inner recent on UserPrincipalName\n| extend NewCountries = set_difference(SeenCountries, Countries)\n| where array_length(NewCountries) > 0\n| project TimeGenerated, UserPrincipalName, NewCountries"
            ),
            (
                "accounts added to admin groups last 12 hours",
                "AuditLogs\n| where TimeGenerated >= ago(12h)\n| where OperationName has ""Add member to role"" or OperationName has ""Add member to group""\n| where TargetResources has ""Admin"" or InitiatedBy has ""Admin""\n| project TimeGenerated, OperationName, InitiatedBy = InitiatedBy.user.userPrincipalName, Targets = TargetResources"
            ),
        ]
    fewshot_block = ""
    if fewshots:
        parts = []
        for q, kql in fewshots:
            parts.append(f"Request: {q}\nKQL:\n{kql}\n---\n")
        fewshot_block = "".join(parts)
    full_prompt = f"{system_prefix}{fewshot_block}Request: {prompt}\nKQL:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    # Keep only newly generated tokens to avoid echoing few-shots/system prompt
    gen_only = output_ids[0][input_len:]
    text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    # If model still includes the delimiter, trim to content after last KQL:
    if "KQL:" in text:
        text = text.split("KQL:")[-1]
    print("\nGenerated KQL:\n" + text.strip())

