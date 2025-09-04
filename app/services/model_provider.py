from typing import List, Dict, Any, Optional, TYPE_CHECKING
import json
import os
from functools import lru_cache

try:  # pragma: no cover - optional heavy deps
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
    from peft import PeftModel
    import torch
except Exception:  # Dependencies missing in stub/inference-light installs
    AutoModelForCausalLM = AutoTokenizer = PeftModel = None  # type: ignore
    torch = None  # type: ignore

class ModelProvider:
    """Abstracts the LLM interaction. Currently a stub; replace with fine-tuned Mistral later."""
    def __init__(self, stub_mode: bool = True, adapter_path: Optional[str] = None, base_model_name: Optional[str] = None, max_new_tokens: int = 256):
        self.stub_mode = stub_mode
        self.adapter_path = adapter_path or os.getenv("MODEL_ADAPTER_PATH")
        # Default now points to Phi-3 Mini 4K unless overridden
        self.base_model_name = base_model_name or os.getenv("BASE_MODEL_NAME", "microsoft/phi-3-mini-4k-instruct")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", str(max_new_tokens)))
        self._pipe = None
        # Lazy init flag (default true) to prevent huge downloads on startup
        self._lazy = os.getenv("LAZY_MODEL_INIT", "true").lower() in {"1", "true", "yes", "on"}
        if not self.stub_mode and not self._lazy:
            self._init_real_model()

    def _init_real_model(self):
        if not self.adapter_path:
            raise ValueError("MODEL_ADAPTER_PATH missing while stub_mode=False")
        if AutoModelForCausalLM is None:
            raise ImportError("Transformers/PEFT not installed. Install inference deps first.")
        # Pre-read adapter config to align base model if needed
        try:
            adapter_cfg_path = os.path.join(self.adapter_path, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    _cfg = json.load(f)
                adapter_base = _cfg.get("base_model_name_or_path") or _cfg.get("base_model")
                force_keep = os.getenv("FORCE_BASE_MODEL_NAME", "").lower() in {"1","true","yes","on"}
                if adapter_base and adapter_base != self.base_model_name and not force_keep:
                    print(f"[info] Overriding BASE_MODEL_NAME '{self.base_model_name}' with adapter base '{adapter_base}'")
                    self.base_model_name = adapter_base
        except Exception:
            pass
        # Device selection: keep simple CPU vs CUDA. We intentionally avoid bitsandbytes on native Windows.
        device = 0 if torch and torch.cuda.is_available() else -1
        # Dtype override (MODEL_DTYPE=fp16|bf16|fp32). For CPU default fp32; for CUDA prefer bf16/16 if available.
        raw_dtype = os.getenv("MODEL_DTYPE", "auto").lower()
        torch_dtype = None
        if torch is not None:
            if raw_dtype in {"fp16", "half", "float16"}:
                torch_dtype = torch.float16
            elif raw_dtype in {"bf16", "bfloat16"}:
                torch_dtype = getattr(torch, "bfloat16", torch.float32)
            elif raw_dtype in {"fp32", "float32"}:
                torch_dtype = torch.float32
            else:  # auto heuristic
                if torch.cuda.is_available():
                    torch_dtype = getattr(torch, "bfloat16", torch.float16)
                else:
                    torch_dtype = torch.float32
        # Load base model in 8-bit or 4-bit if desired (simplified to default float16 if CUDA available)
        # Hugging Face auth token support for gated models
        # Try multiple possible env var names for HF auth token
        hf_token = (
            os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or None
        )
        trust_remote = os.getenv("MODEL_TRUST_REMOTE_CODE", "true").lower() in {"1","true","yes","on"}
        # Keep device_map simple: use "auto" only when CUDA else None (CPU load). Attach dtype if resolved.
        load_kwargs = {"trust_remote_code": trust_remote}
        if torch_dtype is not None:  # type: ignore[assignment]
            load_kwargs["torch_dtype"] = torch_dtype  # type: ignore[assignment]
        if device == -1:
            load_kwargs["device_map"] = None  # type: ignore[assignment]
        else:
            load_kwargs["device_map"] = "auto"  # type: ignore[assignment]
        if hf_token:
            # Prefer new 'token' argument only. We'll fallback to legacy 'use_auth_token' if needed.
            load_kwargs["token"] = hf_token  # type: ignore
        # If no token provided we simply rely on public model access / local cache. We do NOT force use_auth_token.

        # Friendly pre-check for 4bit model on unsupported platform (bitsandbytes missing)
        if "bnb-4bit" in self.base_model_name.lower():
            try:
                import bitsandbytes  # type: ignore  # noqa: F401
            except Exception:
                raise RuntimeError(
                    "Selected base model requires bitsandbytes (4-bit). bitsandbytes isn't available (especially on native Windows). "
                    "Options: 1) Run on Linux GPU/WSL2 and pip install bitsandbytes. 2) Use the full-precision base model (e.g. mistralai/Mistral-7B-Instruct-v0.3) with FORCE_BASE_MODEL_NAME=true (memory ~14GB). "
                    "3) Keep MODEL_STUB_MODE=true locally and run real inference in Colab."
                )
        try:
            model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **load_kwargs)  # type: ignore[union-attr]
        except TypeError as e:
            # Older transformers might not accept 'token'
            if hf_token and 'token' in str(e).lower():
                fallback_kwargs = dict(load_kwargs)
                fallback_kwargs.pop('token', None)
                fallback_kwargs['use_auth_token'] = hf_token  # type: ignore
                model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **fallback_kwargs)  # type: ignore[union-attr]
            else:
                raise
        except OSError as e:
            msg = str(e)
            if "gated" in msg.lower() or "403" in msg:
                raise OSError(
                    "Gated model access denied. Steps: 1) Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and click 'Access repository' / accept license while logged in. "
                    "2) Obtain a read token: Hugging Face profile -> Settings -> Access Tokens (scopes: read). "
                    "3) Add to .env as HUGGING_FACE_HUB_TOKEN=hf_... (or HF_TOKEN). 4) Restart server. "
                    f"Checked env vars: token_present={bool(hf_token)}. Original error: {msg}"
                ) from e
            raise
    # (base already aligned earlier)
        try:
            model = PeftModel.from_pretrained(model, self.adapter_path)  # type: ignore[union-attr]
        except AttributeError as e:
            # Common when architecture path changed (embed_tokens) or version mismatch
            raise RuntimeError(
                "Adapter merge failed (embed_tokens mismatch). Possible causes: 1) Base model name/version different from training. "
                "2) Transformers/peft version mismatch. 3) Missing trust_remote_code=True for custom model class. "
                f"Base={self.base_model_name} Adapter={self.adapter_path} Error={e}"
            ) from e
        tokenizer_load_kwargs = {}
        if hf_token:
            tokenizer_load_kwargs["token"] = hf_token  # type: ignore
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.adapter_path, **tokenizer_load_kwargs)  # type: ignore[union-attr]
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)  # type: ignore[union-attr]
        except Exception:
            # Fallback: use base model tokenizer if adapter does not include tokenizer files
            base_tok_kwargs = dict(tokenizer_load_kwargs)
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, **base_tok_kwargs)  # type: ignore[union-attr]
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)  # type: ignore[union-attr]
        # Use transformers.pipeline for simplicity
        if 'hf_pipeline' not in globals():  # safety guard
            raise ImportError("transformers.pipeline missing")
        generation_kwargs = {}
        if trust_remote:
            generation_kwargs["trust_remote_code"] = True
        self._pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, **generation_kwargs)  # type: ignore

    def generate_kql(self, natural_language_prompt: str, context: Dict[str, Any] | None = None) -> str:
        if self.stub_mode:
            # naive heuristic examples
            lower = natural_language_prompt.lower()
            kql_query = ""
            if "failed" in lower and "login" in lower:
                kql_query = (
                    "SigninLogs\n"
                    "| where ResultType != 0\n"
                    "| summarize FailureCount=count(), StartTime=min(TimeGenerated), EndTime=max(TimeGenerated) by IPAddress, UserPrincipalName\n"
                    "| top 50 by FailureCount desc"
                )
            elif "list" in lower and "ips" in lower:
                kql_query = (
                    "SigninLogs\n"
                    "| summarize Count=count() by IPAddress\n"
                    "| top 100 by Count desc"
                )
            else:
                # default
                kql_query = (
                    "SigninLogs\n"
                    "| take 50"
                )
            print("\n============ CODE STARTS HERE ============")
            print(kql_query)
            print("============= CODE ENDS HERE =============\n")
            return kql_query
        else:
            if not self._pipe:
                # Lazy load when first needed
                self._init_real_model()
            prompt = self._build_generation_prompt(natural_language_prompt, context)
            result = self._pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)  # type: ignore[operator]
            out = result[0].get("generated_text", "") if isinstance(result, list) and result else ""
            # Post-process: extract KQL after a delimiter if using a template; here naive cleanup
            kql = self._extract_kql(out)
            print("\n============ CODE STARTS HERE ============")
            print(kql)
            print("============= CODE ENDS HERE =============\n")
            return kql

    def summarize(self, logs: List[Dict[str, Any]], original_prompt: str) -> Dict[str, Any]:
        if not logs:
            return {
                "paragraph": "No logs returned for the generated KQL; nothing to summarize.",
                "bullet_points": ["No data"],
            }
        # Simple heuristic summary
        max_preview = 5
        preview = logs[:max_preview]
        keys = set()
        for row in preview:
            for k in row.keys():
                keys.add(k)
        keys = list(keys)
        paragraph = (
            f"Summary derived from {len(logs)} log rows for request: '{original_prompt}'. "
            f"Showing {min(len(logs), max_preview)} sample rows to infer structure. "
            f"Detected fields include: {', '.join(keys[:12])}."
        )
        bullet_points = [
            f"Total Rows: {len(logs)}",
            f"Sample Fields: {', '.join(keys[:8])}",
            f"Original Intent: {original_prompt[:120]}" + ("..." if len(original_prompt) > 120 else ""),
            f"Sample Row 1: {json.dumps(preview[0], default=str) if preview else 'N/A'}",
        ]
        return {"paragraph": paragraph, "bullet_points": bullet_points}

    # --- Helper utilities ---
    def _build_generation_prompt(self, nl_prompt: str, context: Optional[Dict[str, Any]]):
        # --- System prompt and few-shot logic ---
        system_prefix = (
            "You are an expert KQL assistant for Microsoft Sentinel. Your task is to convert a user's natural language request into a single, valid KQL query.\n\n"
            "Rules:\n"
            "1.  Output ONLY the raw KQL query. Do not include any commentary, explanations, or markdown backticks.\n"
            "2.  Always include a bounded time filter (e.g., `... | where TimeGenerated > ago(24h)`).\n"
            "3.  If the user provides a specific entity (like a username, IP, or filename), use it directly in the query.\n"
            "4.  Use the most relevant and common fields for the specified log table and task.\n\n"
            "---\nExample:\n\nRequest: Show me failed logins for alice.admin@example.com in the last day.\n\n"
            "KQL:\nSigninLogs | where TimeGenerated > ago(1d) | where UserPrincipalName =~ 'alice.admin@example.com' | where ResultType != 0 | project TimeGenerated, IPAddress, Location, AppDisplayName, ResultType, ResultDescription\n---\n"
        )

        # Optional few-shot examples (enable via context)
        ctx = context or {}
        fewshot_enabled = ctx.get("KQL_FEWSHOT") in {1, "1", True, "true", "True"}
        fewshots = []
        if fewshot_enabled:
            fewshots = [
                (
                    "failed sign-ins from user 'example.user@contoso.com' in the last hour",
                    "SigninLogs\n| where TimeGenerated >= ago(1h)\n| where UserPrincipalName =~ 'example.user@contoso.com'\n| where ResultType != 0\n| project TimeGenerated, UserPrincipalName, AppDisplayName, IPAddress, ResultType"
                ),
                (
                    "successful sign-ins from user 'demo.account@example.com' from last 7 days",
                    "SigninLogs\n| where TimeGenerated >= ago(7d)\n| where UserPrincipalName =~ 'demo.account@example.com'\n| where ResultType == 0  // Successful sign-ins\n| project TimeGenerated, IPAddress, Location, AppDisplayName, ResultType, ResultDescription"
                ),
            ]
        fewshot_block = ""
        if fewshots:
            parts = []
            for q, kql in fewshots:
                parts.append(f"Request: {q}\nKQL:\n{kql}\n---\n")
            fewshot_block = "".join(parts)
        full_prompt = f"{system_prefix}{fewshot_block}Request: {nl_prompt}\nKQL:\n"
        return full_prompt

    def _extract_kql(self, raw: str) -> str:
        """Extract the KQL query corresponding to the user's request."""
        # Split by sections and clean them up
        sections = []
        current_section = {"request": "", "kql": []}
        
        for line in raw.splitlines():
            line = line.strip()
            if not line or line == "---":
                if current_section["request"] or current_section["kql"]:
                    sections.append(current_section.copy())
                    current_section = {"request": "", "kql": []}
            elif line.startswith("Request: "):
                if current_section["request"] or current_section["kql"]:
                    sections.append(current_section.copy())
                    current_section = {"request": "", "kql": []}
                current_section["request"] = line[len("Request: "):].strip()
            elif line.startswith("KQL:"):
                continue  # Skip the KQL: marker
            elif current_section["request"] and not line.startswith(("explanation:", "note:")):
                current_section["kql"].append(line)
        
        # Add the last section if not empty
        if current_section["request"] or current_section["kql"]:
            sections.append(current_section)
            
        # Find the last section that's not a few-shot example
        # (few-shot examples use @example.com or @contoso.com domains)
        for section in reversed(sections):
            kql = "\n".join(section["kql"]).strip()
            if "@example.com" not in section["request"] and "@contoso.com" not in section["request"]:
                return kql
                
        # If no matching section found, return empty string
        return ""

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "stub_mode": self.stub_mode,
            "base_model_name": self.base_model_name,
            "adapter_path": self.adapter_path,
            "pipeline_initialized": self._pipe is not None,
        }
        # Read adapter config if present
        if self.adapter_path:
            cfg_path = os.path.join(self.adapter_path, "adapter_config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    info["adapter_base_model_in_config"] = cfg.get("base_model_name_or_path") or cfg.get("base_model")
                    info["lora_r"] = cfg.get("r")
                    info["lora_alpha"] = cfg.get("lora_alpha")
                    info["lora_dropout"] = cfg.get("lora_dropout")
                except Exception as e:
                    info["adapter_config_error"] = str(e)
        return info
