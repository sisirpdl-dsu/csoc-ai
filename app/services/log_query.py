from typing import List, Dict, Any, Optional
from datetime import timedelta, datetime, timezone
import os
from azure.identity import (
    ClientSecretCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    DefaultAzureCredential,
)
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from ..core.config import get_settings

class SentinelLogService:
    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._disabled_reason = None
        self.auth_method = None  # record which credential succeeded
        self._init_client_with_fallbacks()

    def _init_client_with_fallbacks(self):
        if not self.settings.log_workspace_id:
            self._disabled_reason = "Missing LOG_WORKSPACE_ID; running in stub mode."
            return

        # Ordered strategies (name, callable returning credential or raising)
        strategies = []

        # 1. Explicit service principal if full trio present
        if self.settings.tenant_id and self.settings.client_id and self.settings.client_secret:
            def _sp():
                # Cast to str (mypy/pylance) since we guarded for None
                return ClientSecretCredential(
                    tenant_id=str(self.settings.tenant_id),
                    client_id=str(self.settings.client_id),
                    client_secret=str(self.settings.client_secret),
                )
            strategies.append(("client_secret", _sp))

        # 2. Azure CLI (dev) if env flag set
        if os.getenv("AZURE_CLI_AUTH") in {"1", "true", "True"}:
            strategies.append(("azure_cli", lambda: AzureCliCredential()))

        # 3. Managed Identity if flag set (or if running in MI environment and user forces)
        if os.getenv("MANAGED_IDENTITY") in {"1", "true", "True"}:
            strategies.append(("managed_identity", lambda: ManagedIdentityCredential()))

        # 4. Default credential (broad chain) if env flag set
        if os.getenv("DEFAULT_AZURE_CREDENTIAL") in {"1", "true", "True"}:
            strategies.append(("default_chain", lambda: DefaultAzureCredential(exclude_interactive_browser_credential=True)))

        # If no strategy queued, set reason
        if not strategies:
            self._disabled_reason = (
                "No credential strategy configured. Provide client secret vars, or set AZURE_CLI_AUTH=1, "
                "or MANAGED_IDENTITY=1, or DEFAULT_AZURE_CREDENTIAL=1."
            )
            return

        errors = []
        for name, factory in strategies:
            try:
                cred = factory()
                # Try constructing client (lightweight); actual token fetched on first query
                self._client = LogsQueryClient(cred)
                self.auth_method = name
                self._disabled_reason = None
                break
            except Exception as e:  # capture and continue
                errors.append(f"{name}: {e}")
                self._client = None
        if self._client is None:
            self._disabled_reason = "All credential strategies failed: " + " | ".join(errors)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def query(self, kql: str, timespan: Optional[timedelta] = None, max_rows: Optional[int] = None) -> Dict[str, Any]:
        # If client not initialized, return stub rows
        if not self.enabled or not self._client or not self.settings.log_workspace_id:
            stub = self._stub_rows(kql)
            return {
                "mode": "stub",
                "reason": self._disabled_reason,
                "auth_method": self.auth_method,
                "kql": kql,
                "rows": stub,
                "statistics": {"retrieved": len(stub)},
            }
        if timespan is None:
            timespan = timedelta(hours=24)
        max_rows = max_rows or self.settings.max_rows
        try:
            print("\n[DEBUG] Running KQL query:", kql)
            response = self._client.query_workspace(self.settings.log_workspace_id, kql, timespan=timespan)  # type: ignore[arg-type]
            
            # Print raw response for debugging
            print("\n[DEBUG] Raw Azure Response:")
            print("Response Status:", response.status)
            print("Response Type:", type(response))
            print("Response Dir:", dir(response))
            
            # Print raw tables data
            if response.status == LogsQueryStatus.PARTIAL:
                tables = getattr(response, "partial_tables", None) or []  # type: ignore[assignment]
                print("\n[DEBUG] Using partial_tables")
            else:
                tables = getattr(response, "tables", None) or []  # type: ignore[assignment]
                print("\n[DEBUG] Using tables")
            
            print("\n[DEBUG] Number of tables:", len(tables))
            
            table = tables[0] if tables else None
            if table:
                print("\n[DEBUG] First table dir:", dir(table))
                print("[DEBUG] First table type:", type(table))
                print("[DEBUG] Raw table data:", table)
            
            rows: List[Dict[str, Any]] = []
            if table:
                try:
                    print("\n[DEBUG] Raw table columns:", table.columns)  # type: ignore[attr-defined]
                    print("[DEBUG] Raw table rows:", table.rows)  # type: ignore[attr-defined]
                    
                    columns = [c.name for c in table.columns]  # type: ignore[attr-defined]
                    print(f"\n[DEBUG] Column names: {columns}")
                    print(f"[DEBUG] Azure returned {len(table.rows)} rows")  # type: ignore[attr-defined]
                    
                    print("\n[DEBUG] Raw row data:")
                    for idx, r in enumerate(table.rows[:max_rows]):  # type: ignore[attr-defined]
                        print(f"Row {idx}:", r)
                        row_dict = {}
                        for col, val in zip(columns, r):
                            print(f"  {col} = {val} (type: {type(val)})")
                            # Convert datetime objects to ISO format strings
                            if isinstance(val, datetime):
                                row_dict[col] = val.isoformat()
                            else:
                                row_dict[col] = val
                        rows.append(row_dict)
                    print(f"\n[DEBUG] Processed {len(rows)} rows into dictionary format:")
                except Exception as e:
                    print(f"[ERROR] Failed to process Azure response: {str(e)}")
                    import traceback
                    traceback.print_exc()
            return {
                "mode": "live",
                "auth_method": self.auth_method,
                "kql": kql,
                "rows": rows,
                "statistics": {"retrieved": len(rows), "status": response.status.value},
            }
        except Exception as e:
            return {
                "mode": "error-fallback",
                "auth_method": self.auth_method,
                "kql": kql,
                "error": str(e),
                "rows": [],
                "statistics": {"retrieved": 0},
            }

    def _stub_rows(self, kql: str) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        if "SigninLogs" in kql:
            return [
                {
                    "TimeGenerated": now.isoformat(),
                    "IPAddress": "203.0.113.75",
                    "UserPrincipalName": "j.doe@example.com",
                    "ResultType": 50126,
                    "FailureCount": 7,
                },
                {
                    "TimeGenerated": now.isoformat(),
                    "IPAddress": "203.0.113.75",
                    "UserPrincipalName": "j.doe@example.com",
                    "ResultType": 50126,
                    "FailureCount": 7,
                },
            ]
        return [
            {"TimeGenerated": now.isoformat(), "Message": "Stub log row", "KQL": kql[:60]},
        ]
