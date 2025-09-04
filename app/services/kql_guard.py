from __future__ import annotations
import re
from typing import Tuple

TIME_PATTERN = re.compile(r"timegenerated", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"\|\s*where\b.*timegenerated.*(ago|datetime|between)", re.IGNORECASE)
FORBIDDEN = [r";", r"<script", r"\b(drop|delete|truncate)\b"]

def validate_kql(kql: str) -> Tuple[bool, list[str]]:
    """Return (ok, issues). Pure string heuristics.
    Rules:
      - Must reference TimeGenerated.
      - Must contain a time filter (ago/datetime/between) on TimeGenerated.
      - Must not contain obvious multi-statement separators or destructive verbs.
    """
    issues: list[str] = []
    if not TIME_PATTERN.search(kql):
        issues.append("Missing TimeGenerated reference")
    if not RANGE_PATTERN.search(kql):
        issues.append("Missing explicit TimeGenerated time filter (ago/datetime/between)")
    lower = kql.lower()
    for f in FORBIDDEN:
        if re.search(f, lower):
            issues.append(f"Forbidden pattern detected: {f}")
    return (len(issues) == 0, issues)

def enforce_or_fix(kql: str, hours: int = 24) -> str:
    ok, issues = validate_kql(kql)
    if ok:
        return kql
    # Minimal auto-fix: append a where clause if missing TimeGenerated filter.
    if any("Missing explicit TimeGenerated time filter" in i for i in issues):
        # If there's already a where but no filter, append another pipe.
        addition = f"| where TimeGenerated > ago({hours}h)"
        if "TimeGenerated" not in kql:
            kql = kql.rstrip() + "\n| where TimeGenerated > ago({}h)".format(hours)
        else:
            kql = kql.rstrip() + f"\n{addition}"
    return kql