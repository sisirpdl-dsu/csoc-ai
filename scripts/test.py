
import pandas as pd
import sys, os
import requests
from datetime import datetime
from azure.identity import AzureCliCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.core.exceptions import HttpResponseError

# --- Prompt for user input ---
if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
else:
    prompt = input("Enter your natural language request for KQL: ").strip()
print(f"[DEBUG] Prompt: {prompt}")

# --- Use backend for KQL generation, with few-shots enabled ---
BACKEND_URL = "http://localhost:8000/generate-kql"
payload = {"prompt": prompt, "context": {"KQL_FEWSHOT": 1}}
try:
    resp = requests.post(BACKEND_URL, json=payload)
    resp.raise_for_status()
    generated_kql_query = resp.json().get("kql")
    print("\n[BACKEND] Generated KQL (raw):\n" + str(generated_kql_query))
    # Robust KQL extraction: keep only the first valid KQL statement (starting with a known table)
    kql_lines = []
    found = False
    for line in str(generated_kql_query).splitlines():
        l = line.strip()
        # Skip empty lines and delimiters
        if not l or l.startswith("---") or l.lower().startswith(("example output:", "kql:")):
            continue
        # Only keep lines that look like a KQL query (start with a table name, e.g., SigninLogs)
        if l.startswith("SigninLogs") or l.startswith("AuditLogs") or l.startswith("SecurityEvent"):
            kql_lines.append(l)
            found = True
        elif found and l and not l.startswith("{") and not l.lower().startswith("example"):
            # If already found a table, keep subsequent lines until a blank or delimiter
            kql_lines.append(l)
        elif found:
            break
    cleaned_kql = "\n".join(kql_lines).strip()
    if not cleaned_kql:
        print("[ERROR] No valid KQL found in backend response.")
        sys.exit(2)
    print("\n[BACKEND] Cleaned KQL to send to Azure:\n" + cleaned_kql)
    generated_kql_query = cleaned_kql
except Exception as e:
    print(f"[ERROR] Failed to get KQL from backend: {e}")
    sys.exit(1)

# --- Azure Query ---
WORKSPACE_ID = os.getenv("LOG_WORKSPACE_ID")
if not WORKSPACE_ID:
    print("[ERROR] LOG_WORKSPACE_ID environment variable not set")
    print("Please set LOG_WORKSPACE_ID in your .env file or environment variables")
    sys.exit(1)

credential = AzureCliCredential()
client = LogsQueryClient(credential)
try:
    # Extract timespan from the query
    import re
    timespan_days = 1  # default
    time_match = re.search(r'ago\((\d+)([dhm])\)', generated_kql_query)
    if time_match:
        amount = int(time_match.group(1))
        unit = time_match.group(2)
        if unit == 'd':
            timespan_days = amount
        elif unit == 'h':
            timespan_days = amount / 24
        elif unit == 'm':
            timespan_days = amount / (24 * 60)
            
    print(f"[DEBUG] Using timespan of {timespan_days} days based on query")
    print("[DEBUG] Running KQL query against Azure...")
    response = client.query_workspace(
        workspace_id=WORKSPACE_ID,
        query=generated_kql_query,
        timespan=pd.Timedelta(days=timespan_days)
    )
    print("[DEBUG] Query executed, processing response...")
    print("\n[DEBUG] Raw Azure Response:")
    print("Response Status:", response.status)
    print("Response Attributes:", dir(response))
    
    # Handle both partial and complete results
    tables = []
    if response.status == LogsQueryStatus.PARTIAL:
        if hasattr(response, 'partial_tables'):
            tables = response.partial_tables
    elif hasattr(response, 'tables'):
        tables = response.tables
    
    if tables and len(tables) > 0:
        table = tables[0]
        print("\n[DEBUG] Table Info:")
        print("Table Attributes:", dir(table))
        print("Number of Columns:", len(table.columns))
        print("Number of Rows:", len(table.rows))
        print("\nColumns:", [c.name if hasattr(c, 'name') else c for c in table.columns])
        print("\nSample Raw Row:", table.rows[0] if table.rows else "No rows")
        print("\nProcessing rows into DataFrame...")
        
        # Convert rows to proper format
        try:
            processed_rows = []
            columns = []
            if table.columns:
                # Get column names safely
                columns = []
                for col in table.columns:
                    if hasattr(col, 'name'):
                        columns.append(col.name)
                    else:
                        columns.append(str(col))
                print("\nColumn Names:", columns)
                
                print("\nProcessing rows:")
                for row_idx, row in enumerate(table.rows):
                    processed_row = []
                    for val in row:
                        if isinstance(val, datetime):
                            processed_row.append(val.isoformat())
                        else:
                            processed_row.append(val)
                    processed_rows.append(processed_row)
                    print(f"Row {row_idx}:", dict(zip(columns, processed_row)))
            
            print("\nCreating DataFrame...")
            df = pd.DataFrame(processed_rows, columns=columns)
            print("\nQuery successful! Results:")
            print(df)
            print("\nDataFrame Info:")
            print(df.info())
            
            # Print cleaned and colored table
            print("\n\033[1m=== Cleaned Results Table ===\033[0m")
            
            # Get max lengths for each column for formatting
            col_widths = {}
            for col in df.columns:
                max_val_len = df[col].astype(str).map(len).max()
                col_widths[col] = max(max_val_len, len(col))
            
            # Print headers in blue
            header_row = ""
            for col in df.columns:
                header_row += f"\033[34m{col.ljust(col_widths[col] + 2)}\033[0m"
            print(header_row)
            
            # Print separator
            print("-" * sum(width + 2 for width in col_widths.values()))
            
            # Print rows alternating between yellow and green
            for idx, row in df.iterrows():
                row_color = "\033[33m" if idx % 2 == 0 else "\033[32m"  # Yellow or Green
                row_str = ""
                for col in df.columns:
                    val = str(row[col])
                    # Clean up datetime format
                    if col == "TimeGenerated":
                        try:
                            dt = pd.to_datetime(val)
                            val = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    row_str += f"{row_color}{val.ljust(col_widths[col] + 2)}"
                print(row_str + "\033[0m")  # Reset color at end of row
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process results: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("[DEBUG] No tables returned in response.")
        print(response)
except HttpResponseError as e:
    print(f"Query failed. Error: {e.message}")
except Exception as ex:
    print(f"[DEBUG] Unhandled exception: {type(ex).__name__}: {ex}")