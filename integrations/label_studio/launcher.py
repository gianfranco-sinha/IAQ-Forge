"""Label Studio launcher — starts LS with local file serving + CORS CSV server."""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger("integrations.label_studio.launcher")

# Default ports
LS_PORT = 8080
CSV_PORT = 9999

# Directory where time series CSVs are served from
TIMESERIES_DIR = Path("export_timeseries")


class CORSHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers for Label Studio to fetch CSVs."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs


def _start_csv_server(directory: Path, port: int = CSV_PORT) -> HTTPServer:
    """Start a CORS-enabled HTTP server for time series CSVs."""
    os.chdir(str(directory))
    server = HTTPServer(("127.0.0.1", port), CORSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def generate_timeseries_csvs(
    raw_df, clean_df, profile, standard, drop_reasons=None,
    include_dropped=False, output_dir=None, csv_port=CSV_PORT,
    gap_threshold_seconds=30.0,
):
    """Generate time series CSVs split at discontinuities, with tasks.json.

    Segments are split where time gaps exceed gap_threshold_seconds. Each
    contiguous segment becomes one CSV file and one Label Studio task.

    Returns:
        (output_dir, tasks) — path to output directory, list of task dicts.
    """
    import pandas as pd

    out_dir = Path(output_dir or TIMESERIES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean old CSVs
    for old in out_dir.glob("*.csv"):
        old.unlink()

    target = standard.target_column
    raw_features = list(profile.raw_features)
    data_columns = raw_features + ([target] if target in raw_df.columns else [])

    # Build a combined DataFrame with raw_ and clean_ prefixed columns
    ts_rows = []
    kept_indices = clean_df.index

    for idx in kept_indices:
        row = {"timestamp": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx)}
        for col in data_columns:
            row[f"raw_{col}"] = raw_df.loc[idx].get(col)
            row[f"clean_{col}"] = clean_df.loc[idx].get(col)
        ts_rows.append(row)

    if include_dropped:
        dropped_indices = raw_df.index.difference(kept_indices)
        for idx in dropped_indices:
            row = {"timestamp": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx)}
            for col in data_columns:
                row[f"raw_{col}"] = raw_df.loc[idx].get(col)
            ts_rows.append(row)

    ts_df = pd.DataFrame(ts_rows)
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
    ts_df = ts_df.sort_values("timestamp").reset_index(drop=True)

    # Split at discontinuities (time gaps > threshold)
    threshold = pd.Timedelta(seconds=gap_threshold_seconds)
    deltas = ts_df["timestamp"].diff()
    split_points = deltas[deltas > threshold].index.tolist()

    # Build segment boundaries: [0, split1, split2, ..., len]
    boundaries = [0] + split_points + [len(ts_df)]

    tasks = []
    for seg_idx in range(len(boundaries) - 1):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]
        segment = ts_df.iloc[start:end]

        if len(segment) == 0:
            continue

        seg_start = segment["timestamp"].iloc[0]
        seg_end = segment["timestamp"].iloc[-1]
        date_str = seg_start.strftime("%Y-%m-%d")
        time_str = seg_start.strftime("%H%M%S")
        fname = f"{date_str}_{time_str}_seg{seg_idx:03d}.csv"

        # Format timestamps for Label Studio
        segment = segment.copy()
        segment["timestamp"] = segment["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        segment.to_csv(out_dir / fname, index=False)

        duration_min = (seg_end - seg_start).total_seconds() / 60
        tasks.append({
            "data": {
                "timeseries": f"http://localhost:{csv_port}/{fname}",
                "date": date_str,
                "segment": seg_idx + 1,
                "num_readings": len(segment),
                "duration_minutes": round(duration_min, 1),
                "start": seg_start.strftime("%H:%M:%S"),
                "end": seg_end.strftime("%H:%M:%S"),
            }
        })

    # Write tasks.json
    with open(out_dir / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)

    return out_dir, tasks


def get_labeling_config() -> str:
    """Return the labeling config XML for time series annotation."""
    config_path = Path(__file__).parent / "labeling_config.xml"
    return config_path.read_text()


def setup_project(url, api_key, project_title="IAQ Cleanse Review", port=CSV_PORT):
    """Create or find a Label Studio project with the correct labeling config.

    Returns:
        project_id
    """
    import requests

    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    labeling_config = get_labeling_config()

    # Check existing projects
    resp = requests.get(f"{url}/api/projects", headers=headers, timeout=10)
    resp.raise_for_status()
    projects = resp.json().get("results", []) if isinstance(resp.json(), dict) else resp.json()

    for p in projects:
        if p.get("title") == project_title:
            pid = p["id"]
            # Update labeling config
            requests.patch(
                f"{url}/api/projects/{pid}",
                headers=headers,
                json={"label_config": labeling_config},
                timeout=10,
            )
            logger.info("Updated existing project %d: '%s'", pid, project_title)
            return pid

    # Create new project
    resp = requests.post(
        f"{url}/api/projects",
        headers=headers,
        json={"title": project_title, "label_config": labeling_config},
        timeout=10,
    )
    resp.raise_for_status()
    pid = resp.json()["id"]
    logger.info("Created new project %d: '%s'", pid, project_title)
    return pid


def import_tasks(url, api_key, project_id, tasks, batch_size=50):
    """Import tasks into a Label Studio project via API."""
    import math
    import requests

    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    total = len(tasks)
    batches = math.ceil(total / batch_size)

    for i in range(0, total, batch_size):
        batch = tasks[i : i + batch_size]
        resp = requests.post(
            f"{url}/api/projects/{project_id}/import",
            json=batch,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        batch_num = (i // batch_size) + 1
        print(f"  Imported batch {batch_num}/{batches} ({len(batch)} tasks)")

    return total


def clear_project(url, api_key, project_id):
    """Delete all tasks from a Label Studio project."""
    import requests

    headers = {"Authorization": f"Token {api_key}"}
    resp = requests.delete(
        f"{url}/api/projects/{project_id}/tasks",
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()


def launch(
    ls_port=LS_PORT,
    csv_port=CSV_PORT,
    timeseries_dir=None,
):
    """Start Label Studio and the CSV file server.

    Blocks until the user hits Ctrl+C.
    """
    ts_dir = Path(timeseries_dir or TIMESERIES_DIR)

    # Start CSV server if timeseries dir exists
    csv_server = None
    if ts_dir.exists() and list(ts_dir.glob("*.csv")):
        csv_server = _start_csv_server(ts_dir, csv_port)
        print(f"CSV server: http://localhost:{csv_port}/ ({len(list(ts_dir.glob('*.csv')))} files)")
    else:
        print(f"No time series CSVs found in {ts_dir}/ — run export-to-ls first")

    # Start Label Studio
    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(ts_dir.resolve())

    ls_bin = shutil.which("label-studio")
    if not ls_bin:
        print("Error: label-studio not found. Install with: brew install label-studio")
        sys.exit(1)

    print(f"Starting Label Studio on http://localhost:{ls_port}/ ...")
    print("Press Ctrl+C to stop.\n")

    proc = subprocess.Popen(
        [ls_bin, "start", "--port", str(ls_port), "--no-browser"],
        env=env,
    )

    def _shutdown(sig, frame):
        print("\nShutting down...")
        proc.terminate()
        if csv_server:
            csv_server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    proc.wait()
