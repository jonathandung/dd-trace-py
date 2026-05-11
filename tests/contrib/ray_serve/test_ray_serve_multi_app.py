import os
from pathlib import Path
import signal
import subprocess
import time

import pytest
import requests


RAY_MULTI_APP_SNAPSHOT_IGNORES = [
    "meta.tracestate",
    "meta.ray.serve.handle_id",
    "meta.ray.serve.request_id",
    "meta.ray.serve.replica_id",
    "meta.ray.serve.deployment_id",
    "meta.ray.serve.handle_source",
    "meta.error.message",
    "meta.error.stack",
]
MULTI_APP_SERVE_DIR = Path(__file__).parent / "multi_app"


def _wait_for_multi_app_deployments(env):
    deadline = time.time() + 120
    last_status = ""

    while time.time() < deadline:
        result = subprocess.run(
            ["serve", "status"],
            cwd=MULTI_APP_SERVE_DIR,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        last_status = result.stdout + result.stderr
        if (
            result.returncode == 0
            and "hello_app" in last_status
            and "goodbye_app" in last_status
            and last_status.count("RUNNING") >= 2
        ):
            return
        time.sleep(0.5)

    raise AssertionError("Ray Serve multi-app deployments were not ready.\n%s" % last_status)


@pytest.fixture
def multi_app_serve_url(snapshot):
    env = os.environ.copy()
    env.update(
        {
            "DD_ENV": "test",
            "DD_PATCH_MODULES": "ray:true,aiohttp:false,grpc:false,requests:false",
        }
    )

    subprocess.run(["ray", "stop", "--force"], env=env, check=False, capture_output=True)
    server_process = subprocess.Popen(
        ["ddtrace-run", "serve", "run", "serve_config.yaml"],
        cwd=MULTI_APP_SERVE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        close_fds=True,
        start_new_session=True,
    )

    try:
        try:
            _wait_for_multi_app_deployments(env)
        except Exception:
            stdout, stderr = server_process.communicate(timeout=1) if server_process.poll() is not None else ("", "")
            raise AssertionError(
                "Ray Serve multi-app server failed.\n"
                "=== Captured STDOUT ===\n%s\n=== End of captured STDOUT ===\n"
                "=== Captured STDERR ===\n%s\n=== End of captured STDERR ===" % (stdout, stderr)
            )
        snapshot.clear()
        yield "http://127.0.0.1:8000"
        time.sleep(5)
    finally:
        if server_process.poll() is None:
            os.killpg(server_process.pid, signal.SIGTERM)
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server_process.pid, signal.SIGKILL)
                server_process.wait()
        subprocess.run(["ray", "stop", "--force"], env=env, check=False, capture_output=True)


@pytest.mark.snapshot(ignores=RAY_MULTI_APP_SNAPSHOT_IGNORES)
def test_multi_app_deployment_routes(multi_app_serve_url):
    hello_resp = requests.get(f"{multi_app_serve_url}/hello", timeout=2)
    assert hello_resp.status_code == 200
    assert hello_resp.json() == {"message": "Hello from app1"}

    goodbye_resp = requests.get(f"{multi_app_serve_url}/goodbye", timeout=2)
    assert goodbye_resp.status_code == 200
    assert goodbye_resp.json() == {"message": "Goodbye from app2"}
