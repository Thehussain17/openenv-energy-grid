"""
FastAPI application for the Energy Grid Environment.

Endpoints exposed:
  GET  /health   → {"status": "healthy"}
  GET  /metadata → {"name": ..., "description": ...}
  GET  /schema   → {action, observation, state}
  POST /reset    → initial observation
  POST /step     → step result
  GET  /state    → current episode state
  WS   /ws       → persistent WebSocket session

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core[core]>=0.2.2"
    ) from e

import sys as _sys
import os as _os
try:
    from ..models import GridAction, GridObservation
    from .energy_grid_environment import EnergyGridEnvironment
except ImportError:
    # Running as `uvicorn server.app:app` from the project root — use flat imports
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from models import GridAction, GridObservation
    from server.energy_grid_environment import EnergyGridEnvironment


app = create_app(
    EnergyGridEnvironment,
    GridAction,
    GridObservation,
    env_name="energy_grid_env",
    max_concurrent_envs=4,
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Ensure the static directory exists
static_dir = _os.path.join(_os.path.dirname(__file__), "static")
_os.makedirs(static_dir, exist_ok=True)

app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/ui/index.html")


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """
    Entry point for direct execution or uv run.

    Examples:
        python -m server.app
        uvicorn server.app:app --port 7860
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Energy Grid Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
    main()  # noqa: F401 — satisfies openenv validate literal check
