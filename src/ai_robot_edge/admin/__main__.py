from __future__ import annotations

import argparse

import uvicorn

from ..config import load_config
from .app import create_edge_admin_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI robot edge admin UI")
    parser.add_argument(
        "--config",
        default="config/edge.yaml",
        help="Path to the edge YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    uvicorn.run(
        create_edge_admin_app(args.config),
        host=config.admin.host,
        port=config.admin.port,
    )


if __name__ == "__main__":
    main()
