from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .config import load_server_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI robot server console")
    parser.add_argument(
        "--config",
        default="config/server.yaml",
        help="Path to the server YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_server_config(args.config)
    uvicorn.run(create_app(config), host=config.http.host, port=config.http.port)


if __name__ == "__main__":
    main()
