from __future__ import annotations

import argparse
import asyncio

from .app import EdgeApp
from .config import load_config
from .logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI robot edge service")
    parser.add_argument(
        "--config",
        default="config/edge.yaml",
        help="Path to the edge YAML configuration file",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.runtime.log_level)
    app = EdgeApp(config)
    await app.run()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
