from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace

from ..config import load_config
from .gimbal import PanTiltGimbal, SongJiaProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or control the USB pan-tilt gimbal")
    parser.add_argument("--config", default="config/edge.yaml")
    parser.add_argument(
        "command",
        choices=["probe", "center", "stop", "move"],
        help="probe never writes to the servo controller",
    )
    parser.add_argument("--pan", type=float)
    parser.add_argument("--tilt", type=float)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow movement commands. Without this flag commands are dry-run only.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "probe":
        print(
            json.dumps(
                {
                    "port": config.servo.port,
                    "baudrate": config.servo.baudrate,
                    "pan_id": config.servo.pan.servo_id,
                    "tilt_id": config.servo.tilt.servo_id,
                    "center_command": SongJiaProtocol.group_move_command(
                        [
                            (config.servo.pan.servo_id, config.servo.pan.neutral_angle),
                            (config.servo.tilt.servo_id, config.servo.tilt.neutral_angle),
                        ],
                        config.servo.default_move_time_ms,
                    ),
                },
                indent=2,
            )
        )
        return

    servo_config = config.servo if args.live else replace(config.servo, dry_run=True)
    gimbal = PanTiltGimbal(servo_config)
    try:
        if args.command == "center":
            position = await gimbal.center()
            print(position)
        elif args.command == "stop":
            await gimbal.stop()
        elif args.command == "move":
            if args.pan is None or args.tilt is None:
                raise SystemExit("move requires --pan and --tilt")
            position = await gimbal.move_to(args.pan, args.tilt)
            print(position)
    finally:
        await gimbal.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
