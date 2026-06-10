from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from ..config import ServoAxisConfig, ServoConfig
from ..events import ActionIntent, ActionName
from .base import ServoController

LOGGER = logging.getLogger(__name__)


class CommandTransport(Protocol):
    def write(self, data: bytes) -> int:
        """Write one command to the controller."""

    def read_feedback(self) -> list[str]:
        """Read currently available feedback lines."""

    def close(self) -> None:
        """Close the transport."""


class PySerialTransport:
    def __init__(
        self,
        port: str,
        baudrate: int,
        timeout_seconds: float,
        startup_delay_seconds: float,
    ) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for the SongJia USB servo controller"
            ) from exc

        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout_seconds,
            bytesize=8,
            parity="N",
            stopbits=1,
        )
        time.sleep(startup_delay_seconds)

    def write(self, data: bytes) -> int:
        return int(self._serial.write(data))

    def read_feedback(self) -> list[str]:
        feedback: list[str] = []
        while self._serial.in_waiting > 0:
            message = self._serial.readline().decode("utf-8", errors="ignore").strip()
            if message:
                feedback.append(message)
        return feedback

    def close(self) -> None:
        self._serial.close()


class SongJiaProtocol:
    MIN_PWM = 500
    MAX_PWM = 2500
    MAX_ANGLE = 270.0

    @classmethod
    def angle_to_pwm(cls, angle: float) -> int:
        clamped = max(0.0, min(float(angle), cls.MAX_ANGLE))
        return round(cls.MIN_PWM + clamped / cls.MAX_ANGLE * (cls.MAX_PWM - cls.MIN_PWM))

    @classmethod
    def move_command(cls, servo_id: int, angle: float, move_time_ms: int) -> str:
        return (
            f"#{servo_id:03d}P{cls.angle_to_pwm(angle):04d}"
            f"T{_clamp_move_time(move_time_ms):04d}!"
        )

    @classmethod
    def group_move_command(
        cls, moves: list[tuple[int, float]], move_time_ms: int
    ) -> str:
        return (
            "{"
            + "".join(
                cls.move_command(servo_id, angle, move_time_ms)
                for servo_id, angle in moves
            )
            + "}"
        )

    @staticmethod
    def stop_all_command() -> str:
        return "$DST!"

    @staticmethod
    def stop_servo_command(servo_id: int) -> str:
        return f"$DST:{servo_id}!"


@dataclass(frozen=True)
class GimbalPosition:
    pan: float
    tilt: float


class PanTiltGimbal(ServoController):
    """Safe, stateful controller for a SongJia two-axis USB gimbal."""

    def __init__(
        self,
        config: ServoConfig,
        transport: CommandTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._position = GimbalPosition(
            pan=config.pan.neutral_angle,
            tilt=config.tilt.neutral_angle,
        )
        self._lock = asyncio.Lock()

    @property
    def position(self) -> GimbalPosition:
        return self._position

    async def connect(self) -> None:
        if self._transport is not None or self.config.dry_run:
            return
        self._transport = await asyncio.to_thread(
            PySerialTransport,
            self.config.port,
            self.config.baudrate,
            self.config.timeout_seconds,
            self.config.startup_delay_seconds,
        )
        LOGGER.info("connected SongJia gimbal on %s", self.config.port)

    async def close(self) -> None:
        if self._transport is not None:
            await asyncio.to_thread(self._transport.close)
            self._transport = None

    async def move_to(
        self,
        pan: float,
        tilt: float,
        move_time_ms: int | None = None,
        *,
        force_all_axes: bool = False,
    ) -> GimbalPosition:
        target = GimbalPosition(
            pan=_limit_axis(pan, self.config.pan),
            tilt=_limit_axis(tilt, self.config.tilt),
        )
        moves: list[tuple[int, float]] = []
        if force_all_axes or not _angles_close(target.pan, self._position.pan):
            moves.append(
                (
                    self.config.pan.servo_id,
                    _physical_angle(target.pan, self.config.pan),
                )
            )
        if force_all_axes or not _angles_close(target.tilt, self._position.tilt):
            moves.append(
                (
                    self.config.tilt.servo_id,
                    _physical_angle(target.tilt, self.config.tilt),
                )
            )
        if not moves:
            return target
        if len(moves) == 1:
            servo_id, angle = moves[0]
            command = SongJiaProtocol.move_command(
                servo_id, angle, move_time_ms or self.config.default_move_time_ms
            )
        else:
            command = SongJiaProtocol.group_move_command(
                moves, move_time_ms or self.config.default_move_time_ms
            )
        async with self._lock:
            await self._send(command)
            self._position = target
        return target

    async def move_by(
        self,
        pan_delta: float,
        tilt_delta: float,
        move_time_ms: int | None = None,
    ) -> GimbalPosition:
        return await self.move_to(
            self._position.pan + pan_delta,
            self._position.tilt + tilt_delta,
            move_time_ms,
        )

    async def center(self, move_time_ms: int | None = None) -> GimbalPosition:
        return await self.move_to(
            self.config.pan.neutral_angle,
            self.config.tilt.neutral_angle,
            move_time_ms,
            force_all_axes=True,
        )

    async def stop(self) -> None:
        async with self._lock:
            await self._send(SongJiaProtocol.stop_all_command())

    async def execute(self, intent: ActionIntent) -> None:
        if intent.name == ActionName.LOOK_AT_USER:
            await self.move_to(
                float(intent.parameters.get("pan", self._position.pan)),
                float(intent.parameters.get("tilt", self._position.tilt)),
                int(intent.parameters.get("move_time_ms", self.config.default_move_time_ms)),
            )
        elif intent.name == ActionName.IDLE:
            await self.center()
        elif intent.name == ActionName.NOD:
            amplitude = min(20.0, max(2.0, float(intent.parameters.get("amplitude", 8))))
            await self.move_by(0, amplitude, 250)
            await self.move_by(0, -amplitude, 250)
        elif intent.name == ActionName.WELCOME_MOTION:
            await self.move_by(8, 0, 300)
            await self.move_by(-16, 0, 500)
            await self.move_by(8, 0, 300)

    async def _send(self, command: str) -> list[str]:
        if self.config.dry_run:
            LOGGER.info("gimbal dry-run command: %s", command)
            return []
        await self.connect()
        assert self._transport is not None
        LOGGER.debug("sending gimbal command: %s", command)
        await asyncio.to_thread(self._transport.write, command.encode("ascii"))
        return await asyncio.to_thread(self._transport.read_feedback)


def _physical_angle(logical_angle: float, axis: ServoAxisConfig) -> float:
    if not axis.inverted:
        return logical_angle
    return axis.max_angle - (logical_angle - axis.min_angle)


def _limit_axis(angle: float, axis: ServoAxisConfig) -> float:
    return max(axis.min_angle, min(float(angle), axis.max_angle))


def _clamp_move_time(move_time_ms: int) -> int:
    return max(0, min(int(move_time_ms), 9999))


def _angles_close(left: float, right: float, tolerance: float = 1e-3) -> bool:
    return abs(left - right) <= tolerance
