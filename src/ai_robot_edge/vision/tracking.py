from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ..config import TrackingConfig
from ..devices.gimbal import GimbalPosition, PanTiltGimbal


@dataclass(frozen=True)
class TrackingTarget:
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    distance_m: float | None = None

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class TrackingDecision:
    target: TrackingTarget
    error_x: float
    error_y: float
    pan_delta: float
    tilt_delta: float
    position: GimbalPosition | None


def select_nearest_target(targets: list[TrackingTarget]) -> TrackingTarget | None:
    if not targets:
        return None
    targets_with_distance = [target for target in targets if target.distance_m is not None]
    if targets_with_distance:
        return min(targets_with_distance, key=lambda target: float(target.distance_m))
    # With a monocular camera, a larger face box is the best available proximity proxy.
    return max(targets, key=lambda target: target.area)


def select_tracking_target(
    targets: list[TrackingTarget],
    *,
    previous_target: TrackingTarget | None,
    stickiness: float,
) -> TrackingTarget | None:
    baseline = select_nearest_target(targets)
    if baseline is None or previous_target is None or stickiness <= 0:
        return baseline

    def score(target: TrackingTarget) -> float:
        continuity_bonus = 1.0 + _target_closeness(previous_target, target) * stickiness
        return target.area * continuity_bonus

    return max(targets, key=score)


def select_locked_target(
    targets: list[TrackingTarget],
    *,
    locked_target: TrackingTarget,
    closeness_threshold: float = 0.75,
) -> TrackingTarget | None:
    if not targets:
        return None
    best_match = max(targets, key=lambda target: _target_closeness(locked_target, target))
    if _target_closeness(locked_target, best_match) < closeness_threshold:
        return None
    return best_match


class _PIDAxis:
    """Single-axis PID controller with anti-windup and output clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        dead_zone: float = 0.0,
        max_output: float = 5.0,
        max_integral: float = 2.0,
        integral_active_zone: float = 0.0,
        derivative_filter_alpha: float = 0.6,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dead_zone = dead_zone
        self.max_output = max_output
        self.max_integral = max_integral
        self.integral_active_zone = integral_active_zone
        self.derivative_filter_alpha = derivative_filter_alpha
        self._integral: float = 0.0
        self._prev_error: float | None = None
        self._filtered_derivative: float = 0.0

    def compute(self, error: float, dt_s: float) -> float:
        if dt_s <= 0:
            return 0.0

        if abs(error) <= self.dead_zone:
            error = 0.0

        p_term = self.kp * error

        if self._prev_error is not None:
            raw_derivative = (error - self._prev_error) / dt_s
        else:
            raw_derivative = 0.0
        alpha = self.derivative_filter_alpha
        self._filtered_derivative = (
            alpha * raw_derivative + (1 - alpha) * self._filtered_derivative
        )
        d_term = self.kd * self._filtered_derivative

        can_integrate = self.integral_active_zone <= 0 or abs(error) <= self.integral_active_zone
        next_integral = self._integral
        if error == 0.0 or not can_integrate:
            next_integral *= max(0.0, 1.0 - dt_s * 4.0)
        else:
            next_integral += error * dt_s
        next_integral = max(-self.max_integral, min(next_integral, self.max_integral))
        i_term = self.ki * next_integral

        self._prev_error = error

        output = p_term + i_term + d_term
        clamped_output = max(-self.max_output, min(output, self.max_output))

        if can_integrate and clamped_output != output and error * clamped_output > 0:
            i_term = self.ki * self._integral
            clamped_output = max(-self.max_output, min(p_term + i_term + d_term, self.max_output))
        else:
            self._integral = next_integral
        return clamped_output

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._filtered_derivative = 0.0


class PanTiltTracker:
    """PID-based tracker that converts target boxes into gimbal movements."""

    def __init__(
        self,
        gimbal: PanTiltGimbal,
        config: TrackingConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gimbal = gimbal
        self.config = config
        self._clock = clock
        self._last_update_ms = 0.0
        self._target_lost_at_ms: float | None = None
        self._centered_after_loss = False
        self._last_target: TrackingTarget | None = None
        self._last_target_sample_ms: float | None = None
        self._locked_target_last_seen_ms: float | None = None
        self._last_raw_error_x: float | None = None
        self._last_raw_error_y: float | None = None
        self._last_pan_delta: float = 0.0
        self._last_tilt_delta: float = 0.0

        self._pan_pid = _PIDAxis(
            kp=config.pan_gain,
            ki=config.pan_ki,
            kd=config.pan_kd,
            dead_zone=config.dead_zone_x,
            max_output=config.max_pan_step_degrees,
            max_integral=_integral_limit(config.pan_ki, config.max_pan_step_degrees),
            integral_active_zone=config.pan_integral_zone,
            derivative_filter_alpha=config.derivative_filter_alpha,
        )
        self._tilt_pid = _PIDAxis(
            kp=config.tilt_gain,
            ki=config.tilt_ki,
            kd=config.tilt_kd,
            dead_zone=config.dead_zone_y,
            max_output=config.max_tilt_step_degrees,
            max_integral=_integral_limit(config.tilt_ki, config.max_tilt_step_degrees),
            integral_active_zone=config.tilt_integral_zone,
            derivative_filter_alpha=config.derivative_filter_alpha,
        )

    async def update(
        self,
        targets: list[TrackingTarget],
        frame_width: int,
        frame_height: int,
        *,
        target_age_ms: float = 0.0,
    ) -> TrackingDecision | None:
        now_ms = self._clock() * 1000
        target = self._select_target(targets, now_ms=now_ms)
        if target is None or frame_width <= 0 or frame_height <= 0:
            return None
        self._target_lost_at_ms = None
        self._centered_after_loss = False
        self._last_target = target
        self._locked_target_last_seen_ms = now_ms

        error_x = (target.center_x - frame_width / 2) / (frame_width / 2)
        error_y = (target.center_y - frame_height / 2) / (frame_height / 2)

        error_x, error_y = self._predict_errors(
            error_x, error_y, now_ms=now_ms, target_age_ms=target_age_ms
        )
        shaped_error_x = _shape_error(error_x, self.config.pan_response_exponent)
        shaped_error_y = _shape_error(error_y, self.config.tilt_response_exponent)

        dt_s = max(1e-3, (now_ms - self._last_update_ms) / 1000.0) if self._last_update_ms > 0 else 0.02

        pan_delta = self._pan_pid.compute(shaped_error_x, dt_s) * self.config.pan_direction
        tilt_delta = 0.0
        if self.config.tilt_enabled:
            tilt_delta = self._tilt_pid.compute(shaped_error_y, dt_s) * self.config.tilt_direction

        if (
            self.config.stale_detection_timeout_ms > 0
            and target_age_ms > self.config.stale_detection_timeout_ms
        ):
            pan_delta = 0.0
            tilt_delta = 0.0
        pan_delta = _limit_delta_change(
            pan_delta,
            previous=self._last_pan_delta,
            max_change=_adaptive_delta_change_limit(
                abs(error_x),
                self.config.max_delta_change_degrees,
            ),
        )
        tilt_delta = _limit_delta_change(
            tilt_delta,
            previous=self._last_tilt_delta,
            max_change=_adaptive_delta_change_limit(
                abs(error_y),
                self.config.max_delta_change_degrees,
            ),
        )

        should_move = (
            abs(pan_delta) > self.config.min_effective_pan_delta
            or abs(tilt_delta) > self.config.min_effective_tilt_delta
        ) and now_ms - self._last_update_ms >= self.config.min_update_interval_ms

        position = None
        if should_move:
            position = await self.gimbal.move_by(
                pan_delta,
                tilt_delta,
                self.config.move_time_ms,
            )
            self._last_update_ms = now_ms
            self._last_pan_delta = pan_delta
            self._last_tilt_delta = tilt_delta
        else:
            self._last_pan_delta = 0.0
            self._last_tilt_delta = 0.0

        return TrackingDecision(
            target=target,
            error_x=error_x,
            error_y=error_y,
            pan_delta=pan_delta,
            tilt_delta=tilt_delta,
            position=position,
        )

    async def target_lost(self) -> GimbalPosition | None:
        """Center once after a configurable target-loss grace period."""
        now_ms = self._clock() * 1000
        if self._target_lost_at_ms is None:
            self._target_lost_at_ms = now_ms
            self._reset_motion_state()
            return None
        if (
            not self.config.center_on_target_lost
            or self._centered_after_loss
            or now_ms - self._target_lost_at_ms
            < self.config.target_lost_timeout_seconds * 1000
        ):
            return None
        self._centered_after_loss = True
        self._reset_motion_state()
        return await self.gimbal.center(self.config.move_time_ms)

    def _reset_motion_state(self) -> None:
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._last_target = None
        self._last_target_sample_ms = None
        self._locked_target_last_seen_ms = None
        self._last_raw_error_x = None
        self._last_raw_error_y = None
        self._last_pan_delta = 0.0
        self._last_tilt_delta = 0.0

    def _select_target(
        self,
        targets: list[TrackingTarget],
        *,
        now_ms: float,
    ) -> TrackingTarget | None:
        baseline = select_tracking_target(
            targets,
            previous_target=self._last_target,
            stickiness=self.config.target_stickiness,
        )
        if baseline is None or self._last_target is None:
            return baseline

        locked_target = select_locked_target(
            targets,
            locked_target=self._last_target,
        )
        if locked_target is not None:
            return locked_target

        if self.config.target_lock_timeout_ms <= 0:
            return baseline
        if self._locked_target_last_seen_ms is None:
            return baseline
        if now_ms - self._locked_target_last_seen_ms < self.config.target_lock_timeout_ms:
            LOGGER.debug(
                "holding target lock for %.0fms more before switching",
                self.config.target_lock_timeout_ms
                - (now_ms - self._locked_target_last_seen_ms),
            )
            return None
        return baseline

    def _predict_errors(
        self,
        error_x: float,
        error_y: float,
        *,
        now_ms: float,
        target_age_ms: float,
    ) -> tuple[float, float]:
        if self._last_target_sample_ms is None:
            self._last_target_sample_ms = now_ms
            self._last_raw_error_x = error_x
            self._last_raw_error_y = error_y
            return error_x, error_y

        sample_dt_s = max(1e-3, (now_ms - self._last_target_sample_ms) / 1000.0)
        error_velocity_x = (error_x - float(self._last_raw_error_x)) / sample_dt_s
        error_velocity_y = (error_y - float(self._last_raw_error_y)) / sample_dt_s
        lead_s = self.config.prediction_lead_seconds + max(0.0, target_age_ms) / 1000.0
        predicted_x = max(-1.0, min(1.0, error_x + error_velocity_x * lead_s))
        predicted_y = max(-1.0, min(1.0, error_y + error_velocity_y * lead_s))
        self._last_target_sample_ms = now_ms
        self._last_raw_error_x = error_x
        self._last_raw_error_y = error_y
        return predicted_x, predicted_y


def _target_closeness(previous: TrackingTarget, current: TrackingTarget) -> float:
    dx = previous.center_x - current.center_x
    dy = previous.center_y - current.center_y
    distance = (dx * dx + dy * dy) ** 0.5
    max_span = max(previous.width, previous.height, current.width, current.height, 1.0)
    closeness = max(0.0, 1.0 - distance / (max_span * 2.5))
    return closeness * 4.0


def _integral_limit(ki: float, max_output: float) -> float:
    if ki <= 0:
        return 0.0
    return max_output * 0.45 / ki


def _shape_error(error: float, exponent: float) -> float:
    if error == 0.0 or exponent == 1.0:
        return error
    return abs(error) ** exponent * (1.0 if error > 0 else -1.0)


def _limit_delta_change(desired: float, *, previous: float, max_change: float) -> float:
    if max_change <= 0:
        return desired
    lower = previous - max_change
    upper = previous + max_change
    return max(lower, min(desired, upper))


def _adaptive_delta_change_limit(error_magnitude: float, base_limit: float) -> float:
    if base_limit <= 0:
        return base_limit
    clamped_error = max(0.0, min(1.0, error_magnitude))
    # Keep near-center motion conservative, but let the head accelerate harder
    # when the face moves far away from center.
    return base_limit * (1.0 + clamped_error * 1.5)
