# Pan-Tilt Gimbal SDK

## Hardware Confirmed

- Edge device: Atlas 200I DK A2
- Servo controller: SongJia USB controller with CH340 serial converter
- Stable serial path: `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`
- Baud rate: `115200`
- Pan servo ID: `0`
- Tilt servo ID: `1`
- Protocol angle range: `0-270` degrees maps to PWM `500-2500`
- USB camera: `/dev/video0` and `/dev/video1`
- USB microphone: ALSA `card 0, device 0`

## Safety Defaults

The sample config keeps `servo.enabled: false`, `tracking.enabled: false`, and
`servo.dry_run: true`. The robot-head operating limits use `135` degrees as
the forward/level reset pose:

- Pan/yaw: `75-195` degrees, or `-60/+60` degrees relative to reset.
- Tilt/pitch: `105-165` degrees, or `-30/+30` degrees relative to reset.

The first Atlas hardware test successfully sent center, stop, pan `105-165`
degrees (`-30/+30`), and tilt `120-150` degrees (`-15/+15`) commands without
serial errors. These results verify the command path, not mechanical clearance
or the physical meaning of positive/negative movement. Before unattended live
movement:

1. Confirm pan and tilt physical directions.
2. Confirm the configured safe limits leave enough cable clearance.
3. Confirm the neutral pose does not collide with the frame or cables.
4. Run a dry-run command and inspect the generated protocol.
5. Enable live movement only while someone can disconnect servo power.

## Tracking Tuning Notes

The tracker now uses a conservative PID profile with the same guardrail idea as
`llm-pid-tuner`: small incremental changes, bounded output, and rollback-style
protection against obviously dangerous parameter jumps.

- `pan_gain` / `tilt_gain`: primary responsiveness. Keep these modest because
  the controller output is already in servo degrees per update.
- `pan_ki` / `tilt_ki`: only clean up small steady-state bias. They are gated by
  `pan_integral_zone` / `tilt_integral_zone` so integral does not build while
  the face is far from center.
- `pan_kd` / `tilt_kd` plus `derivative_filter_alpha`: damping against jitter
  and sudden reversals.
- `max_pan_step_degrees` / `max_tilt_step_degrees`: hard output clamp.
- `max_delta_change_degrees`: slew-rate limiter between consecutive commands.
- `min_effective_pan_delta` / `min_effective_tilt_delta`: ignore micro-corrections
  that only make the head buzz.

## SDK Usage

```python
from ai_robot_edge.devices.gimbal import PanTiltGimbal
from ai_robot_edge.vision.tracking import PanTiltTracker, TrackingTarget

gimbal = PanTiltGimbal(config.servo)
tracker = PanTiltTracker(gimbal, config.tracking)

await tracker.update(
    [TrackingTarget(x=100, y=80, width=120, height=120, confidence=0.9)],
    frame_width=640,
    frame_height=480,
)
```

When multiple targets are supplied, the tracker selects the smallest measured
`distance_m`. If no depth measurement exists, it selects the largest face box
as a monocular-camera approximation of the nearest person.

## Safe Diagnostic CLI

Probe configuration without opening or writing to the serial device:

```bash
ai-robot-gimbal --config config/edge.yaml probe
```

Preview a center command without moving:

```bash
ai-robot-gimbal --config config/edge.yaml center
```

Allow a real movement only after limits are verified:

```bash
ai-robot-gimbal --config config/edge.yaml center --live
```

## Main Program Integration

The current person detector only returns presence and confidence. The next
integration step is a face detector that returns bounding boxes. Its worker
should call `PanTiltTracker.update()` only while tracking mode is enabled by a
voice or server intent. On target loss, stop sending updates; optionally center
the gimbal after a configurable timeout.
