# Voice-Triggered Face Tracking Plan

## Goal

After the server recognizes a tracking intent from speech, the edge device
starts or activates the USB camera, detects faces continuously, selects the
nearest person, and controls the two-axis gimbal to keep that face near the
frame center.

## Implemented Foundation

- SongJia CH340 USB serial protocol and safe pan-tilt SDK.
- Per-axis servo IDs, direction, neutral position, and physical angle limits.
- Dry-run mode, grouped movement, stop command, incremental movement, and
  high-level action support.
- Multi-target selection: use measured distance when available, otherwise use
  the largest face box as the nearest-person approximation.
- Tracking control: center dead zone, proportional gain, maximum step, movement
  time, and update-rate limiting.

## Main Program Integration

1. Extend server intents with `start_tracking` and `stop_tracking`.
2. Add a `FaceDetector` adapter that returns bounding boxes and confidence.
3. Add a `FaceTrackingWorker` that owns tracking state and calls
   `PanTiltTracker.update()` for each detection frame.
4. Keep the camera worker idle or low-rate until `start_tracking` is received.
5. On multiple faces, pass all valid face boxes to the tracker.
6. On target loss, hold the last position briefly, search within safe limits,
   then return to neutral after a configurable timeout.
7. Stop tracking immediately on `stop_tracking`, emergency stop, serial error,
   or camera error.

## Required Calibration

- Measure safe pan and tilt limits on the assembled robot.
- Confirm pan and tilt inversion flags.
- Confirm camera orientation relative to gimbal axes.
- Tune dead zones and gains using a stationary face first.
- Measure the highest safe update rate without shaking or cable strain.

## Acceptance Tests

- Voice intent enables and disables tracking.
- A centered face does not cause repeated servo movement.
- A face moving left/right causes bounded pan correction in the correct direction.
- A face moving up/down causes bounded tilt correction in the correct direction.
- The largest face is selected when no depth estimate exists.
- Physical limits are never exceeded.
- Target loss and serial disconnect do not cause uncontrolled movement.
