# Servo TODO

The first SongJia USB pan-tilt SDK is implemented in
`ai_robot_edge.devices.gimbal`. See `docs/servo_sdk.md`.

Confirmed:

- Atlas 200I DK A2 running Ubuntu 22.04;
- SongJia USB controller using CH340 at 115200 baud;
- stable path `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`;
- servo ID `0` is pan and servo ID `1` is tilt;
- protocol supports grouped movement and emergency stop commands.

Still required before enabling live tracking:

- measure safe pan and tilt angle limits;
- confirm whether either axis direction must be inverted;
- confirm neutral pose and cable clearance;
- confirm max safe movement speed;
- validate emergency power-disconnect procedure.

The sample configuration keeps real movement disabled and uses dry-run mode.
