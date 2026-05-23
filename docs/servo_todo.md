# Servo TODO

Real servo control is intentionally not implemented in v1. Before adding a
driver, confirm:

- exact Orange Pi model and OS image;
- GPIO/PWM pin mapping;
- whether a PCA9685 or other servo controller board will be used;
- servo voltage and current requirements;
- external power supply design and common ground wiring;
- signal voltage and level shifting requirements;
- channel count;
- safe angle range for each joint;
- neutral pose;
- max speed and acceleration;
- physical collision limits;
- emergency stop behavior.

The current `NoopServoController` logs all action intents and is safe to run on
machines without hardware.
