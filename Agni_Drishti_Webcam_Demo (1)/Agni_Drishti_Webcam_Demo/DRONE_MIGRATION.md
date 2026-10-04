# Drone migration

Implement a camera adapter that yields RGB frames and a localization provider using the drone's calibrated camera and navigation sensors. Keep detector, tracker, dashboard, and event APIs unchanged. Validate latency, gimbal motion, rolling shutter, altitude changes, coordinate transforms, and identity continuity with recorded flight data before field use. This repository does not control a drone and contains no autonomous action or targeting.
