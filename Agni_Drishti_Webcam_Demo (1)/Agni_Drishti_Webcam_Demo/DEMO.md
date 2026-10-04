# Demo

With a webcam and OpenCV installed, run the app and move through the camera view. The HOG baseline may emit person boxes; tracks display session IDs and screen-space direction/speed. `/api/status` reports actual runtime state and measured processing throughput. If no camera is available, the dashboard should report that state rather than displaying fabricated footage or detections.

To verify LAN viewing, connect a second device to the same private Wi-Fi and visit the laptop's LAN IP on port 8765.
