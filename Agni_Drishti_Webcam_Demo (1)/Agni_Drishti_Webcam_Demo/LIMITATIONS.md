# Limitations and honest status

- Detector: OpenCV HOG's bundled people baseline only. There is no project-trained model or dataset in this workspace.
- Industrial mode: object observations/events framework only; this prototype has no reliable component detector.
- Tracking: greedy IoU; IDs can switch at crossings and restart at each run.
- Localization: unavailable until configured/calibrated; no metric velocity from pixels.
- Network: unauthenticated HTTP for a trusted LAN; no TLS or internet exposure.
- Future modules: drone SDK, GPS/UWB/IMU, thermal sensor, phone-camera protocol, C++ acceleration, and manufacturing-specific model/data.

No production accuracy, industrial safety rating, or field performance is claimed.
