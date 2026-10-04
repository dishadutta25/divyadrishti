# Architecture

`app.py` owns camera capture, optional OpenCV HOG baseline, greedy IoU association, calibration mapping, and the local stdlib HTTP server. The browser dashboard consumes JSON endpoints and MJPEG. No cloud endpoint exists.

The seams are intentionally small: replace the detector behind the frame loop; replace `Tracker.update` behind the track update boundary; replace `localize` with a LocalizationProvider; industrial rules consume tracked-object observations. Current tracking/localization logic is Python for a low-dependency laptop prototype. Performance-critical C++ and external camera adapters remain future work.

Localization requires four image points and matching room-plane meter points in `config/localization.yaml`. Coordinates describe the calibrated plane, not GPS. A homography only applies when the camera and target ground plane remain fixed.
