Offline Vision-X
A local laptop prototype for camera capture, people detection baseline, multi-object tracking, optional calibrated ground-plane coordinates, and a LAN dashboard. It runs without cloud services. No YOLO or remote inference is used.

Run
Install Python 3.10+ and the dependencies, then:

python -m pip install -r requirements.txt
python app.py
Open http://127.0.0.1:8765. On a trusted local network, open http://<laptop-LAN-IP>:8765 from another device. Windows Firewall may require allowing Python on Private networks. Camera and network access depend on host permissions.

Use python app.py --camera 1 for another webcam. Select --mode industrial for the industrial rules view. The app remains usable when OpenCV or a camera is absent; status will explicitly show that detection/video is unavailable.

What works now
Real webcam capture and MJPEG stream when OpenCV and a camera are available.
OpenCV's built-in HOG people detector as a baseline, never presented as a custom-trained model.
Session-local multi-object tracking using greedy IoU association, missed-frame recovery, velocity and direction estimates.
A local JSON API: /api/status, /api/tracks, /api/events.
Configurable homography from image points to room-plane coordinates (disabled until calibrated).
Local-only HTTP dashboard and a rule-based industrial event framework.
Limits
HOG detects people only and can be inaccurate, slow, or fail under occlusion. It is not a custom detector and provides no industrial component detection. IDs are session-local; IoU can swap IDs when objects cross. Pixel speed is not metric speed. World coordinates remain null until the camera is calibrated. No GPS, drone SDK, thermal camera, or phone-camera protocol is included. Industrial alerts are rule-based and require zones/expected counts to be configured. Do not use this prototype as a safety system.

See RUN.md, ARCHITECTURE.md, MODEL_TRAINING.md, and LIMITATIONS.md.
