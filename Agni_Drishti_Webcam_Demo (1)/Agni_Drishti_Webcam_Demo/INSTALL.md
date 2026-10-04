# Installation

## Windows

Install Python 3.10 or newer, open PowerShell in this directory, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Alternatively, double-click `start.bat`. It creates `.venv`, installs OpenCV and NumPy from `requirements.txt`, and runs the app. Keep the terminal open to see startup errors. If the camera cannot be opened, close other programs using it or try `start.bat --camera 1`.

For a phone or another device, connect it to the same trusted Wi-Fi as the laptop. The terminal prints the laptop's private network URL, and the dashboard repeats it under **PHONE / SAME WI-FI ACCESS**. Open that URL on the phone. If it cannot connect, allow Python through Windows Defender Firewall for **Private networks** and ensure the Wi-Fi is not a guest network with client isolation. Do not port-forward the app: the local dashboard has no login/authentication. CSV download links on the dashboard provide tracking, events, assistance-review, and quality-inspection logs.

Choose **Industrial** in the dashboard to enable the configurable visual line-mark screening and industrial zone rules. Point the webcam at a stable, well-lit inspection surface; edit `quality.yaml` for the inspection ROI and thresholds. Confirm every advisory using the approved quality process. Findings are available on the dashboard/API and are appended to `logs\quality_inspection.csv`; they are not a validated crack/failure diagnosis.

For RTX CUDA inference, install the CUDA-enabled PyTorch build selected for your installed NVIDIA driver from the official PyTorch installation selector, then install `ultralytics`. Place a compatible lightweight YOLO person model at `models\yolov8n.pt` (create the `models` folder if needed). The app will report CUDA as active only after an inference confirms the model is on CUDA. Without these optional components or local weights, the app still runs with the OpenCV HOG CPU baseline.

## Offline use

After dependencies are installed, inference and the dashboard are local. To move to an offline machine, install the same Python version and transfer locally built wheels or this environment's compatible wheel cache; no model or camera footage is downloaded by the program. User checkpoints/datasets must be transferred separately.
