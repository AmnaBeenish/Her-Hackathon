# Repolymer AI 🌸♻️

An interactive, browser-only factory simulation: AI plastic sorting → shredder → smart pyrolysis reactor → circular fuel tanks.
Everything runs in the browser (React + TensorFlow.js). There's no Python and no backend.

## Run it (Windows)

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** (use Chrome or Edge for the camera).

> If PowerShell says *"running scripts is disabled"*, use `npm.cmd install` and `npm.cmd run dev`.

Production build: `npm run build`, then `npm run preview`.

## Input modes

| Mode | What it does |
|---|---|
| **Demo Waste Tray** | Tap or drag 8 illustrated items onto the conveyor. No camera or internet needed. |
| **Live Camera** | Auto-scans the glowing viewfinder. A result locks in after 3 matching frames. **Phone Screen Boost** sharpens photos shown on a phone. |
| **Snap & Scan** | Freezes the frame, then runs 3 preprocessing passes that vote on one result. |
| **Upload Photo** | Analyses any JPG, PNG, or WEBP image. |

## How the AI works

1. **Screen-to-cam preprocessing:** centre ROI crop, a contrast stretch that ignores glare, auto-gamma brightness balance, and an unsharp mask (Boost).
2. **COCO-SSD (TensorFlow.js):** finds objects. If it sees a phone or monitor, it zooms into the screen and runs again.
3. **Polymer & Screen-Glare Visual Analyzer:** colour, saturation, glare, edge density, and shape cues map objects to PET, HDPE, LDPE, PP, PVC, multi-layer foil, organic, or metal/glass.

The COCO-SSD weights (about 5 MB) download from Google's CDN on first use. Offline, the app switches to analyzer-only mode, and the Demo Tray always works.

Accepted: PET #1, HDPE #2, LDPE #4, PP #5. Rejected: PVC #3 (chlorine), multi-layer foil wrappers, organics, metal/glass.

Reactor auto-temperatures: PET 400 °C, PP 420 °C, LDPE 430 °C, HDPE 450 °C. Yields and CO₂ figures are simplified demo estimates.

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera blocked | Click the camera icon in the address bar and allow it, or use Demo Tray / Upload. |
| `npm install` errors on OneDrive | Pause OneDrive syncing during install, or copy the project to a local folder (e.g. `C:\dev`). |
| AI badge says "offline mode" | No internet for the model download. Detection still works using the analyzer. |

The original Python/OpenCV prototype is still available in `CV.py`.
