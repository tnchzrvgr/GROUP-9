# AI-Powered Real-Time Object Detection and Voice Announcement System
Smartphone camera → Laptop web app → Pre-trained model → Voice output

## How to run

1. **Phone as webcam.** Install one of these on the phone and the matching desktop client on the laptop
   (phone and laptop on the same Wi-Fi, or connected by USB):
   - DroidCam (Android/iOS), Iriun Webcam, Camo, or EpocCam.
   - The phone then appears as a normal webcam on the laptop.
2. **Start a local web server** in this folder (the camera API requires `http://localhost` or HTTPS, not `file://`):
   - Python: `python -m http.server 8000`
   - or Node: `npx serve`
   - or VS Code → "Live Server" extension.
3. Open **http://localhost:8000** in Chrome or Edge (internet is needed once to load TensorFlow.js and the model).
4. Click **Start Camera** and allow permission. If the phone camera is not listed yet, make sure its webcam app is connected and click **Refresh cameras**. Choose the phone camera in **Camera source** (for example, "Iriun Webcam"); changing the selection switches cameras.
5. Show objects to the phone camera. Labels, boxes and confidence appear; the laptop speaks "<object> detected".

## System integration diagram

```
 ┌──────────────┐   Wi-Fi / USB    ┌─────────────────┐
 │  Smartphone  │ ───────────────► │ Phone-as-webcam │
 │   Camera     │  video stream    │ desktop client  │
 └──────────────┘                  └────────┬────────┘
                                            │ appears as a webcam device
                                            ▼
 ┌─────────────────────────── LAPTOP: Web Application (Browser) ───────────────────────────┐
 │                                                                                          │
 │   getUserMedia (WebRTC)  ──►  <video> live feed  ──►  COCO-SSD model (TensorFlow.js)     │
 │                                      │                       │ labels, boxes, scores     │
 │                                      ▼                       ▼                           │
 │                              UI: video + bounding      Announcement logic                │
 │                              boxes + detection list    (confirmation + cooldown)         │
 │                                                              │                           │
 │                                                              ▼                           │
 │                                                   SpeechSynthesis API ──► 🔊 Speaker     │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Categories

Each detected object is also shown with its category (Person, Vehicle, Animal, Kitchen, Food, Furniture, Electronic, Appliance, Accessory, Sports, Outdoor, Indoor), based on the COCO dataset's supercategories. The category appears on the box label, in the detection list and in the log. Tick **Also say category** to have it spoken too (e.g. "bottle detected. Category: Kitchen").

## Technologies used

| Component | Technology | What it does |
|---|---|---|
| Camera link | DroidCam / Iriun / Camo (phone-as-webcam) | Streams the phone camera to the laptop and exposes it as a webcam device |
| Camera access | Browser `getUserMedia` (WebRTC media API) | Opens the selected camera and feeds frames into a `<video>` element |
| Object detection | TensorFlow.js + COCO-SSD (MobileNet v2) | Pre-trained model that detects 80 common object classes (person, cup, bottle, cell phone, laptop, book, apple, etc.) and returns class, bounding box and confidence |
| Interface | HTML, CSS, JavaScript | Live video, canvas overlay for boxes, detection list, announcement log, status badges |
| Voice | Browser `SpeechSynthesis` API | Text-to-speech that says "<object> detected" |

## How it works (for the explanation part of the demo)

1. **Camera feed reaches the laptop:** the phone app streams video to its desktop client, which registers a virtual webcam. The browser lists it via `enumerateDevices()` and opens it with `getUserMedia()`.
2. **Model identifies objects:** about every 100 ms the current video frame is passed to COCO-SSD running in the browser through TensorFlow.js (WebGL-accelerated). It returns objects with class name, bounding box and confidence. Results below the confidence slider value are ignored.
3. **Voice announcement:** `SpeechSynthesisUtterance("apple detected")` is spoken through the laptop speakers.
4. **Avoiding repeated speech:**
   - *Confirmation:* an object must be seen in 3 consecutive detections before it is announced.
   - *Once per appearance:* it is not announced again while it stays in view.
   - *Reset:* it can be announced again only after it has left the view for 2 seconds.
   - *Cooldown:* and at least N seconds (slider, default 5) since its last announcement.
   - *Global gap:* never interrupts or overlaps speech already playing.

## Demo script: at least 5 object types

COCO-SSD recognises these well, and most are easy to find: **cell phone, bottle, cup, book, laptop, keyboard, mouse, scissors, backpack, banana, apple, orange, clock, person, chair**.
Suggested demo order: person (yourself) → bottle → cup → book → cell phone (use a second phone) → banana or apple.
Tips: good lighting, hold the object ~30–60 cm from the camera, plain background.

## Files

- `index.html` – page layout and script loading
- `style.css` – styling
- `app.js` – camera, detection loop, drawing, announcement logic

## Troubleshooting

- **Phone camera not in the dropdown:** make sure the phone app and desktop client are running and connected, then click Start Camera again (device names appear only after permission is granted).
- **DroidCam is active but the browser preview is solid green:** the browser has opened the virtual camera, but it is receiving unusable frames. Close other apps using the camera (including OBS, Teams, or Windows Camera), restart the DroidCam desktop client and phone connection, then reload the detector and select DroidCam again. If the DroidCam client preview itself is also green, fix the phone feed in that client first.
- **"Camera busy":** close other apps using the camera (Zoom, Teams, the phone client's own preview).
- **No voice:** click "Test voice"; check the laptop volume and that "Voice announcements" is ticked. Use Chrome or Edge.
- **Model failed to load:** check internet access (the libraries load from a CDN).
- **Low FPS:** lower the camera resolution in the phone app, or change `mobilenet_v2` to `lite_mobilenet_v2` in `app.js` (faster, slightly less accurate).
