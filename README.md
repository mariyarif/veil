# Veil

**Check your photo before you post it.**

Veil finds the clues a photo gives away about where you are, then helps you remove them before you post. A stalker can use a single photo to find where you live or spend your time, so Veil looks at what's visible in the picture, not only the hidden data.

Live demo: https://veil-ufor.onrender.com

---

## The problem

Photos reveal location in three ways people rarely notice:

1. **Hidden metadata.** Phones often store exact GPS coordinates inside the image file.
2. **Readable text.** Number plates, shop signs, and house numbers in the background.
3. **The scene itself.** Streets and buildings can be enough to narrow down a location.

Most existing tools only remove GPS metadata. Veil also checks what's visible, and it checks again after the fix.

## What Veil does

1. **Upload** a photo. Processing happens on the server, and nothing is stored after the result is returned.
2. **Scan** the photo for three things:
   - GPS coordinates in the EXIF metadata
   - Readable text, using OCR (Tesseract)
   - Vehicle number plates, using Indian plate grammar to validate matches
3. **Review** a 0 to 100 risk score, the reasons for it, and boxes drawn over every detected region.
4. **Remove** location data and obscure the flagged areas. The image is rebuilt from raw pixels, so no metadata carries over.
5. **Re-scan** the cleaned photo and show the risk before and after, with the remaining reasons.

## Why it's different

- **It checks after the fix.** Veil doesn't stop at removing metadata. It re-scans the cleaned image and shows what changed.
- **Honest scoring.** The score is a simple, explainable rule set: GPS adds 50 points, a readable number plate adds 40, and other readable text adds up to 30. Every point is traceable to a reason shown on screen.
- **Privacy by default.** Stateless processing, metadata stripped on output, and no accounts or tracking.

## Privacy design

| Question | Answer |
|---|---|
| Is the photo stored? | No. It's processed in memory and discarded after the response. |
| Is metadata kept? | GPS and other EXIF data are removed from the cleaned file. |
| Who sees the result? | Only the person who uploaded the photo. |
| Are cookies or accounts used? | No. |
| What is not checked yet? | Whether the scene itself identifies a location. See limitations. |

Privacy is a design constraint here, not a feature added later. Before a real launch, we would run a Data Protection Impact Assessment on this flow.

## Limitations

We'd rather tell you what Veil misses than let you assume it catches everything.

- **The score is a heuristic.** It counts visible clues. It does not measure how easily someone could find you.
- **Scene-based location is not analysed.** A photo can score 0 and still show a recognisable street or café.
- **OCR misses stylised fonts.** Some signboards, especially decorative ones, aren't read reliably.
- **Obscuring is broad.** Every flagged region is blurred, which can also blur things that are harmless. This is a deliberate trade-off.
- **A clean result is not a safe photo.** Always check the background before posting.

## Tech stack

- **Backend:** Python, Flask, Gunicorn
- **Image processing:** Pillow
- **OCR:** Tesseract OCR via pytesseract
- **Frontend:** HTML, CSS, and vanilla JavaScript (no framework)
- **Deployment:** Docker, Render

## Run it locally

Requirements: Python 3.11+ and Tesseract OCR installed.

```bash
git clone https://github.com/mariyarif/veil.git
cd veil
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

On Windows, if Tesseract isn't on your PATH, set the location first:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python app.py
```

## Run with Docker

```bash
docker build -t veil .
docker run -p 10000:10000 veil
```

Then open http://localhost:10000.

## Project structure
veil/
├── app.py # Flask app: scanning, scoring, obscuring, routes
├── templates/
│ └── index.html # Single-page interface
├── requirements.txt
├── Dockerfile
└── .dockerignore
## Future work

- Scene-based location risk using a vision model, returning a region and a confidence level
- Tap-to-select boxes, so users choose what to blur
- Client-side metadata stripping, so the photo never leaves the phone unstripped
- Better plate detection with a dedicated detector
- A browser extension for social platforms

## Team

- MARIYA ARIF BIN HAID
- SAHNEN TABASSUM
