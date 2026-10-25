import io, json, os, re
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image, ImageFilter, ImageOps

try:
    import pytesseract
    from pytesseract import Output
    # Local Windows: set TESSERACT_CMD to the full path of tesseract.exe.
    # On Render (Docker), the Dockerfile sets this to /usr/bin/tesseract.
    tess_cmd = os.environ.get("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

GPS_IFD = 0x8825                     # EXIF tag that holds GPS data
MAX_UPLOAD = 10 * 1024 * 1024
OCR_MAX_SIDE = 1600                  # analyse at this size; plenty for text detection
OCR_SCALES = (2,)                    # one upscale is enough once the image is resized
OCR_PSMS = ("--psm 11", "--psm 6")   # sparse text, and uniform text blocks
OCR_MIN_CONF = 40                    # minimum confidence for a box to be blurred
OCR_WORD_CONF = 75                   # minimum confidence for a word to count toward the score
ALNUM = re.compile(r"[A-Za-z0-9]")
LETTERS = re.compile(r"[A-Za-z]")
# Indian-style plate, e.g. TS 07 GZ 4821 or TS07GZ4821
PLATE = re.compile(r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


def load_image(file_storage):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)   # apply rotation before analysis
    return img.convert("RGB")


def find_gps(img):
    exif = img.getexif()
    gps = exif.get_ifd(GPS_IFD) if exif else {}
    return bool(gps)


def ocr_pass(gray, scale, psm, back=1.0):
    """One OCR run at one scale. Returns tokens in original-image coordinates."""
    big = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
    data = pytesseract.image_to_data(big, output_type=Output.DICT, config=psm)
    k = back / scale
    tokens = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            continue
        tokens.append({
            "text": text,
            "conf": conf,
            "x": int(data["left"][i] * k),
            "y": int(data["top"][i] * k),
            "w": int(data["width"][i] * k),
            "h": int(data["height"][i] * k),
        })
    return tokens


def find_text(img):
    """Return text boxes in original-image pixel coordinates, de-duplicated across OCR passes."""
    if not OCR_AVAILABLE:
        return []
    orig_w = img.width
    work = img.copy()
    work.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))   # shrink large phone photos first
    back = orig_w / work.width                     # maps working coords back to original
    gray = ImageOps.autocontrast(work.convert("L"))
    tokens = []
    try:
        for scale in OCR_SCALES:
            for psm in OCR_PSMS:
                tokens.extend(ocr_pass(gray, scale, psm, back))
    except pytesseract.TesseractNotFoundError:
        return []

    boxes, seen = [], set()
    for t in tokens:
        ld = ALNUM.findall(t["text"])
        if t["conf"] < OCR_MIN_CONF or len(ld) < 2 or len(ld) / len(t["text"]) < 0.6:
            continue
        # Overlapping passes find the same text; keep one box per location
        key = (t["text"].upper(), t["x"] // 6, t["y"] // 6)
        if key in seen:
            continue
        seen.add(key)
        boxes.append({
            "text": t["text"], "conf": t["conf"],
            "x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"],
        })
    return boxes


def find_plate(boxes):
    """Check whether the readable text forms a number plate."""
    joined = " ".join(b["text"].upper() for b in boxes)
    return bool(PLATE.search(joined))


def meaningful_words(boxes):
    """Confident words with at least 3 letters. Filters out noise like '2h' or '37,,'."""
    return sorted({
        b["text"] for b in boxes
        if b["conf"] >= OCR_WORD_CONF and len(LETTERS.findall(b["text"])) >= 3
    })


def score(has_gps, boxes):
    """Simple, explainable risk score: 0 to 100, with reasons."""
    points, reasons = 0, []
    if has_gps:
        points += 50
        reasons.append("Photo contains GPS coordinates in its metadata")
    if find_plate(boxes):
        points += 40
        reasons.append("A vehicle number plate is readable in the image")
    words = meaningful_words(boxes)
    if words:
        points += min(30, 10 * len(words))
        reasons.append(f"Other readable text in the image: {', '.join(words[:8])}")
    return min(points, 100), reasons


def scan_image(img):
    has_gps = find_gps(img)
    boxes = find_text(img)
    risk, reasons = score(has_gps, boxes)
    return {
        "risk": risk,
        "reasons": reasons,
        "has_gps": has_gps,
        "text_boxes": boxes,
        "ocr_enabled": OCR_AVAILABLE,
    }


def clean_image(img, boxes):
    """Blur the given boxes and drop all metadata by building a fresh image."""
    clean = img.copy()
    for b in boxes:
        region = clean.crop((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=12))
        clean.paste(blurred, (b["x"], b["y"]))
    # Copy pixels into a new image so no EXIF, GPS, or other metadata carries over
    stripped = Image.new("RGB", clean.size)
    stripped.putdata(list(clean.getdata()))
    return stripped


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    if "photo" not in request.files:
        return jsonify({"error": "no photo"}), 400
    img = load_image(request.files["photo"])
    return jsonify(scan_image(img))


@app.route("/api/clean", methods=["POST"])
def api_clean():
    if "photo" not in request.files:
        return jsonify({"error": "no photo"}), 400
    img = load_image(request.files["photo"])
    boxes = json.loads(request.form.get("boxes", "[]"))
    cleaned = clean_image(img, boxes)
    buf = io.BytesIO()
    cleaned.save(buf, format="JPEG", quality=92)   # saved with no EXIF argument
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", download_name="veil_clean.jpg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)