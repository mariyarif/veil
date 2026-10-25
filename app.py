import io, json, os, re, time
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

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
OCR_MAX_SIDE = 1000                  # working size for the whole-image text pass
OCR_SCALES = (2,)
OCR_PSMS = ("--psm 11",)
OCR_MIN_CONF = 40                    # minimum confidence for a text box to be blurred
OCR_WORD_CONF = 75                   # minimum confidence for a word to count toward the score

DET_MAX_SIDE = 1280                  # working size for plate localisation
MAX_CANDIDATES = 10                  # plate candidates verified with OCR per image
PLATE_TIME_BUDGET = 8.0              # seconds of OCR spent on plate candidates
BLUR_UNVERIFIED = os.environ.get("BLUR_UNVERIFIED_PLATES", "1") == "1"

ALNUM = re.compile(r"[A-Za-z0-9]")
LETTERS = re.compile(r"[A-Za-z]")
PLATE_WHITELIST = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ---- Indian plate grammar -------------------------------------------------
STATES = set("""AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP
MZ NL OD OR PB PY RJ SK TN TR TS UK UP WB""".split())
# L = letter, D = digit
TEMPLATES = {
    11: ["LLDDLLLDDDD"],
    10: ["LLDDLLDDDD", "LLDLLLDDDD", "DDLLDDDDLL"],   # last one is the BH series
    9:  ["LLDDLDDDD", "LLDLLDDDD"],
    8:  ["LLDLDDDD"],
}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
                   "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def load_image(file_storage):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)   # apply rotation before analysis
    return img.convert("RGB")


def find_gps(img):
    exif = img.getexif()
    gps = exif.get_ifd(GPS_IFD) if exif else {}
    return bool(gps)


def overlap_ratio(a, b):
    """Intersection area divided by the smaller box's area (0..1)."""
    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / max(1, min(a["w"] * a["h"], b["w"] * b["h"]))


def union_box(items):
    x0 = min(t["x"] for t in items)
    y0 = min(t["y"] for t in items)
    x1 = max(t["x"] + t["w"] for t in items)
    y1 = max(t["y"] + t["h"] for t in items)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


# ---------------------------------------------------------------------------
# Plate text matching
# ---------------------------------------------------------------------------
def _fit(sub, template):
    """Fit a string to a letter/digit template, fixing look-alike characters.
    Returns (corrections, fixed_string) or None."""
    out, fixes = [], 0
    for c, t in zip(sub, template):
        if t == "L":
            if c.isalpha():
                out.append(c)
            elif c in DIGIT_TO_LETTER:
                out.append(DIGIT_TO_LETTER[c]); fixes += 1
            else:
                return None
        else:
            if c.isdigit():
                out.append(c)
            elif c in LETTER_TO_DIGIT:
                out.append(LETTER_TO_DIGIT[c]); fixes += 1
            else:
                return None
    if fixes > 2:
        return None
    fixed = "".join(out)
    if template == "DDLLDDDDLL":
        if fixed[2:4] != "BH":
            return None
    elif fixed[:2] not in STATES:
        return None
    return fixes, fixed


def match_plate(raw):
    """Return a corrected plate string if `raw` contains an Indian-style plate, else None."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    for n in (11, 10, 9, 8):             # longest first so we don't match a slice of a longer plate
        for i in range(0, len(s) - n + 1):
            best = None
            for tpl in TEMPLATES[n]:
                r = _fit(s[i:i + n], tpl)
                if r and (best is None or r[0] < best[0]):
                    best = r
            if best:
                return best[1]
    return None


# ---------------------------------------------------------------------------
# Whole-image OCR (general text + plate fragments)
# ---------------------------------------------------------------------------
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
            "text": text, "conf": conf,
            "x": int(data["left"][i] * k), "y": int(data["top"][i] * k),
            "w": int(data["width"][i] * k), "h": int(data["height"][i] * k),
        })
    return tokens


def read_tokens(img):
    """All raw OCR tokens in original-image coordinates (no filtering)."""
    if not OCR_AVAILABLE:
        return []
    work = img.copy()
    work.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
    back = img.width / work.width
    gray = ImageOps.autocontrast(work.convert("L"))
    tokens = []
    try:
        for scale in OCR_SCALES:
            for psm in OCR_PSMS:
                tokens.extend(ocr_pass(gray, scale, psm, back))
    except Exception:                      # Tesseract missing or crashed
        return []
    return tokens


def filter_text_boxes(tokens):
    boxes, seen = [], set()
    for t in tokens:
        ld = ALNUM.findall(t["text"])
        if t["conf"] < OCR_MIN_CONF or len(ld) < 2 or len(ld) / len(t["text"]) < 0.6:
            continue
        key = (t["text"].upper(), t["x"] // 6, t["y"] // 6)
        if key in seen:
            continue
        seen.add(key)
        boxes.append({"kind": "text", "text": t["text"], "conf": t["conf"],
                      "x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"]})
    return boxes


def plates_from_tokens(tokens):
    """Merge neighbouring OCR fragments (e.g. 'TS', '07', 'GZ4821') into plate boxes."""
    toks = [t for t in tokens if t["conf"] >= 20 and ALNUM.search(t["text"])]
    toks.sort(key=lambda t: t["y"] + t["h"] / 2)
    lines = []
    for t in toks:
        cy = t["y"] + t["h"] / 2
        for line in lines:
            ref = line[0]
            if abs(cy - (ref["y"] + ref["h"] / 2)) < 0.6 * max(ref["h"], t["h"], 1):
                line.append(t)
                break
        else:
            lines.append([t])

    found = []
    for line in lines:
        line.sort(key=lambda t: t["x"])
        i = 0
        while i < len(line):
            acc, hit = "", None
            for j in range(i, min(len(line), i + 6)):
                if j > i:
                    gap = line[j]["x"] - (line[j - 1]["x"] + line[j - 1]["w"])
                    if gap > 2.0 * max(line[j]["h"], 1):
                        break
                acc += line[j]["text"]
                m = match_plate(acc)
                if m:
                    hit = (j, m)
                    break
            if hit:
                j, m = hit
                found.append((m, union_box(line[i:j + 1])))
                i = j + 1
            else:
                i += 1
    return found


# ---------------------------------------------------------------------------
# Plate localisation (OpenCV) + per-crop OCR verification
# ---------------------------------------------------------------------------
def char_blobs(gray):
    """Count character-sized blobs of similar height. Returns (count, consistency 0..1)."""
    h, w = gray.shape[:2]
    if h < 8 or w < 16:
        return 0, 0.0
    s = 64.0 / h
    g = cv2.resize(gray, (max(16, int(w * s)), 64), interpolation=cv2.INTER_AREA)
    best, best_score = (0, 0.0), 0.0
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, b = cv2.threshold(g, 0, 255, flag + cv2.THRESH_OTSU)
        n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        hs = []
        for i in range(1, n):
            cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if 0.25 * 64 <= ch <= 0.95 * 64 and 2 <= cw <= 1.2 * ch:
                hs.append(ch)
        if len(hs) < 3:
            continue
        cons = max(0.0, 1.0 - float(np.std(hs)) / float(np.mean(hs)))
        if len(hs) * cons > best_score:
            best, best_score = (len(hs), cons), len(hs) * cons
    return best


def plate_candidates(img):
    """Find plate-shaped regions without relying on OCR. Boxes are in original coordinates."""
    if not CV_AVAILABLE:
        return []
    work = img.copy()
    work.thumbnail((DET_MAX_SIDE, DET_MAX_SIDE))
    k = img.width / work.width
    gray = cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    H, W = gray.shape

    kw = max(17, W // 50)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, max(5, kw // 4)))
    bh_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(13, W // 45), max(5, W // 130)))
    maps = [
        cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)),   # vertical edges (any plate colour)
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_k),                  # dark characters on light plate
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, bh_k),                    # light characters on dark plate
    ]

    raw = []
    for m in maps:
        m = cv2.GaussianBlur(m, (5, 5), 0)
        _, th = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k)
        th = cv2.dilate(cv2.erode(th, None, iterations=2), None, iterations=2)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 28 or h < 10:
                continue
            if not (1.0 <= w / h <= 7.5):          # one-row plates ~2-5:1, two-row ~1-2:1
                continue
            if w * h > 0.25 * W * H:
                continue
            px, py = int(w * 0.04), int(h * 0.08)
            x0 = max(0, int((x - px) * k)); y0 = max(0, int((y - py) * k))
            x1 = min(img.width, int((x + w + px) * k)); y1 = min(img.height, int((y + h + py) * k))
            if x1 - x0 > 0 and y1 - y0 > 0:
                raw.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})

    # score each candidate by how much it looks like a row of characters
    cands = []
    full_gray = np.asarray(img.convert("L"))
    for r in raw:
        crop = full_gray[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]
        count, cons = char_blobs(crop)
        r["blobs"], r["cons"] = count, cons
        r["plate_like"] = 5 <= count <= 13 and cons >= 0.75
        r["score"] = (count * cons if r["plate_like"] else 0.0) + 0.001 * min(r["w"] * r["h"], 1e5) / 1e5
        cands.append(r)

    cands.sort(key=lambda c: c["score"], reverse=True)
    kept = []
    for c in cands:                                 # drop near-duplicates
        if all(overlap_ratio(c, o) < 0.7 for o in kept):
            kept.append(c)
    return kept


def read_plate(gray):
    """OCR one cropped candidate with several preprocessing variants. Returns plate text or None."""
    h, w = gray.shape[:2]
    if h < 8 or w < 16:
        return None
    s = min(max(110.0 / h, 0.4), 5.0)
    g = cv2.resize(gray, (max(16, int(w * s)), max(8, int(h * s))),
                   interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu.mean() < 127:                           # white text on dark plate -> make text black
        otsu = 255 - otsu
    variants = [cv2.copyMakeBorder(v, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                for v in (otsu, clahe)]
    psms = (6, 7) if w / h < 2.0 else (7, 6)        # squarer crops are likely two-row plates
    for v in variants:
        for psm in psms:
            try:
                txt = pytesseract.image_to_string(v, config=f"--psm {psm} {PLATE_WHITELIST}", timeout=4)
            except Exception:
                continue
            m = match_plate(txt)
            if m:
                return m
    return None


def find_plates(img, tokens):
    """Return plate boxes: kind='plate' (text verified) or 'plate_candidate' (shape only)."""
    plates = []
    for text, box in plates_from_tokens(tokens):
        plates.append({"kind": "plate", "text": text, "conf": 90, **box})

    start = time.monotonic()
    unverified = []
    full_gray = np.asarray(img.convert("L")) if CV_AVAILABLE else None
    for c in plate_candidates(img)[:MAX_CANDIDATES]:
        if any(overlap_ratio(c, p) > 0.5 for p in plates):
            continue
        verified = None
        if OCR_AVAILABLE and time.monotonic() - start < PLATE_TIME_BUDGET:
            crop = full_gray[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
            verified = read_plate(crop)
        box = {"x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"]}
        if verified:
            plates.append({"kind": "plate", "text": verified, "conf": 85, **box})
        elif c["plate_like"]:
            unverified.append({"kind": "plate_candidate", "text": "", "conf": 50, **box})

    if BLUR_UNVERIFIED:
        for u in unverified[:3]:
            if all(overlap_ratio(u, p) < 0.5 for p in plates):
                plates.append(u)
    return plates


# ---------------------------------------------------------------------------
# Scoring, scanning, cleaning
# ---------------------------------------------------------------------------
def meaningful_words(boxes):
    """Confident words with at least 3 letters. Filters out noise like '2h' or '37,,'."""
    return sorted({
        b["text"] for b in boxes
        if b.get("kind") == "text" and b["conf"] >= OCR_WORD_CONF
        and len(LETTERS.findall(b["text"])) >= 3
    })


def score(has_gps, boxes):
    """Simple, explainable risk score: 0 to 100, with reasons."""
    points, reasons = 0, []
    if has_gps:
        points += 50
        reasons.append("Photo contains GPS coordinates in its metadata")
    if any(b.get("kind") == "plate" for b in boxes):
        points += 40
        reasons.append("A vehicle number plate is readable in the image")
    elif any(b.get("kind") == "plate_candidate" for b in boxes):
        points += 25
        reasons.append("A possible vehicle number plate was detected")
    words = meaningful_words(boxes)
    if words:
        points += min(30, 10 * len(words))
        reasons.append(f"Other readable text in the image: {', '.join(words[:8])}")
    return min(points, 100), reasons


def scan_image(img):
    has_gps = find_gps(img)
    tokens = read_tokens(img)
    plates = find_plates(img, tokens)
    # text boxes already covered by a plate box are redundant
    text_boxes = [b for b in filter_text_boxes(tokens)
                  if not any(overlap_ratio(b, p) > 0.5 for p in plates)]
    boxes = plates + text_boxes
    risk, reasons = score(has_gps, boxes)
    return {
        "risk": risk,
        "reasons": reasons,
        "has_gps": has_gps,
        "text_boxes": boxes,            # plates first; each box has a "kind"
        "plate_count": len(plates),
        "ocr_enabled": OCR_AVAILABLE,
        "cv_enabled": CV_AVAILABLE,
    }


def obscure(img, x, y, w, h):
    """Pad the box, pixelate it into ~4 blocks tall, then blur. Not reversible."""
    px, py = int(w * 0.06) + 2, int(h * 0.15) + 2
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(img.width, x + w + px), min(img.height, y + h + py)
    if x1 <= x0 or y1 <= y0:
        return
    region = img.crop((x0, y0, x1, y1))
    rw, rh = region.size
    block = max(6, rh // 4)
    small = region.resize((max(1, rw // block), max(1, rh // block)), Image.BOX)
    pix = small.resize((rw, rh), Image.NEAREST).filter(
        ImageFilter.GaussianBlur(radius=max(2, block // 3)))
    img.paste(pix, (x0, y0))


def sanitize_boxes(raw, size):
    """Client-supplied boxes are untrusted: coerce to ints and clamp to the image."""
    W, H = size
    out = []
    if not isinstance(raw, list):
        return out
    for b in raw[:500]:
        try:
            x, y, w, h = (int(float(b[k])) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            continue
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            out.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return out


def clean_image(img, boxes):
    """Obscure the given boxes and drop all metadata by building a fresh image."""
    clean = img.copy()
    for b in boxes:
        obscure(clean, b["x"], b["y"], b["w"], b["h"])
    # Rebuild from raw pixels so no EXIF, GPS, or other metadata carries over
    return Image.frombytes("RGB", clean.size, clean.tobytes())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.errorhandler(UnidentifiedImageError)
def bad_image(_):
    return jsonify({"error": "unsupported or corrupt image"}), 400


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
    if "boxes" in request.form:
        try:
            raw = json.loads(request.form["boxes"])
        except json.JSONDecodeError:
            return jsonify({"error": "invalid boxes"}), 400
        boxes = sanitize_boxes(raw, img.size)
    else:
        # No boxes sent: detect and blur everything on the server
        boxes = sanitize_boxes(scan_image(img)["text_boxes"], img.size)
    cleaned = clean_image(img, boxes)
    buf = io.BytesIO()
    cleaned.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg", download_name="veil_clean.jpg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)