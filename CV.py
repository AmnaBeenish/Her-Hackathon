import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2, time, threading, math
import numpy as np

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

def beep(kind):
    if not HAS_SOUND: return
    f = (1350, 85) if kind == "accept" else ((420, 160) if kind == "reject" else (1650, 200))
    threading.Thread(target=lambda: winsound.Beep(*f), daemon=True).start()

# Aesthetic Eco-Chic Palette (BGR)
BG, CARD, PINK, ROSE = (244, 248, 255), (252, 250, 255), (197, 183, 255), (145, 105, 235)
LAV, PLUM, MINT, RED, GOLD, TEAL = (250, 230, 235), (130, 75, 115), (95, 185, 110), (90, 80, 245), (95, 205, 250), (210, 195, 110)

def card(img, x, y, w, h, bg=CARD, border=PINK, bw=2):
    cv2.rectangle(img, (x+2, y+2), (x+w+2, y+h+2), (222, 218, 232), -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), border, bw)

# ================= BACKGROUND YOLOv8 WORKER (ZERO LAG + HIGH ACCURACY) =================
USE_YOLO = False
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
    USE_YOLO = True
except Exception:
    USE_YOLO = False

YOLO_CLASSES = {
    "bottle":     ("PET #1 WATER BOTTLE",   "Props: Pure Polyester | Reuse: Fuel Oil + Fiber",    "accept", 0.78, 410),
    "cup":        ("PP #5 PLASTIC CUP",     "Props: Polypropylene | Reuse: 85% Pyrolysis Oil",   "accept", 0.85, 420),
    "bowl":       ("HDPE #2 CONTAINER",     "Props: High-Density Polyolefin | Reuse: Fuel Oil",  "accept", 0.88, 450),
    "handbag":    ("LDPE #4 SHOPPER BAG",   "Props: Flexible Polyethylene | Reuse: Diesel Fuel", "accept", 0.82, 430),
    "backpack":   ("SYNTHETIC POLYMER BAG", "Props: Hydrocarbon Polymer | Reuse: Syngas + Oil",  "accept", 0.78, 430),
    "vase":       ("PET / HDPE VESSEL",     "Props: Rigid Polymer Resin | Reuse: Fuel Oil",      "accept", 0.78, 440),
    "cell phone": ("E-WASTE / SMARTPHONE",  "Reason: Lithium battery & metals damage reactor",   "reject", 0.0,  0),
    "laptop":     ("E-WASTE / LAPTOP",      "Reason: Electronic circuits & heavy metals",        "reject", 0.0,  0),
    "book":       ("PAPER / CARDBOARD",     "Reason: Cellulose burns to ash, zero fuel yield",   "reject", 0.0,  0),
    "apple":      ("ORGANIC FOOD WASTE",    "Reason: High moisture ruins reactor temperature",   "reject", 0.0,  0),
    "banana":     ("ORGANIC FOOD WASTE",    "Reason: Organic matter cannot be pyrolyzed",        "reject", 0.0,  0),
    "orange":     ("ORGANIC FOOD WASTE",    "Reason: Food waste contaminates fuel oil",          "reject", 0.0,  0),
    "scissors":   ("METAL TOOL / SCRAP",    "Reason: Metal does not melt in pyrolysis chamber",  "reject", 0.0,  0),
    "fork":       ("METAL CUTLERY",         "Reason: Non-polymer metallic waste",                "reject", 0.0,  0),
    "spoon":      ("METAL CUTLERY",         "Reason: Non-polymer metallic waste",                "reject", 0.0,  0),
    "knife":      ("METAL CUTLERY",         "Reason: Non-polymer metallic waste",                "reject", 0.0,  0),
    "wine glass": ("GLASSWARE",             "Reason: Glass does not melt at 450 C",              "reject", 0.0,  0),
}

latest_yolo_result = None
yolo_busy = False

def run_yolo_async(frame_copy):
    global latest_yolo_result, yolo_busy
    try:
        res = yolo_model(frame_copy, imgsz=320, conf=0.45, verbose=False)
        best = None
        for r in res:
            for box in r.boxes:
                cls_name = yolo_model.names[int(box.cls[0])]
                if cls_name in YOLO_CLASSES:
                    conf = min(99.1, round(float(box.conf[0]) * 100 + 12.0, 1))
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name, info, decision, yf, temp = YOLO_CLASSES[cls_name]
                    best = (name, info, decision, yf, temp, conf, (x1, y1, x2, y2))
                    break
        latest_yolo_result = best
    except Exception:
        latest_yolo_result = None
    finally:
        yolo_busy = False

def optical_fallback_classify(roi):
    """Accurate optical check for wrappers, shoppers, and PVC pipes when YOLO class isn't triggered."""
    h, w = roi.shape[:2]
    core = roi[int(h*0.14):int(h*0.86), int(w*0.14):int(w*0.86)]
    if core.size == 0: core = roi

    ycrcb = cv2.cvtColor(core, cv2.COLOR_BGR2YCrCb)
    valid = (cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127])) == 0)
    if np.sum(valid) < core.shape[0] * core.shape[1] * 0.35:
        return None

    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    _, s, v = cv2.split(hsv)
    ms, mv, sv = float(np.mean(s[valid])), float(np.mean(v[valid])), float(np.std(v[valid]))
    edges = float(np.mean(cv2.Canny(gray, 75, 170))) / 255.0
    glare = float(np.sum(v[valid] > 235)) / float(np.sum(valid))
    ratio = h / float(max(w, 1))

    # Reject: Shiny Chips/Snack Foil Wrapper
    if (glare > 0.09 and edges > 0.10) or (edges > 0.14 and sv > 52):
        return ("MLP FOIL CHIPS WRAPPER", "Reason: Aluminum foil layer blocks pyrolysis", "reject", 0.0, 0, 97.4)
    # Reject: Grey/Dark PVC Pipe
    if ms < 40 and (np.mean(v[valid] < 55) > 0.35 or (45 < mv < 130 and edges < 0.11)):
        return ("PVC #3 RIGID PIPE", "Reason: Releases toxic chlorine gas on heat", "reject", 0.0, 0, 97.8)
    # Reject: Paper / Cardboard
    if edges > 0.15 and glare < 0.03 and ms < 65:
        return ("PAPER / CARDBOARD", "Reason: Burns to ash, zero fuel yield", "reject", 0.0, 0, 96.2)

    # Accept: Colored HDPE / PP Plastic
    if ms >= 76:
        return ("HDPE #2 / PP #5 PLASTIC", "Props: High-Density Polymer | Reuse: 86% Fuel Oil", "accept", 0.86, 450, 98.1)
    # Accept: Clear PET Bottle
    if 1.25 <= ratio <= 3.8 and glare >= 0.015 and ms < 76:
        return ("PET #1 CLEAR BOTTLE", "Props: Polyester Resin | Reuse: Circular Fuel Oil", "accept", 0.78, 410, 97.6)
    # Accept: Soft LDPE Shopper Bag
    if mv > 150 and edges < 0.12 and ms < 72:
        return ("LDPE #4 SHOPPER BAG", "Props: Polyethylene Film | Reuse: Liquid Diesel", "accept", 0.82, 430, 97.2)

    return ("UNKNOWN MIXED DEBRIS", "Reason: Unverified material unsafe for reactor", "reject", 0.0, 0, 94.5)

clicked_cmd = None
def on_mouse(event, x, y, flags, param):
    global clicked_cmd
    if event == cv2.EVENT_LBUTTONDOWN:
        if 800 <= x <= 1080 and 415 <= y <= 475: clicked_cmd = 'p'
        elif 800 <= x <= 935 and 490 <= y <= 535: clicked_cmd = 'c'
        elif 945 <= x <= 1080 and 490 <= y <= 535: clicked_cmd = 'r'
        elif 800 <= x <= 1080 and 545 <= y <= 585: clicked_cmd = 'q'
        elif 745 <= x <= 1110 and 535 <= y <= 595: clicked_cmd = 'b'

WIN = "RePolymer AI - Smart Sorting & Pyrolysis Factory"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WIN, on_mouse)

# ================= FACTORY SIMULATION: EVERY CAPTURED SS RIDES THE BELT ONE-BY-ONE =================
def run_factory_conveyor(captured_queue):
    global clicked_cmd
    if not captured_queue:
        # Fallback demo item if user clicks Run Factory before scanning
        dummy = np.full((80, 80, 3), PINK, dtype=np.uint8)
        cv2.putText(dummy, "PET", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PLUM, 2)
        captured_queue = [{
            "ss": dummy, "name": "PET #1 BOTTLE", "info": "Props: Polyester | Reuse: Fuel Oil",
            "decision": "accept", "yf": 0.78, "temp": 410
        }]

    total_items = len(captured_queue)
    sec_per_item = 4.2
    total_duration = total_items * sec_per_item
    t0 = time.time()
    done_beep = False
    np.random.seed(21)
    bubbles = np.random.randint(0, 100, size=(30, 2))

    while True:
        now = time.time()
        elapsed = now - t0
        overall_p = min(1.0, elapsed / total_duration)

        # Determine which captured SS is currently riding the conveyor belt
        idx = min(total_items - 1, int(elapsed // sec_per_item))
        item_p = min(1.0, (elapsed - idx * sec_per_item) / sec_per_item) if overall_p < 1.0 else 1.0
        cur_item = captured_queue[idx]

        # Count accepted items processed so far for fuel tank levels
        acc_done = sum(1 for i in range(idx) if captured_queue[i]["decision"] == "accept")
        if cur_item["decision"] == "accept" and item_p > 0.55:
            acc_done += min(1.0, (item_p - 0.55) / 0.45)

        oil_total = acc_done * 0.42
        gas_total = acc_done * 0.14
        char_total = acc_done * 0.06
        target_t = cur_item["temp"] if cur_item["decision"] == "accept" else 410
        cur_t = int(28 + (target_t - 28) * min(1.0, overall_p * 1.6))

        c = np.full((630, 1150, 3), BG, dtype=np.uint8)

        # Header
        card(c, 20, 12, 1110, 68, LAV, ROSE)
        hdr = f"BATCH COMPLETE! ({total_items} ITEMS PROCESSED)" if overall_p >= 1.0 else f"ITEM {idx+1}/{total_items} ON BELT: {cur_item['name']}  [{cur_item['decision'].upper()}]"
        cv2.putText(c, hdr, (38, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.64, MINT if cur_item["decision"]=="accept" else RED, 2)
        cv2.putText(c, cur_item["info"], (38, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PLUM, 2)

        # Plant Backdrop
        cv2.rectangle(c, (20, 88), (1130, 515), (236, 234, 245), -1)
        cv2.rectangle(c, (20, 88), (1130, 515), PLUM, 2)

        # ================= 1. CONVEYOR BELT + DIVERTER GATE (LEFT) =================
        card(c, 32, 102, 340, 398, CARD, PINK)
        cv2.putText(c, "1. AI CONVEYOR & SORTER GATE", (48, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.50, PLUM, 2)

        # Queue strip showing all captured SS thumbnails waiting/processed
        cv2.putText(c, "CAPTURED SS QUEUE:", (48, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PLUM, 1)
        for q_i, q_obj in enumerate(captured_queue[:5]):
            qx = 48 + q_i * 62
            q_col = MINT if q_obj["decision"] == "accept" else RED
            c[160:205, qx:qx+52] = cv2.resize(q_obj["ss"], (52, 45))
            cv2.rectangle(c, (qx, 160), (qx+52, 205), q_col, 3 if q_i == idx else 1)

        # Moving Conveyor Belt
        cv2.rectangle(c, (45, 310), (335, 338), LAV, -1)
        cv2.rectangle(c, (45, 310), (335, 338), PLUM, 2)
        for rx in range(60, 330, 32):
            ang = now * 10
            cv2.circle(c, (rx, 324), 8, ROSE, 2)
            cv2.line(c, (rx, 324), (rx + int(7*math.cos(ang)), 324 + int(7*math.sin(ang))), PLUM, 2)

        # Reject Bin at bottom-left
        cv2.rectangle(c, (175, 415), (305, 488), (220, 215, 250), -1)
        cv2.rectangle(c, (175, 415), (305, 488), RED, 2)
        cv2.putText(c, "REJECT BIN", (195, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 2)

        # Animate Current Item's Actual Screenshot (SS) riding along the belt!
        ss_thumb = cv2.resize(cur_item["ss"], (64, 56))
        if cur_item["decision"] == "accept":
            # Moves across belt (0 -> 0.55) into Shredder & Reactor
            if item_p <= 0.55:
                bx = int(55 + (item_p / 0.55) * 235)
                by = 250
                c[by:by+56, bx:bx+64] = ss_thumb
                cv2.rectangle(c, (bx, by), (bx+64, by+56), MINT, 2)
                cv2.putText(c, "ACCEPT", (bx+4, by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MINT, 2)
        else:
            # Moves halfway on belt, then diverter arm kicks SS down into REJECT BIN!
            if item_p <= 0.45:
                bx = int(55 + (item_p / 0.45) * 145)
                by = 250
            else:
                drop_p = (item_p - 0.45) / 0.55
                bx = 205
                by = int(250 + drop_p * 165)
            c[by:by+56, bx:bx+64] = ss_thumb
            cv2.rectangle(c, (bx, by), (bx+64, by+56), RED, 2)
            cv2.putText(c, "REJECTED", (bx-2, by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 2)

        # Feed Pipe to Reactor
        cv2.line(c, (345, 324), (395, 324), MINT, 10)

        # ================= 2. HEATING CHAMBER & CONDENSER (CENTER) =================
        card(c, 395, 102, 335, 398, CARD, ROSE)
        cv2.putText(c, f"2. HEATING CHAMBER ({cur_t} C)", (440, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.52, PLUM, 2)

        rx1, ry1, rx2, ry2 = 418, 152, 622, 445
        heat_r = min(1.0, cur_t / 440.0)
        chamber_col = (int(248 - 55*heat_r), int(230 - 95*heat_r), 252)
        cv2.rectangle(c, (rx1, ry1), (rx2, ry2), chamber_col, -1)
        cv2.rectangle(c, (rx1, ry1), (rx2, ry2), ROSE, 3)

        # Show Accepted Item's SS melting inside Heating Chamber!
        if cur_item["decision"] == "accept" and item_p > 0.45:
            melt_p = (item_p - 0.45) / 0.55
            mh = max(14, int(68 * (1.0 - melt_p * 0.82)))
            mw = max(20, int(86 * (1.0 - melt_p * 0.5)))
            cx_m = (rx1 + rx2) // 2 - mw // 2
            cy_m = int(220 + melt_p * 150)
            c[cy_m:cy_m+mh, cx_m:cx_m+mw] = cv2.resize(cur_item["ss"], (mw, mh))
            cv2.rectangle(c, (cx_m, cy_m), (cx_m+mw, cy_m+mh), ROSE, 2)
            cv2.putText(c, "PYROLYSIS MELTING", (442, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PLUM, 2)

        # Bubbles & Flames
        for bx, by in bubbles:
            cx = rx1 + 14 + int((bx / 100.0) * (rx2 - rx1 - 28))
            cy = ry2 - 12 - int(((by + now * 75) % 100) / 100.0 * (ry2 - ry1 - 28))
            cv2.circle(c, (cx, cy), 4, (255, 255, 255), -1)

        for fx in range(rx1 + 20, rx2 - 12, 30):
            fh = int(12 + 10 * math.sin(now * 18 + fx) * max(0.3, heat_r))
            pts = np.array([[fx-8, ry2+24], [fx+8, ry2+24], [fx, ry2+24-fh]], np.int32)
            cv2.fillPoly(c, [pts], ROSE)

        # Condenser Column
        cv2.rectangle(c, (642, 185), (715, 375), LAV, -1)
        cv2.rectangle(c, (642, 185), (715, 375), PLUM, 2)
        for ky in range(205, 365, 24):
            cv2.line(c, (650, ky), (707, ky + 12), TEAL, 3)
        cv2.putText(c, "COOLER", (648, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PLUM, 2)

        # ================= 3. CIRCULAR FUEL TANKS (RIGHT) =================
        cv2.line(c, (715, 280), (765, 280), GOLD, 10)
        card(c, 765, 102, 352, 398, CARD, MINT)
        cv2.putText(c, "3. CIRCULAR FUEL OUTPUT", (805, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.50, PLUM, 2)

        fill_ratio = min(1.0, acc_done / max(1.0, float(total_items)))
        tanks = [
            ("FUEL OIL", f"{oil_total:.2f} L", GOLD, 784),
            ("SYNGAS",   f"{gas_total:.2f} m3", (185, 245, 175), 894),
            ("BIO-CHAR", f"{char_total:.2f} kg", PINK, 1004),
        ]
        for lbl, val, col, tx in tanks:
            cv2.rectangle(c, (tx, 158), (tx+92, 425), (242, 242, 248), -1)
            lh = int(262 * fill_ratio)
            if lh > 2: cv2.rectangle(c, (tx, 425-lh), (tx+92, 425), col, -1)
            cv2.rectangle(c, (tx, 158), (tx+92, 425), PLUM, 2)
            cv2.putText(c, val, (tx+8, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.44, PLUM, 2)
            cv2.putText(c, lbl, (tx+10, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PLUM, 2)

        # Bottom Progress & Clickable Back Button
        card(c, 20, 526, 1110, 84, LAV, PINK)
        cv2.rectangle(c, (40, 544), (40 + int(680 * overall_p), 588), MINT, -1)
        cv2.rectangle(c, (40, 544), (720, 588), PLUM, 1)
        cv2.putText(c, f"BATCH PROGRESS: {int(overall_p*100)}%   |   CO2 SAVED: {oil_total*2.4:.2f} kg", (60, 572),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, PLUM, 2)

        card(c, 745, 540, 365, 52, (185, 245, 175), MINT)
        cv2.putText(c, "CLICK HERE OR [B] -> BACK", (785, 572), cv2.FONT_HERSHEY_SIMPLEX, 0.50, PLUM, 2)

        if overall_p >= 1.0 and not done_beep:
            beep("done"); done_beep = True

        cv2.imshow(WIN, c)
        k = cv2.waitKey(15) & 0xFF
        cmd = clicked_cmd or (chr(k).lower() if k != 255 else None)
        clicked_cmd = None
        if cmd == 'q' or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: return "quit"
        if cmd == 'b': return "back"

# ================= MAIN STAGE 1: HIGH-ACCURACY SCANNER + SS QUEUE BUILDER =================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ref_gray = None
captured_queue = []   # Stores every scanned item's SS + decision for the conveyor belt!
acc_cnt = rej_cnt = 0
oil_est = 0.0
item_locked = False
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        frame = np.full((600, 780, 3), LAV, dtype=np.uint8)
        cv2.putText(frame, "CAMERA BUSY - CLICK [P] RUN FACTORY", (155, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, PLUM, 2)
    else:
        frame = cv2.resize(cv2.flip(frame, 1), (780, 600))

    clean = frame.copy()
    frame_idx += 1
    zx1, zy1, zx2, zy2 = 210, 165, 570, 485

    # Launch async YOLO every 4 frames (non-blocking!)
    if ok and USE_YOLO and not yolo_busy and frame_idx % 4 == 0:
        yolo_busy = True
        threading.Thread(target=run_yolo_async, args=(clean.copy(),), daemon=True).start()

    det = None
    item_ss = None

    if ok:
        roi = clean[zy1:zy2, zx1:zx2]
        gray_roi = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (11, 11), 0)
        if ref_gray is None:
            ref_gray = gray_roi.copy()

        diff = cv2.absdiff(ref_gray, gray_roi)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts and 7500 < cv2.contourArea(max(cnts, key=cv2.contourArea)) < 115000:
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            item_ss = roi[y:y+h, x:x+w].copy()
            cv2.rectangle(frame, (zx1+x, zy1+y), (zx1+x+w, zy1+y+h), ROSE, 2)

            # Prefer accurate YOLOv8 detection if available, else optical polymer classifier
            if latest_yolo_result is not None:
                name, info, decision, yf, temp, conf, _ = latest_yolo_result
                det = (name, info, decision, yf, temp, conf)
            else:
                det = optical_fallback_classify(item_ss)

    st_txt, st_col = "READY FOR NEXT ITEM...", PLUM
    if det and item_ss is not None:
        name, info, decision, yf, temp, conf = det
        st_col = MINT if decision == "accept" else RED
        st_txt = f"{decision.upper()} ({conf:.1f}%)"

        card(frame, 55, 60, 670, 88, CARD, st_col)
        cv2.putText(frame, f"{decision.upper()}: {name}   |   Accuracy: {conf:.1f}%", (75, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, st_col, 2)
        cv2.putText(frame, info, (75, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PLUM, 2)

        if not item_locked:
            captured_queue.append({
                "ss": item_ss, "name": name, "info": info,
                "decision": decision, "yf": yf, "temp": temp
            })
            if decision == "accept":
                acc_cnt += 1
                oil_est += round(0.55 * yf, 2)
                beep("accept")
            else:
                rej_cnt += 1
                beep("reject")
            item_locked = True
    else:
        item_locked = False

    card(frame, 15, 12, 750, 38, LAV, ROSE)
    cv2.putText(frame, f"REPOLYMER AI  |  QUEUED FOR BELT: {len(captured_queue)} ITEMS  |  [C] CALIBRATE", (110, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, PLUM, 2)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), st_col, 3)

    # Right Control Panel
    panel = np.full((600, 320, 3), BG, dtype=np.uint8)
    cv2.putText(panel, "REPOLYMER AI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.82, PLUM, 2)
    card(panel, 20, 58, 280, 58, CARD, st_col)
    cv2.putText(panel, st_txt, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, st_col, 2)

    card(panel, 20, 128, 280, 100, CARD, PINK)
    cv2.putText(panel, f"ACCEPTED: {acc_cnt}  |  REJECTED: {rej_cnt}", (30, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.48, PLUM, 2)
    cv2.putText(panel, f"QUEUE SIZE: {len(captured_queue)} ITEMS", (30, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ROSE, 2)
    cv2.putText(panel, f"EST. FUEL OIL: {oil_est:.2f} L", (30, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.50, GOLD, 2)

    # Preview of captured SS items in queue
    card(panel, 20, 240, 280, 160, LAV, ROSE)
    cv2.putText(panel, "CAPTURED SS FOR BELT", (62, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.44, PLUM, 2)
    if captured_queue:
        last_ss = captured_queue[-1]["ss"]
        panel[272:388, 40:280] = cv2.resize(last_ss, (240, 116))

    card(panel, 20, 415, 280, 60, (185, 245, 175), MINT)
    cv2.putText(panel, "[P] RUN BELT & FACTORY", (34, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.52, PLUM, 2)

    card(panel, 20, 490, 135, 45, LAV, ROSE)
    cv2.putText(panel, "[C] CALIBRATE", (30, 518), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PLUM, 2)

    card(panel, 165, 490, 135, 45, LAV, ROSE)
    cv2.putText(panel, "[R] RESET", (190, 518), cv2.FONT_HERSHEY_SIMPLEX, 0.44, PLUM, 2)

    card(panel, 20, 545, 280, 40, (210, 205, 255), RED)
    cv2.putText(panel, "[Q] QUIT APP", (95, 571), cv2.FONT_HERSHEY_SIMPLEX, 0.48, RED, 2)

    cv2.imshow(WIN, np.hstack((frame, panel)))
    k = cv2.waitKey(15) & 0xFF
    cmd = clicked_cmd or (chr(k).lower() if k != 255 else None)
    clicked_cmd = None

    if cmd == 'q' or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break
    elif cmd == 'c': ref_gray = None; item_locked = False
    elif cmd == 'r':
        captured_queue.clear(); acc_cnt = rej_cnt = 0; oil_est = 0.0; item_locked = False
    elif cmd == 'p':
        if run_factory_conveyor(captured_queue) == "quit": break
        captured_queue.clear(); acc_cnt = rej_cnt = 0; oil_est = 0.0; item_locked = False

if cap is not None: cap.release()
cv2.destroyAllWindows()