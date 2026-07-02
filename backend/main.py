from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import uuid
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import re
import os
from datetime import datetime
from typing import Optional
import easyocr
import time
import json
from dotenv import load_dotenv

load_dotenv()

# ─── OCR ENGINES ─────────────────────────────────────

print("Loading EasyOCR reader...")
easyocr_reader = easyocr.Reader(['en'], verbose=False)
print("✅ EasyOCR loaded")


try:
    from ultralytics import YOLO
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    YOLO_AVAILABLE = True
    print("✅ YOLO loaded")
except Exception as e:
    yolo_model = None
    YOLO_AVAILABLE = False
    print(f"⚠️  YOLO skipped: {e}")

app = FastAPI(title="PMSSS Scholarship System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MySQL Connection ────────────────────────────────────────────────────────

def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "pmsss_db")
    )

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INT AUTO_INCREMENT PRIMARY KEY,
            application_id VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(100),
            education_level VARCHAR(50),
            aadhar_no VARCHAR(20),
            state VARCHAR(50),
            pin_code VARCHAR(10),
            father_name VARCHAR(100),
            email VARCHAR(100),
            phone VARCHAR(15),
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            cv_aadhar_match TINYINT(1) DEFAULT NULL,
            cv_marks FLOAT DEFAULT NULL,
            cv_income FLOAT DEFAULT NULL,
            cv_processed TINYINT(1) DEFAULT 0,
            eligibility_status VARCHAR(20) DEFAULT 'pending',
            rejection_reasons TEXT DEFAULT NULL,
            ticket_raised TINYINT(1) DEFAULT 0,
            admin_status VARCHAR(20) DEFAULT 'pending',
            admin_notes TEXT DEFAULT NULL,
            admin_reviewed_at DATETIME DEFAULT NULL
        )
    """)
    db.commit()
    cursor.close()
    db.close()

try:
    init_db()
    print("✅ Database initialized")
except Exception as e:
    print(f"⚠️  DB init failed (run manually): {e}")

# ─── Computer Vision Utilities ───────────────────────────────────────────────

def detect_document_with_yolo(img):
    """Detect document region using YOLO — skip if YOLO unavailable"""
    if not YOLO_AVAILABLE:
        return img, 0.0
    try:
        results = yolo_model(img, verbose=False)
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best_box = boxes[0]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                # Only crop if the detected region is large enough
                h, w = img.shape[:2]
                crop_h = y2 - y1
                crop_w = x2 - x1
                if crop_h > h * 0.3 and crop_w > w * 0.3:
                    cropped = img[y1:y2, x1:x2]
                    confidence = float(best_box.conf[0])
                    return cropped, confidence
        return img, 0.0
    except Exception as e:
        print("YOLO Detection Error:", e)
        return img, 0.0

def preprocess_image(img_array):
    """Multiple preprocessing variants for better OCR"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # Resize if too small
    h, w = gray.shape
    if w < 1000:
        scale = 1000 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return gray

def preprocess_variants(img_array):
    """Return multiple preprocessing variants for better OCR coverage"""
    gray = preprocess_image(img_array)
    variants = []

    # 1. Basic grayscale
    variants.append(gray)

    # 2. Otsu threshold
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(thresh_otsu)

    # 3. Adaptive threshold
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)

    # 4. Sharpened
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(sharpened)

    return variants

def run_tesseract_ocr(img):
    """Run tesseract with multiple configs and return best result"""
    start = time.time()
    best_text = ""
    try:
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 3',
            r'--oem 3 --psm 4',
            r'--oem 1 --psm 6',
        ]
        for variant in preprocess_variants(img):
            for config in configs:
                try:
                    text = pytesseract.image_to_string(variant, config=config, lang='eng')
                    if len(text) > len(best_text):
                        best_text = text
                except:
                    pass
        return {
            "engine": "tesseract",
            "text": best_text,
            "confidence": min(len(best_text) / 100, 1.0) * 100,
            "time": round(time.time() - start, 2),
            "success": True
        }
    except Exception as e:
        return {
            "engine": "tesseract",
            "text": best_text,
            "confidence": 0,
            "success": len(best_text) > 0,
            "error": str(e)
        }

def run_easyocr(img):
    """Run EasyOCR"""
    start = time.time()
    try:
        results = easyocr_reader.readtext(img)
        texts = []
        confidences = []
        for r in results:
            texts.append(r[1])
            confidences.append(r[2])
        avg_conf = np.mean(confidences) if confidences else 0
        return {
            "engine": "easyocr",
            "text": " ".join(texts),
            "confidence": round(avg_conf * 100, 2),
            "time": round(time.time() - start, 2),
            "success": True
        }
    except Exception as e:
        return {
            "engine": "easyocr",
            "text": "",
            "confidence": 0,
            "success": False,
            "error": str(e)
        }

def extract_text_from_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO Detection — use full image if YOLO crop is bad
    detected_img, yolo_conf = detect_document_with_yolo(img)

    ocr_results = []

    # Tesseract on both detected and original
    for src_img in [detected_img, img]:
        t_result = run_tesseract_ocr(src_img)
        t_result["yolo_confidence"] = yolo_conf
        ocr_results.append(t_result)

    # EasyOCR on detected image
    easy_result = run_easyocr(detected_img)
    easy_result["yolo_confidence"] = yolo_conf
    ocr_results.append(easy_result)

    # Combine ALL text for extraction (don't just pick "best")
    all_text = "\n".join([r["text"] for r in ocr_results if r.get("success") and r.get("text")])

    # Also pick the longest individual result
    best_result = max(
        [r for r in ocr_results if r.get("success")],
        key=lambda r: len(r.get("text", "")),
        default=ocr_results[0] if ocr_results else {"text": "", "engine": "none"}
    )

    return {
        "best_text": best_result.get("text", ""),
        "all_text": all_text,
        "best_engine": best_result.get("engine", "none"),
        "ocr_results": ocr_results
    }

# ─── Extraction Functions ────────────────────────────────────────────────────

def extract_aadhar_from_text(text: str) -> Optional[str]:
    """Extract 12-digit Aadhaar number — very permissive patterns"""
    # Normalize: remove extra spaces, common OCR mistakes
    text = text.replace('O', '0').replace('l', '1').replace('I', '1')

    patterns = [
        r'\b(\d{4}[\s\-]\d{4}[\s\-]\d{4})\b',   # XXXX XXXX XXXX
        r'\b(\d{12})\b',                            # 12 digits together
        r'\b(\d{4})\s+(\d{4})\s+(\d{4})\b',       # with spaces
        r'(\d{4})[^\d]{0,3}(\d{4})[^\d]{0,3}(\d{4})',  # loose
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if isinstance(m, tuple):
                num = ''.join(m)
            else:
                num = m
            num = re.sub(r'[\s\-]', '', num)
            if len(num) == 12 and num.isdigit():
                # Basic Aadhaar validity: first digit not 0 or 1
                if num[0] not in ('0', '1'):
                    return num
    return None

def extract_marks_percentage(text: str) -> Optional[float]:
    """Extract percentage marks — broad patterns for real marksheets"""
    patterns = [
        # Explicit percentage mentions
        r'percentage[\s\:\-]+(\d{1,3}(?:\.\d{1,2})?)\s*%?',
        r'total\s+percentage[\s\:\-]+(\d{1,3}(?:\.\d{1,2})?)',
        r'grand\s+total[\s\:\-]+.*?(\d{1,3}\.\d{1,2})\s*%',
        r'aggregate[\s\:\-]+(\d{1,3}(?:\.\d{1,2})?)\s*%?',
        r'overall[\s\:\-]+(\d{1,3}(?:\.\d{1,2})?)\s*%',
        # XX.XX%
        r'(\d{2,3}\.\d{1,2})\s*%',
        # Just XX% (with percent sign)
        r'\b(\d{2,3})\s*%',
        # Score / total patterns
        r'(\d{3})\s*/\s*(?:300|400|500|600)',
        r'(\d{3})\s+out\s+of\s+(?:300|400|500)',
        # Board result patterns
        r'marks\s+obtained[\s\:\-]+(\d{3,4})',
        r'total\s+marks[\s\:\-]+(\d{2,3}(?:\.\d{1,2})?)',
    ]

    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            try:
                if isinstance(m, tuple):
                    m = m[0]
                val = float(m)
                if 30 < val <= 100:
                    candidates.append(val)
                elif val > 100 and val <= 600:
                    # Convert raw marks to percentage (assume /500 or /600)
                    for total in [500, 400, 600, 300]:
                        pct = (val / total) * 100
                        if 30 < pct <= 100:
                            candidates.append(round(pct, 2))
                            break
            except:
                continue

    if candidates:
        # Return the most likely value (highest reasonable one)
        valid = [v for v in candidates if 30 < v <= 100]
        if valid:
            return max(valid)
    return None

def extract_income_amount(text: str) -> Optional[float]:
    """Extract annual income — broad patterns for income certificates"""
    patterns = [
        r'annual\s+income[\s\:\-]+(?:rs\.?|inr|₹|rupees)?\s*([\d,\s]+)',
        r'total\s+annual\s+income[\s\:\-]+(?:rs\.?|inr|₹)?\s*([\d,\s]+)',
        r'yearly\s+income[\s\:\-]+(?:rs\.?|inr|₹)?\s*([\d,\s]+)',
        r'income[\s\:\-]+(?:rs\.?|inr|₹)?\s*([\d,\s]+)',
        r'(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)',
        r'salary[\s\:\-]+(?:rs\.?|inr|₹)?\s*([\d,\s]+)',
        r'earnings?[\s\:\-]+(?:rs\.?|inr|₹)?\s*([\d,\s]+)',
        # Words like "two lakh" etc.
        r'(\d[\d,]+)\s*(?:lakh|lac)',   # N lakh
    ]

    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            try:
                if isinstance(m, tuple):
                    m = m[0]
                raw = re.sub(r'[,\s]', '', m)
                raw = re.sub(r'[^\d.]', '', raw)
                if not raw:
                    continue
                val = float(raw)
                # Handle "lakh" multiplier
                if re.search(r'lakh|lac', m, re.IGNORECASE):
                    val *= 100000
                if 10000 <= val <= 10000000:  # 10k to 1cr range
                    candidates.append(val)
            except:
                continue

    if candidates:
        # Return the most common or highest value
        return max(candidates)
    return None

def process_documents(aadhar_bytes, income_bytes, marksheet_bytes, entered_aadhar: str):
    """Main CV pipeline: OCR + extract + validate"""
    results = {
        "aadhar_match": False,
        "extracted_aadhar": None,
        "marks": None,
        "income": None,
        "errors": [],
        "ocr_details": {
            "aadhar": None,
            "marksheet": None,
            "income": None
        }
    }

    # Process Aadhaar
    try:
        aadhar_ocr = extract_text_from_image(aadhar_bytes)
        results["ocr_details"]["aadhar"] = aadhar_ocr
        # Try all_text first (combines all OCR results), then best_text
        aadhar_text = aadhar_ocr.get("all_text", "") or aadhar_ocr.get("best_text", "")
        extracted = extract_aadhar_from_text(aadhar_text)
        results["extracted_aadhar"] = extracted
        if extracted:
            clean_entered = re.sub(r'[\s\-]', '', entered_aadhar)
            results["aadhar_match"] = (extracted == clean_entered)
        else:
            results["errors"].append("Could not extract Aadhaar number from uploaded document")
    except Exception as e:
        results["errors"].append(f"Aadhaar processing error: {str(e)}")

    # Process Marksheet
    try:
        marks_ocr = extract_text_from_image(marksheet_bytes)
        results["ocr_details"]["marksheet"] = marks_ocr
        marks_text = marks_ocr.get("all_text", "") or marks_ocr.get("best_text", "")
        results["marks"] = extract_marks_percentage(marks_text)
        if results["marks"] is None:
            results["errors"].append("Could not extract marks percentage from marksheet")
    except Exception as e:
        results["errors"].append(f"Marksheet processing error: {str(e)}")

    # Process Income Certificate
    try:
        income_ocr = extract_text_from_image(income_bytes)
        results["ocr_details"]["income"] = income_ocr
        income_text = income_ocr.get("all_text", "") or income_ocr.get("best_text", "")
        results["income"] = extract_income_amount(income_text)
        if results["income"] is None:
            results["errors"].append("Could not extract income amount from certificate")
    except Exception as e:
        results["errors"].append(f"Income certificate processing error: {str(e)}")

    return results

# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "PMSSS Scholarship API is running", "version": "2.0.0"}

@app.post("/api/apply")
async def submit_application(
    name: str = Form(...),
    education_level: str = Form(...),
    aadhar_no: str = Form(...),
    state: str = Form(...),
    pin_code: str = Form(...),
    father_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    aadhar_doc: UploadFile = File(...),
    income_doc: UploadFile = File(...),
    marksheet_doc: UploadFile = File(...)
):
    app_id = "PMSSS" + datetime.now().strftime("%Y%m%d") + str(uuid.uuid4())[:6].upper()

    aadhar_bytes = await aadhar_doc.read()
    income_bytes = await income_doc.read()
    marks_bytes = await marksheet_doc.read()

    cv_results = process_documents(aadhar_bytes, income_bytes, marks_bytes, aadhar_no)

    rejection_reasons = []
    if not cv_results["aadhar_match"]:
        rejection_reasons.append("Aadhaar number in document does not match entered number")
    if cv_results["marks"] is not None and cv_results["marks"] <= 70:
        rejection_reasons.append(f"Marks ({cv_results['marks']}%) are below required 70%")
    elif cv_results["marks"] is None:
        rejection_reasons.append("Marks percentage could not be verified from marksheet")
    if cv_results["income"] is not None and cv_results["income"] >= 800000:
        rejection_reasons.append(f"Annual income (₹{cv_results['income']:,.0f}) exceeds ₹8,00,000 limit")
    elif cv_results["income"] is None:
        rejection_reasons.append("Annual income could not be verified from income certificate")
    rejection_reasons.extend(cv_results.get("errors", []))

    eligibility = "approved" if len(rejection_reasons) == 0 else "rejected"

    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO applications
            (application_id, name, education_level, aadhar_no, state, pin_code,
             father_name, email, phone, cv_aadhar_match, cv_marks, cv_income,
             cv_processed, eligibility_status, rejection_reasons)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            app_id, name, education_level, aadhar_no, state, pin_code,
            father_name, email, phone,
            int(cv_results["aadhar_match"]),
            cv_results["marks"], cv_results["income"],
            1, eligibility,
            json.dumps(rejection_reasons) if rejection_reasons else None
        ))
        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "application_id": app_id,
        "status": eligibility,
        "cv_results": cv_results,
        "rejection_reasons": rejection_reasons
    }

@app.get("/api/track/{application_id}")
def track_application(application_id: str):
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM applications WHERE application_id = %s", (application_id,))
        row = cursor.fetchone()
        cursor.close()
        db.close()
        if not row:
            raise HTTPException(status_code=404, detail="Application not found")
        if row.get("rejection_reasons"):
            try:
                row["rejection_reasons"] = json.loads(row["rejection_reasons"])
            except:
                pass
        row.pop("aadhar_no", None)
        return row
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/raise-ticket/{application_id}")
def raise_ticket(application_id: str):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "UPDATE applications SET ticket_raised=1, admin_status='pending_review' WHERE application_id=%s",
            (application_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Application not found")
        db.commit()
        cursor.close()
        db.close()
        return {"message": "Ticket raised successfully. Admin will review your application.", "application_id": application_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── ADMIN ROUTES ─────────────────────────────────────────────────────────────

@app.get("/api/admin/applications")
def get_all_applications(status: str = "all"):
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        if status == "all":
            cursor.execute("""
                SELECT application_id, name, education_level, state, email, phone,
                       submitted_at, cv_marks, cv_income, cv_aadhar_match,
                       eligibility_status, ticket_raised, admin_status, rejection_reasons
                FROM applications ORDER BY submitted_at DESC
            """)
        else:
            cursor.execute("""
                SELECT application_id, name, education_level, state, email, phone,
                       submitted_at, cv_marks, cv_income, cv_aadhar_match,
                       eligibility_status, ticket_raised, admin_status, rejection_reasons
                FROM applications WHERE admin_status=%s OR ticket_raised=1
                ORDER BY submitted_at DESC
            """, (status,))
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        for row in rows:
            if row.get("rejection_reasons"):
                try:
                    row["rejection_reasons"] = json.loads(row["rejection_reasons"])
                except:
                    pass
            if row.get("submitted_at"):
                row["submitted_at"] = str(row["submitted_at"])
        return {"applications": rows, "total": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/application/{application_id}")
def get_application_detail(application_id: str):
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM applications WHERE application_id=%s", (application_id,))
        row = cursor.fetchone()
        cursor.close()
        db.close()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        for key in ["submitted_at", "admin_reviewed_at"]:
            if row.get(key):
                row[key] = str(row[key])
        if row.get("rejection_reasons"):
            try:
                row["rejection_reasons"] = json.loads(row["rejection_reasons"])
            except:
                pass
        return row
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/review/{application_id}")
async def admin_review(
    application_id: str,
    decision: str = Form(...),
    notes: str = Form(default="")
):
    if decision not in ["approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Decision must be 'approved' or 'rejected'")
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            UPDATE applications
            SET admin_status=%s, admin_notes=%s, admin_reviewed_at=%s,
                eligibility_status=%s
            WHERE application_id=%s
        """, (decision, notes, datetime.now(), decision, application_id))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Application not found")
        db.commit()
        cursor.close()
        db.close()
        return {"message": f"Application {decision} successfully", "application_id": application_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/stats")
def get_stats():
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(eligibility_status='approved') as approved,
                SUM(eligibility_status='rejected') as rejected,
                SUM(eligibility_status='pending') as pending,
                SUM(ticket_raised=1) as tickets,
                SUM(admin_status='pending_review') as pending_review
            FROM applications
        """)
        stats = cursor.fetchone()
        cursor.close()
        db.close()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))