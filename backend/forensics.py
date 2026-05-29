import io
import base64
import os
import sys
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import cv2
import numpy as np

# Try to import the deep learning Forensic AI Engine
HAS_FORENSIC_AI = False
try:
    forenic_ai_path = os.path.join(os.path.dirname(__file__), 'forensic_ai')
    if os.path.exists(forenic_ai_path):
        sys.path.append(forenic_ai_path)
        from src.inference.pipeline import ForensicEngine
        HAS_FORENSIC_AI = True
except Exception:
    pass

_ai_engine = None

def get_ai_engine():
    global _ai_engine
    if _ai_engine is None and HAS_FORENSIC_AI:
        try:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), 'forensic_ai', 'outputs', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            _ai_engine = ForensicEngine(checkpoint_dir=checkpoint_dir)
            _ai_engine.warmup()
        except Exception as e:
            print(f"Warning: Failed to warm up Forensic AI Engine: {e}")
    return _ai_engine


def perform_ela(image_bytes, quality=95):
    """
    Error Level Analysis (ELA).

    NOTE: ELA is a JPEG-specific technique.  Running it on a lossless PNG or
    WEBP file produces near-zero differences and meaningless scores.  We
    detect the source format and skip ELA (returning score=0) for non-JPEG
    inputs rather than silently returning garbage numbers.
    """
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── FIX #11: Detect format; ELA is only meaningful on JPEG ──────────────
    source_format = Image.open(io.BytesIO(image_bytes)).format  # "JPEG", "PNG", "WEBP", etc.
    is_jpeg = source_format in ("JPEG", "MPO")

    if not is_jpeg:
        # Return a neutral score and an empty ELA image for non-JPEG files.
        # The frontend will display an appropriate notice instead of fake results.
        return 0.0, "", original, False  # last bool = ela_applicable

    # Save to a temporary buffer with set quality
    temp_buffer = io.BytesIO()
    original.save(temp_buffer, format="JPEG", quality=quality)
    temp_buffer.seek(0)
    resaved = Image.open(temp_buffer)

    diff = ImageChops.difference(original, resaved)

    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale_factor = 255.0 / max_diff
    enhanced_diff = ImageEnhance.Brightness(diff).enhance(scale_factor)

    diff_gray = diff.convert("L")
    diff_arr = np.array(diff_gray)
    h, w = diff_arr.shape

    img_gray = original.convert("L")
    img_arr = np.array(img_gray)

    sobelx = cv2.Sobel(img_arr, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_arr, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)

    block_size = 16
    blocks = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block_diff = diff_arr[y:y+block_size, x:x+block_size]
            block_grad = grad_mag[y:y+block_size, x:x+block_size]
            mean_diff = np.mean(block_diff)
            mean_grad = np.mean(block_grad)
            blocks.append((mean_grad, mean_diff))

    if len(blocks) < 4:
        return 0.0, "", diff, True

    blocks.sort(key=lambda x: x[0])
    num_flat = max(4, int(len(blocks) * 0.30))
    flat_blocks = blocks[:num_flat]
    flat_elas = [b[1] for b in flat_blocks]
    median_ela = np.median(flat_elas)
    max_ela = np.max(flat_elas)

    alpha = 2.0
    anomaly_ratio = (max_ela - median_ela) / (median_ela + alpha)

    if anomaly_ratio < 0.8:
        ela_score = anomaly_ratio * 15.0
    elif anomaly_ratio < 2.0:
        ela_score = 12.0 + (anomaly_ratio - 0.8) * 25.0
    else:
        ela_score = min(100.0, 42.0 + (anomaly_ratio - 2.0) * 15.0)

    ela_strength = min(1.0, max_ela / 6.0)
    ela_score = ela_score * ela_strength

    buffered = io.BytesIO()
    enhanced_diff.save(buffered, format="JPEG")
    ela_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return round(ela_score, 1), ela_base64, diff, True  # ela_applicable=True


def analyze_metadata(image_bytes):
    """
    EXIF Metadata Analyzer.
    Extracts tags and flags software edits (Photoshop, Canva, GIMP, etc.).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()

        metadata_score = 0.0
        found_software = None
        has_camera_info = False
        info = {}

        if exif:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                info[str(tag_name)] = str(value)

            software_keys = ["Software", "ProcessingSoftware"]
            for key in software_keys:
                if key in info:
                    val = info[key].lower()
                    if any(sw in val for sw in [
                        "photoshop", "gimp", "canva", "adobe", "pixelmator",
                        "lightroom", "picsart", "snapseed", "stable diffusion",
                        "midjourney", "dall-e", "comfyui", "invokeai", "faceapp"
                    ]):
                        found_software = info[key]
                        metadata_score = 95.0
                        break

            if "Make" in info or "Model" in info:
                has_camera_info = True

        if not has_camera_info and not found_software:
            metadata_score = 20.0

        return round(metadata_score, 1), info, found_software
    except Exception:
        return 0.0, {}, None


def analyze_noise(image_bytes):
    """Local Noise Level Analysis."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        denoised = cv2.medianBlur(img, 3)
        noise = cv2.absdiff(img, denoised)

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobelx, sobely)

        h, w = img.shape
        block_size = 32
        blocks = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                patch_noise = noise[y:y+block_size, x:x+block_size]
                patch_grad = grad_mag[y:y+block_size, x:x+block_size]
                mean_grad = np.mean(patch_grad)
                var_noise = np.var(patch_noise)
                blocks.append((mean_grad, var_noise))

        if len(blocks) < 4:
            return 0.0

        blocks.sort(key=lambda x: x[0])
        num_flat = max(4, int(len(blocks) * 0.30))
        flat_blocks = blocks[:num_flat]
        noise_vars = [b[1] for b in flat_blocks]
        median_noise = np.median(noise_vars)
        max_noise = np.max(noise_vars)

        alpha = 2.0
        noise_ratio = (max_noise + alpha) / (median_noise + alpha)

        avg_flat_grad = np.mean([b[0] for b in flat_blocks])
        grad_reliability = 1.0
        if avg_flat_grad > 20.0:
            grad_reliability = max(0.1, 1.0 - (avg_flat_grad - 20.0) / 30.0)

        noise_strength = min(1.0, max_noise / 4.0)

        if noise_ratio < 1.8:
            noise_score = noise_ratio * 10.0
        elif noise_ratio < 3.0:
            noise_score = 18.0 + (noise_ratio - 1.8) * 20.0
        else:
            noise_score = min(100.0, 42.0 + (noise_ratio - 3.0) * 10.0)

        final_score = noise_score * grad_reliability * noise_strength
        return round(final_score, 1)
    except Exception:
        return 0.0


def analyze_face_manipulation(image_bytes, ela_diff_image):
    """
    Face Splicing and Deepfake Detector using Haar Cascades.

    Note: Haar Cascades are limited technology. A 0.0 score means no frontal
    face was detected — it does NOT guarantee the image is authentic.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return 0.0, 0

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        num_faces = len(faces)
        if num_faces == 0:
            return 0.0, 0

        diff_arr = np.array(ela_diff_image.convert("L"))
        overall_ela_val = np.mean(diff_arr)

        face_ela_scores = []
        for (x, y, w, h) in faces:
            face_patch = diff_arr[y:y+h, x:x+w]
            face_ela_scores.append(np.mean(face_patch))

        max_face_ela = max(face_ela_scores)
        beta = 2.0
        ratio = (max_face_ela + beta) / (overall_ela_val + beta)

        if ratio <= 1.3:
            face_score = ratio * 10.0
        elif ratio <= 2.2:
            face_score = 13.0 + (ratio - 1.3) * 35.0
        else:
            face_score = min(100.0, 44.5 + (ratio - 2.2) * 12.0)

        return round(face_score, 1), num_faces
    except Exception:
        return 0.0, 0


def _compute_real_confidence(ela_score, noise_score, face_score, metadata_score,
                              num_faces, ela_applicable, image_pixels):
    """
    FIX #12: Compute confidence based on how many independent signals fired
    and the strength of the dominant indicator — NOT purely from image resolution.

    Confidence reflects how certain the algorithm is, not how big the file is.
    """
    active_signals = 0
    total_weight = 0.0
    weighted_sum = 0.0

    if ela_applicable:
        # ELA is the strongest independent signal
        weight = 0.35
        total_weight += weight
        weighted_sum += min(ela_score, 100) * weight
        if ela_score > 15:
            active_signals += 1
    else:
        # No ELA (PNG/WEBP) — reduced confidence ceiling
        total_weight += 0.0  # doesn't contribute

    # Noise
    weight = 0.30
    total_weight += weight
    weighted_sum += min(noise_score, 100) * weight
    if noise_score > 15:
        active_signals += 1

    # Face (only if faces were detected)
    if num_faces > 0:
        weight = 0.25
        total_weight += weight
        weighted_sum += min(face_score, 100) * weight
        if face_score > 15:
            active_signals += 1

    # Metadata
    weight = 0.10
    total_weight += weight
    weighted_sum += min(metadata_score, 100) * weight
    if metadata_score > 20:
        active_signals += 1

    if total_weight == 0:
        return 55.0  # Cannot evaluate — return mid-range

    # Base certainty from proportion of signals firing
    base_certainty = 55.0 + (active_signals / max(1, (3 if num_faces > 0 else 2))) * 35.0

    # Penalize low-resolution images (less information to analyze)
    if image_pixels < 100_000:
        resolution_penalty = 15.0
    elif image_pixels < 400_000:
        resolution_penalty = 5.0
    else:
        resolution_penalty = 0.0

    # Penalize if ELA was inapplicable (PNG/WEBP)
    format_penalty = 10.0 if not ela_applicable else 0.0

    confidence = base_certainty - resolution_penalty - format_penalty
    return round(min(97.0, max(42.0, confidence)), 1)


def run_forensic_suite(image_bytes):
    """
    Executes the entire suite of forensics tools and combines reports.
    Falls back gracefully to classical pipeline if the AI engine is unavailable.
    """
    # Detect source format for ELA applicability check
    try:
        source_format = Image.open(io.BytesIO(image_bytes)).format
        ela_applicable = source_format in ("JPEG", "MPO")
    except Exception:
        ela_applicable = False

    # Always try to generate ELA visual for JPEG images
    ela_score, ela_base64, ela_diff_obj, _ = perform_ela(image_bytes)

    engine = get_ai_engine()
    if HAS_FORENSIC_AI and engine is not None:
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            ai_result = engine.analyze_image(img_pil)

            ai_status = ai_result.get("status", "AUTHENTIC")
            ai_risk = ai_result.get("risk_level", "MINIMAL").lower()

            if ai_status == "MANIPULATED" or ai_risk in ["critical", "high"]:
                status = "Manipulated"
                risk = "red"
            elif ai_risk == "medium" or ai_status == "SUSPICIOUS":
                status = "Suspicious"
                risk = "yellow"
            else:
                status = "Safe"
                risk = "green"

            face_score  = round(ai_result.get("deepfake_score", 0.0) * 100, 1)
            splice_score = round(ai_result.get("manipulation_score", 0.0) * 100, 1)
            metadata_score = round(ai_result.get("metadata_score", 0.0) * 100, 1)
            noise_score = round(ai_result.get("noise_score", 0.0) * 100, 1)
            ela_score_ai = round(ai_result.get("ela_score", 0.0) * 100, 1)

            forensic_score = round(max(face_score, splice_score, noise_score, ela_score_ai), 1)
            # ── FIX #4: Do NOT artificially clamp forensic_score ─────────────
            # If the model says it's Manipulated but all individual scores are
            # genuinely low, report that honestly rather than fabricating a number.

            # ── FIX #12: Real confidence from AI engine output ────────────────
            confidence_score = round(min(97.0, ai_result.get("confidence", 0.75) * 100), 1)

            metadata_details = ai_result.get("metadata", {}).get("details", {})
            software_used = ai_result.get("metadata", {}).get("software")

            return {
                "imageStatus": status,
                "riskLevel": risk,
                "confidenceScore": confidence_score,
                "forensicScore": forensic_score,
                "elaApplicable": ela_applicable,
                "details": {
                    "faceManipulation": face_score,
                    "spliceDetection": splice_score,
                    "metadataAnomaly": metadata_score,
                    "noiseAnalysis": noise_score
                },
                "metadata": {
                    "software": software_used,
                    "has_exif": len(metadata_details) > 0,
                    "details": metadata_details
                },
                "ela_image": ela_base64
            }
        except Exception as e:
            print(f"Warning: Forensic AI Engine execution failed, falling back: {e}")

    # ── CLASSICAL FALLBACK PIPELINE ───────────────────────────────────────────
    ela_score, ela_base64, diff_image, ela_applicable = perform_ela(image_bytes)
    metadata_score, metadata_details, software_used = analyze_metadata(image_bytes)
    noise_score = analyze_noise(image_bytes)

    # Face analysis only makes sense when ELA diff is available (JPEG)
    if ela_applicable and diff_image is not None:
        face_score, faces_count = analyze_face_manipulation(image_bytes, diff_image)
    else:
        face_score, faces_count = 0.0, 0

    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    pixels = w * h

    # Apply resolution correction for small images
    if pixels < 120_000:
        resolution_factor = max(0.2, pixels / 120_000.0)
        ela_score  = round(ela_score * resolution_factor, 1)
        noise_score = round(noise_score * resolution_factor, 1)
        if faces_count > 0:
            face_score = round(face_score * resolution_factor, 1)

    splice_score = round(min(100.0, (ela_score * 0.65 + noise_score * 0.35)), 1)

    is_manipulated = False
    is_suspicious = False

    if splice_score >= 45.0:
        is_manipulated = True
    elif splice_score >= 25.0:
        is_suspicious = True

    if faces_count > 0:
        if face_score >= 45.0:
            is_manipulated = True
        elif face_score >= 25.0:
            is_suspicious = True

    if software_used:
        if ela_score >= 20.0 or noise_score >= 20.0:
            is_manipulated = True
        else:
            is_suspicious = True

    if is_manipulated:
        status = "Manipulated"
        risk = "red"
        forensic_score = max(ela_score, noise_score, face_score if faces_count > 0 else 0)
        # ── FIX #4: Report the REAL score — do not clamp to fake 68.4 ─────────
        # If evidence is weak, show weak numbers and let the user judge.
        forensic_score = round(forensic_score, 1)
    elif is_suspicious:
        status = "Suspicious"
        risk = "yellow"
        forensic_score = round(max(ela_score, noise_score, face_score if faces_count > 0 else 0), 1)
    else:
        status = "Safe"
        risk = "green"
        forensic_score = round(max(1.0, min(15.0, (ela_score * 0.5 + noise_score * 0.5))), 1)

    # ── FIX #12: Real confidence score (not just image resolution) ────────────
    confidence_score = _compute_real_confidence(
        ela_score, noise_score, face_score, metadata_score,
        faces_count, ela_applicable, pixels
    )

    return {
        "imageStatus": status,
        "riskLevel": risk,
        "confidenceScore": confidence_score,
        "forensicScore": forensic_score,
        "elaApplicable": ela_applicable,
        "details": {
            "faceManipulation": face_score,
            "spliceDetection": splice_score,
            "metadataAnomaly": metadata_score,
            "noiseAnalysis": noise_score
        },
        "metadata": {
            "software": software_used,
            "has_exif": len(metadata_details) > 0,
            "details": metadata_details
        },
        "ela_image": ela_base64
    }
