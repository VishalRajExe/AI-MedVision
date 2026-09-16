"""
FastAPI application for ECG Arrhythmia Classification
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import ECGSignal, PredictionResponse, HealthResponse
from inference import load_model, predict_ecg, is_model_loaded, scaler
import logging
import pandas as pd
import numpy as np
import io
import csv
import random
from pathlib import Path
import os

DATA_PATH = Path(os.getenv('DATA_PATH', './data'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ECG Arrhythmia Classifier",
    description="API for heart rhythm disorder classification from ECG signals (with normalization)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== STARTUP ==================

@app.on_event("startup")
async def startup_event():
    """Load model and scaler when app starts"""
    logger.info("🚀 Starting up ECG Classifier API...")
    logger.info("Loading model and scaler...")
    success = load_model()
    if success:
        logger.info("✅ API ready for predictions (with normalization)")
    else:
        logger.error("❌ Failed to load model/scaler - check paths")

# ================== HEALTH CHECK ==================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded()
    }

@app.get("/", tags=["Info"])
async def root():
    """API information and status"""
    return {
        "message": "ECG Arrhythmia Classifier API",
        "version": "1.1.0",
        "docs": "http://localhost:8000/docs",
        "model_loaded": is_model_loaded(),
        "note": "All signals are normalized using StandardScaler before prediction"
    }

# ================== SINGLE PREDICTION ==================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(ecg_data: ECGSignal):
    """
    Classify single ECG signal
    - Signal will be automatically padded/truncated to 187 samples
    - Signal will be normalized using the trained scaler
    """
    try:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

        if len(ecg_data.signal) == 0:
            raise HTTPException(status_code=400, detail="Signal cannot be empty")

        result = predict_ecg(ecg_data.signal)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info(f"✅ Prediction: {result['predicted_class']} (confidence: {result['confidence']:.2%})")
        return result

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"❌ Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ================== FILE UPLOAD - CSV ==================

def parse_csv_ecg_signal(contents: bytes) -> np.ndarray:
    """
    Robust universal extractor for ECG signal data from uploaded CSV bytes.
    Handles:
    - Single row of 187 (or N) comma/semicolon/tab-separated values (no headers)
    - Single row with header row above it
    - Columnar data with header ('sample_index, ecg_value', 'time, signal', etc.)
    - Single column of values (vertical)
    - Two columns without headers (index, value)
    - Encodings: utf-8, utf-8-sig (with BOM), latin-1, cp1252
    - Files with trailing empty cells/commas
    """
    text = None
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            text = contents.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if not text or not text.strip():
        raise ValueError("CSV file is empty")

    text = text.lstrip('\ufeff').strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("CSV file is empty")

    # Detect delimiter
    first_few = '\n'.join(lines[:10])
    counts = {',': first_few.count(','), ';': first_few.count(';'), '\t': first_few.count('\t')}
    delimiter = max(counts, key=counts.get)
    if counts[delimiter] == 0:
        delimiter = ','

    # Parse lines with csv.reader to avoid rigid rectangular shape assumptions
    rows = []
    reader = csv.reader(lines, delimiter=delimiter)
    for r in reader:
        # Strip trailing empty cells
        while r and r[-1].strip() == '':
            r.pop()
        if r:
            rows.append([c.strip() for c in r])

    if not rows:
        raise ValueError("No data found in CSV file")

    # Case 1: Exactly 1 row (user's Excel row export / single-row CSV)
    if len(rows) == 1:
        signal = pd.to_numeric(pd.Series(rows[0]), errors='coerce').dropna().values.astype(np.float32)
        if len(signal) > 0:
            return signal

    # Case 2: 2 rows where row 0 is header text and row 1 is data
    if len(rows) == 2:
        r0_num = pd.to_numeric(pd.Series(rows[0]), errors='coerce').dropna().values
        r1_num = pd.to_numeric(pd.Series(rows[1]), errors='coerce').dropna().values
        if len(r1_num) > len(r0_num) and len(r1_num) > 0:
            return r1_num.astype(np.float32)
        elif len(r0_num) > 0:
            return r0_num.astype(np.float32)

    # Case 3: Columnar format with named header
    header = [str(c).strip().lower() for c in rows[0]]
    keywords = ['ecg_value', 'ecg', 'value', 'signal', 'amplitude', 'mv', 'data', 'reading', 'lead', 'mlii']
    matched_col_idx = None
    for idx, col_name in enumerate(header):
        if any(kw in col_name for kw in keywords):
            matched_col_idx = idx
            break

    if matched_col_idx is not None and len(rows) > 1:
        vals = [r[matched_col_idx] for r in rows[1:] if len(r) > matched_col_idx]
        signal = pd.to_numeric(pd.Series(vals), errors='coerce').dropna().values.astype(np.float32)
        if len(signal) > 0:
            return signal

    # Check if row 0 has non-numeric text (header)
    r0_has_str = any(not cell.replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).replace('e', '', 1).replace('E', '', 1).isdigit() for cell in rows[0] if cell)
    data_rows = rows[1:] if r0_has_str else rows

    # Case 4: Single column (vertical numbers)
    if all(len(r) == 1 for r in data_rows[:20]):
        vals = [r[0] for r in data_rows if len(r) > 0]
        signal = pd.to_numeric(pd.Series(vals), errors='coerce').dropna().values.astype(np.float32)
        if len(signal) > 0:
            return signal

    # Case 5: Two columns (index, value)
    if all(len(r) == 2 for r in data_rows[:20]):
        vals = [r[1] for r in data_rows if len(r) > 1]
        signal = pd.to_numeric(pd.Series(vals), errors='coerce').dropna().values.astype(np.float32)
        if len(signal) > 0:
            return signal

    # Case 6: Find column with the most numeric items
    max_col = max(len(r) for r in data_rows)
    best_col_signal = None
    max_num_count = 0
    for c in range(max_col):
        col_vals = [r[c] for r in data_rows if len(r) > c]
        nums = pd.to_numeric(pd.Series(col_vals), errors='coerce').dropna().values
        if len(nums) > max_num_count:
            max_num_count = len(nums)
            best_col_signal = nums

    if best_col_signal is not None and len(best_col_signal) >= 10:
        return best_col_signal.astype(np.float32)

    # Fallback: flatten all numeric cells
    all_cells = [cell for r in rows for cell in r]
    signal = pd.to_numeric(pd.Series(all_cells), errors='coerce').dropna().values.astype(np.float32)
    if len(signal) > 0:
        return signal

    raise ValueError("No valid numeric ECG signal data found in CSV")


@app.post("/upload-csv", tags=["File Upload"])
async def upload_csv_file(file: UploadFile = File(...)):
    """
    Upload and classify ECG signal from CSV file
    - Handles single-row format (all values in one row)
    - Handles columnar format (values in a named column like 'ecg_value', 'value', 'signal', etc., with optional index column)
    - Automatically normalizes the signal
    - Returns prediction with both raw and normalized signals
    """
    try:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        contents = await file.read()

        try:
            signal_values = parse_csv_ecg_signal(contents)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {e}")

        logger.info(f"📥 Parsed ECG signal: {len(signal_values)} samples")

        # ── Strip excessive trailing zeros for cleaner visualization ────
        signal_for_plot = signal_values.copy()
        trailing_zeros = 0
        for i in range(len(signal_for_plot) - 1, -1, -1):
            if signal_for_plot[i] == 0:
                trailing_zeros += 1
            else:
                break
        if trailing_zeros > 50:
            signal_for_plot = signal_for_plot[:-trailing_zeros]

        # ── Prepare 187 samples for model and scaler (pad or truncate) ──
        model_signal = signal_values[:187].copy()
        if len(model_signal) < 187:
            model_signal = np.pad(model_signal, (0, 187 - len(model_signal)), mode='constant')

        # ── Predict ───────────────────────────────────────────────────────
        result = predict_ecg(model_signal.tolist())

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        result['signal_raw'] = signal_for_plot.tolist()
        normalized_signal = scaler.transform(model_signal.reshape(1, -1))[0]
        result['signal_normalized'] = normalized_signal.tolist()

        logger.info(f"✅ Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing CSV: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


# ================== TEST DATA SAMPLES ==================

@app.get("/test-samples", tags=["Test Data"])
async def get_test_samples(count: int = 5):
    """
    Get random samples from MIT-BIH test dataset with predictions
    - Returns both raw and normalized signals for frontend display
    - Includes true labels and prediction accuracy
    """
    try:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        X_test_path = DATA_PATH / 'X_test.npy'
        y_test_path = DATA_PATH / 'y_test.npy'

        if not X_test_path.exists() or not y_test_path.exists():
            raise FileNotFoundError("Test data files not found in ../data/")

        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        max_count = min(count, len(X_test))
        indices = random.sample(range(len(X_test)), max_count)

        samples = []
        class_map = {
            0: 'Normal', 1: 'Supraventricular', 2: 'Ventricular',
            3: 'Fusion', 4: 'Unknown'
        }

        for idx in indices:
            signal_raw = X_test[idx].flatten()
            label = int(y_test[idx])

            result = predict_ecg(signal_raw.tolist())

            result['signal_raw'] = signal_raw.tolist()

            signal_normalized = scaler.transform(signal_raw.reshape(1, -1))[0]
            result['signal_normalized'] = signal_normalized.tolist()

            result['true_label'] = class_map.get(label, 'Unknown')
            result['true_label_id'] = label
            result['index'] = int(idx)
            result['is_correct'] = result['predicted_class'] == result['true_label']

            samples.append(result)

        logger.info(f"✅ Generated {len(samples)} test samples with signals")
        return {"samples": samples, "count": len(samples)}

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Test samples error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ================== EXCEPTION HANDLER ==================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ================== RUN ==================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
