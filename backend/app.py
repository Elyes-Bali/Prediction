import os
import sys
import io
import time
import pickle
import mimetypes
import traceback

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ------------------------------------------------------------
# Configuration via environment
# ------------------------------------------------------------
DEFAULT_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "128"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "1000"))
BEST_THRESHOLD = float(os.environ.get("BEST_THRESHOLD", "0.69"))

# ------------------------------------------------------------
# MIME TYPES (defensive)
# ------------------------------------------------------------
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
DIST_DIR = os.path.join(FRONTEND_DIR, 'dist')  # must exist after build

print(f"[Startup] BASE_DIR={BASE_DIR}")
print(f"[Startup] FRONTEND_DIR={FRONTEND_DIR}")
print(f"[Startup] DIST_DIR={DIST_DIR}")

# Flask app
app = Flask(__name__, static_folder=DIST_DIR, static_url_path='/', template_folder=DIST_DIR)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Upload folder
UPLOAD_FOLDER_NAME = 'uploads'
UPLOAD_PATH = os.path.join(BASE_DIR, '..', UPLOAD_FOLDER_NAME)
os.makedirs(UPLOAD_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB safeguard

# Model paths (relative to project root)
MODEL_PATH = "backend/model/modele_detection_panne.keras"
SCALER_PATH = "backend/model/scaler.pkl"

FEATURE_COLUMNS = [
    'FIC2003_valeur', 'LI2001_valeur', 'MX2001_valeur',
    'MX2002A_valeur', 'MX2002B_valeur', 'MX2002C_valeur',
    'MX2002D_valeur', 'MX2003A_valeur', 'MX2003B_valeur',
    'MX2003C_valeur', 'temperature_ext', 'TI2001__2_valeur'
]

# ------------------------------------------------------------
# Globals (lazy-loaded)
# ------------------------------------------------------------
_global_model = None
_global_scaler = None

def load_scaler():
    global _global_scaler
    if _global_scaler is not None:
        return _global_scaler
    try:
        with open(SCALER_PATH, 'rb') as f:
            _global_scaler = pickle.load(f)
        print("[Scaler] Loaded training scaler.")
    except FileNotFoundError:
        print("[Scaler] WARNING: scaler.pkl missing; will fit temporary scaler per inference.")
        _global_scaler = None
    return _global_scaler

def load_model():
    global _global_model
    if _global_model is not None:
        return _global_model
    t0 = time.time()
    print(f"[Model] Loading model from {MODEL_PATH} ...")
    _global_model = keras.models.load_model(MODEL_PATH)
    print(f"[Model] Loaded in {time.time()-t0:.2f}s")
    return _global_model

# ------------------------------------------------------------
# Generator (memory-efficient)
# ------------------------------------------------------------
def sequence_generator(data, time_steps, batch_size):
    n_sequences = len(data) - time_steps + 1
    for start in range(0, n_sequences, batch_size):
        end = min(start + batch_size, n_sequences)
        # Build batch
        batch_len = end - start
        X = np.empty((batch_len, time_steps, data.shape[1]), dtype=np.float32)
        for i, seq_idx in enumerate(range(start, end)):
            X[i] = data[seq_idx:seq_idx + time_steps]
        yield (X,)

# ------------------------------------------------------------
# Inference Core
# ------------------------------------------------------------
def run_inference(csv_path, batch_size):
    old_stdout = sys.stdout
    sys.stdout = log_stream = io.StringIO()
    try:
        print(f"[Inference] CSV path: {csv_path}")
        df_raw = pd.read_csv(csv_path)

        if 'Unnamed: 0' in df_raw.columns:
            df_raw = df_raw.drop(columns=['Unnamed: 0'])

        # Validate necessary columns
        missing_cols = [c for c in FEATURE_COLUMNS + ['datetime'] if c not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_features = df_raw[FEATURE_COLUMNS].astype('float32')  # ensure float32
        data_np = df_features.to_numpy(dtype=np.float32)

        print(f"[Inference] Rows={len(df_raw)}, Features={df_features.shape[1]}")
        scaler = load_scaler()
        if scaler is not None:
            data_scaled = scaler.transform(data_np)
        else:
            print("[Inference] Fitting temporary StandardScaler (WARNING)")
            temp_scaler = StandardScaler()
            data_scaled = temp_scaler.fit_transform(data_np)

        data_scaled = data_scaled.astype(np.float32)

        if len(data_scaled) < WINDOW_SIZE:
            raise ValueError(f"Insufficient rows ({len(data_scaled)}) for window size {WINDOW_SIZE}")

        model = load_model()

        n_sequences = len(data_scaled) - WINDOW_SIZE + 1
        # Auto-limit batch size for extremely large sequences
        effective_batch = min(batch_size, max(1, n_sequences // 4) if n_sequences > batch_size * 10 else batch_size)
        print(f"[Inference] Sequences: {n_sequences}, Requested batch: {batch_size}, Effective batch: {effective_batch}")

        steps = int(np.ceil(n_sequences / effective_batch))
        print(f"[Inference] Predicting in {steps} steps...")

        t_pred = time.time()
        y_proba = model.predict(
            sequence_generator(data_scaled, WINDOW_SIZE, effective_batch),
            steps=steps,
            verbose=1
        ).flatten()
        print(f"[Inference] Prediction time: {time.time()-t_pred:.2f}s")

        y_pred_binary = (y_proba >= BEST_THRESHOLD).astype(int)
        labels = np.where(y_pred_binary == 1, 'Fault (Panne)', 'Normal')

        start_index = WINDOW_SIZE - 1
        datetime_end = df_raw['datetime'].iloc[start_index:].reset_index(drop=True)

        results_df = pd.DataFrame({
            'TimeWindow_Ends_At': datetime_end.iloc[:len(y_proba)],
            'Fault_Probability': y_proba,
            'Prediction': labels
        })

        print("[Inference] Aggregating fault events...")
        time_step_duration = df_raw['datetime'].iloc[1] - df_raw['datetime'].iloc[0]
        offset = (WINDOW_SIZE - 1) * time_step_duration
        results_df['TimeWindow_Start_At'] = results_df['TimeWindow_Ends_At'] - offset
        results_df = results_df[['TimeWindow_Start_At', 'TimeWindow_Ends_At', 'Fault_Probability', 'Prediction']]

        is_fault = results_df['Prediction'] == 'Fault (Panne)'
        group_change = is_fault.diff().fillna(is_fault.iloc[0])
        group_id = (group_change == True).cumsum()

        fault_subset = results_df[is_fault].copy()
        if not fault_subset.empty:
            fault_subset['group'] = group_id[is_fault]
            fault_summary = fault_subset.groupby('group').agg(
                Fault_Start_Time=('TimeWindow_Start_At', 'min'),
                Fault_End_Time=('TimeWindow_Ends_At', 'max'),
                Max_Probability=('Fault_Probability', 'max'),
                Duration_Windows=('Prediction', 'count')
            ).reset_index(drop=True)
            fault_summary['Duration'] = fault_summary['Fault_End_Time'] - fault_summary['Fault_Start_Time']
        else:
            fault_summary = pd.DataFrame(columns=[
                'Fault_Start_Time', 'Fault_End_Time', 'Max_Probability', 'Duration_Windows', 'Duration'
            ])

        print("[Inference] Complete.")
        return {
            'status': 'success',
            'logs': log_stream.getvalue(),
            'summary_data': fault_summary.to_json(orient='split', date_format='iso'),
            'detailed_data': results_df.head(100).to_json(orient='split', date_format='iso')
        }
    except Exception as e:
        print("[Inference] ERROR:", e)
        traceback.print_exc()
        return {
            'status': 'error',
            'error_message': str(e),
            'logs': log_stream.getvalue()
        }
    finally:
        sys.stdout = old_stdout

# ------------------------------------------------------------
# API: Preview
# ------------------------------------------------------------
@app.route('/api/upload-preview', methods=['POST'])
def upload_preview():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        stream = io.StringIO(f.stream.read().decode("utf-8"))
        df = pd.read_csv(stream)
        preview_df = df.fillna('N/A')
        return jsonify({
            'status': 'preview_ready',
            'filename': f.filename,
            'preview_data': preview_df.to_json(orient='split', date_format='iso')
        })
    except Exception as e:
        return jsonify({'error': f"Failed to read CSV: {e}"}), 500

# ------------------------------------------------------------
# API: Run Model
# ------------------------------------------------------------
@app.route('/api/run-model', methods=['POST'])
def run_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        f.save(path)
        result = run_inference(path, DEFAULT_BATCH_SIZE)
        code = 200 if result.get('status') == 'success' else 500
        return jsonify(result), code
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error_message': str(e)
        }), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

# ------------------------------------------------------------
# Frontend (SPA) Routes
# ------------------------------------------------------------
@app.route('/')
def index():
    if not os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return "Front-end build missing. Run npm build.", 500
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:subpath>')
def spa_or_asset(subpath):
    full = os.path.join(app.static_folder, subpath)
    if os.path.exists(full) and not os.path.isdir(full):
        return send_from_directory(app.static_folder, subpath)
    return send_from_directory(app.static_folder, 'index.html')

# ------------------------------------------------------------
# Healthcheck
# ------------------------------------------------------------
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': _global_model is not None})

# ------------------------------------------------------------
# Local run
# ------------------------------------------------------------
if __name__ == '__main__':
    if not os.path.isdir(DIST_DIR):
        print("WARNING: dist directory not found. Run build before starting.")
    app.run(host='0.0.0.0', port=5000, debug=True)