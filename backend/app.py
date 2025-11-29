import os
import sys
import io
import time
import pickle
import mimetypes

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ------------------------------------------------------------
# MIME TYPES (defensive; build output usually fine)
# ------------------------------------------------------------
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
DIST_DIR = os.path.join(FRONTEND_DIR, 'dist')  # Vite build output

print(f"[Startup] BASE_DIR={BASE_DIR}")
print(f"[Startup] FRONTEND_DIR={FRONTEND_DIR}")
print(f"[Startup] DIST_DIR={DIST_DIR}")

# Flask will serve built assets from dist
app = Flask(__name__, static_folder=DIST_DIR, static_url_path='/', template_folder=DIST_DIR)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ------------------------------------------------------------
# UPLOAD FOLDER
# ------------------------------------------------------------
UPLOAD_FOLDER_NAME = 'uploads'
UPLOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', UPLOAD_FOLDER_NAME)
os.makedirs(UPLOAD_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH

# ------------------------------------------------------------
# MODEL / SCALER PATHS
# ------------------------------------------------------------
MODEL_PATH = "backend/model/modele_detection_panne.keras"
SCALER_PATH = "backend/model/scaler.pkl"

# ------------------------------------------------------------
# INFERENCE SETTINGS
# ------------------------------------------------------------
BATCH_SIZE = 4096
WINDOW_SIZE = 1000
BEST_THRESHOLD = 0.69

FEATURE_COLUMNS = [
    'FIC2003_valeur', 'LI2001_valeur', 'MX2001_valeur',
    'MX2002A_valeur', 'MX2002B_valeur', 'MX2002C_valeur',
    'MX2002D_valeur', 'MX2003A_valeur', 'MX2003B_valeur',
    'MX2003C_valeur', 'temperature_ext', 'TI2001__2_valeur'
]

# ------------------------------------------------------------
# GENERATOR
# ------------------------------------------------------------
def sequence_generator(data, time_steps, batch_size):
    n_sequences = len(data) - time_steps + 1
    for i in range(0, n_sequences, batch_size):
        batch_end_index = min(i + batch_size, n_sequences)
        X_batch = [data[idx:idx + time_steps] for idx in range(i, batch_end_index)]
        yield (np.array(X_batch),)

# ------------------------------------------------------------
# INFERENCE CORE
# ------------------------------------------------------------
def run_inference(csv_path):
    old_stdout = sys.stdout
    sys.stdout = log_stream = io.StringIO()
    try:
        print(f"Loading CSV: {csv_path}")
        df_raw = pd.read_csv(csv_path)

        if 'Unnamed: 0' in df_raw.columns:
            df_raw = df_raw.drop('Unnamed: 0', axis=1)

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_features = df_raw[FEATURE_COLUMNS].copy()
        data_to_process = df_features.values

        print(f"Data shape: {df_raw.shape}")
        print(f"Attempting scaler load: {SCALER_PATH}")
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            data_scaled = scaler.transform(data_to_process)
            print("Scaler loaded and applied.")
        except FileNotFoundError:
            print("WARNING: scaler.pkl not found. Fitting new scaler (may reduce accuracy).")
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_to_process)

        print(f"Loading model: {MODEL_PATH}")
        t0 = time.time()
        model = keras.models.load_model(MODEL_PATH)
        print(f"Model load time: {time.time() - t0:.2f}s")

        n_sequences = len(data_scaled) - WINDOW_SIZE + 1
        if n_sequences <= 0:
            raise ValueError(f"Not enough rows ({len(data_scaled)}) for WINDOW_SIZE={WINDOW_SIZE}")

        steps = int(np.ceil(n_sequences / BATCH_SIZE))
        print(f"Predicting {n_sequences} sequences in {steps} steps...")
        t1 = time.time()
        y_proba = model.predict(sequence_generator(data_scaled, WINDOW_SIZE, BATCH_SIZE),
                                steps=steps, verbose=1).flatten()
        print(f"Inference time: {time.time() - t1:.2f}s")

        y_pred_binary = (y_proba >= BEST_THRESHOLD).astype(int)
        prediction_labels = np.where(y_pred_binary == 1, 'Fault (Panne)', 'Normal')

        start_idx = WINDOW_SIZE - 1
        datetime_end = df_raw['datetime'].iloc[start_idx:].reset_index(drop=True)
        results_df = pd.DataFrame({
            'TimeWindow_Ends_At': datetime_end.iloc[:len(y_proba)],
            'Fault_Probability': y_proba,
            'Prediction': prediction_labels
        })

        print("Analyzing fault events...")
        step_duration = df_raw['datetime'].iloc[1] - df_raw['datetime'].iloc[0]
        offset = (WINDOW_SIZE - 1) * step_duration
        results_df['TimeWindow_Start_At'] = results_df['TimeWindow_Ends_At'] - offset
        results_df = results_df[['TimeWindow_Start_At', 'TimeWindow_Ends_At', 'Fault_Probability', 'Prediction']]

        is_fault = results_df['Prediction'] == 'Fault (Panne)'
        group_change = is_fault.diff().fillna(is_fault.iloc[0])
        group_id = (group_change == True).cumsum()

        fault_events = results_df[is_fault].copy()
        if not fault_events.empty:
            fault_events['group'] = group_id[is_fault]
            fault_summary = fault_events.groupby('group').agg(
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

        print("Inference complete.")
        return {
            'logs': log_stream.getvalue(),
            'summary_data': fault_summary.to_json(orient='split', date_format='iso'),
            'detailed_data': results_df.head(100).to_json(orient='split', date_format='iso'),
            'status': 'success'
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'logs': log_stream.getvalue(),
            'status': 'error',
            'error_message': str(e)
        }
    finally:
        sys.stdout = old_stdout

# ------------------------------------------------------------
# API ENDPOINTS
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

@app.route('/api/run-model', methods=['POST'])
def run_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        f.save(path)
        results = run_inference(path)
    except Exception as e:
        results = {'logs': f"Error: {e}", 'status': 'error', 'error_message': str(e)}
    finally:
        if os.path.exists(path):
            os.remove(path)
    return jsonify(results)

# ------------------------------------------------------------
# FRONTEND (SPA) ROUTES
# ------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:subpath>')
def spa_or_asset(subpath):
    full_path = os.path.join(app.static_folder, subpath)
    if os.path.exists(full_path) and not os.path.isdir(full_path):
        return send_from_directory(app.static_folder, subpath)
    # Fallback to SPA entrypoint
    return send_from_directory(app.static_folder, 'index.html')

# ------------------------------------------------------------
# LOCAL ENTRY
# ------------------------------------------------------------
if __name__ == '__main__':
    # For local debugging (after build)
    if not os.path.isdir(DIST_DIR):
        print("WARNING: dist directory not found. Run: cd frontend && npm install && npm run build")
    app.run(host='0.0.0.0', port=8000, debug=True)