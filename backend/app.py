import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import io
import time 
import pickle # Added for loading the trained StandardScaler

# --- FLASK SETUP (FIXED PATHS) ---
# Calculate the absolute path to the frontend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FIX: Point directly to 'frontend' as requested, assuming index.html sits there
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend') 

# CRITICAL FIX: Use absolute path to serve the 'frontend' folder
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='/', template_folder=FRONTEND_DIR) 
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Define and create the UPLOAD_FOLDER
UPLOAD_FOLDER = 'uploads'
# Use os.path.join for platform compatibility and ensure the path is correct
UPLOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', UPLOAD_FOLDER)
os.makedirs(UPLOAD_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH

# --- MODEL PATH (Relative to the project root, where gunicorn runs) ---
MODEL_PATH = "backend/model/modele_detection_panne.keras"
# PATH FOR THE TRAINED SCALER (Assumed to be exported from training notebook)
SCALER_PATH = "backend/model/scaler.pkl" 


# --- MODEL PARAMETERS ---
BATCH_SIZE = 4096 
WINDOW_SIZE = 1000
BEST_THRESHOLD = 0.69

FEATURE_COLUMNS = [
    'FIC2003_valeur', 'LI2001_valeur', 'MX2001_valeur',
    'MX2002A_valeur', 'MX2002B_valeur', 'MX2002C_valeur',
    'MX2002D_valeur', 'MX2003A_valeur', 'MX2003B_valeur',
    'MX2003C_valeur', 'temperature_ext', 'TI2001__2_valeur'
]

# --- 2. OPTIMIZED GENERATOR FUNCTION ---
def sequence_generator(data, time_steps, batch_size):
    """Yields batches of scaled sequences, returning a tuple as required by Keras."""
    n_sequences = len(data) - time_steps + 1
    for i in range(0, n_sequences, batch_size):
        batch_end_index = min(i + batch_size, n_sequences)
        X_batch = []
        for seq_index in range(i, batch_end_index):
            X_batch.append(data[seq_index : (seq_index + time_steps)])
        yield (np.array(X_batch),) 

# --- 3. CORE INFERENCE FUNCTION ---
def run_inference(csv_path):
    # Redirect standard output to capture print logs
    old_stdout = sys.stdout
    sys.stdout = log_stream = io.StringIO()
    
    try:
        print(f"Loading data from: {csv_path}")
        
        # Load Data
        df_raw = pd.read_csv(csv_path)
        
        # ... (data processing logic is unchanged) ...
        
        if 'Unnamed: 0' in df_raw.columns:
            df_raw = df_raw.drop('Unnamed: 0', axis=1)

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_features = df_raw[FEATURE_COLUMNS].copy()
        data_to_process = df_features.values
        
        print(f"Data shape: {df_raw.shape}. Features extracted: {df_features.shape[1]}")

        # --- PROPER NORMALIZATION STEP (Loads the training-fitted scaler) ---
        print(f"Attempting to load StandardScaler from: {SCALER_PATH}")
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            
            print("Applying training-fitted StandardScaler.")
            data_scaled = scaler.transform(data_to_process)
        except FileNotFoundError:
            # Fallback for when the scaler file is missing
            print("WARNING: Scaler file not found at expected path. Falling back to temporary StandardScaler fitted on inference data.")
            print("This will lead to incorrect predictions unless the inference data perfectly matches the training data statistics.")
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_to_process)

        # Load the trained Keras model
        print(f"Loading Keras model from: {MODEL_PATH}")
        model_load_start = time.time()
        
        # Ensure model path is correct relative to the root or execution path
        # Since gunicorn runs from the root, MODEL_PATH must be relative to the root.
        model = keras.models.load_model(MODEL_PATH)
        
        model_load_end = time.time()
        print(f"Model loading time: {model_load_end - model_load_start:.2f} seconds.")
        
        n_sequences = len(data_scaled) - WINDOW_SIZE + 1
        if n_sequences <= 0:
            raise ValueError(f"Data too short ({len(data_scaled)} rows) for WINDOW_SIZE {WINDOW_SIZE}.")
            
        steps_per_epoch = int(np.ceil(n_sequences / BATCH_SIZE))
        
        # Prediction using the generator
        print(f"Starting prediction across {n_sequences} sequences in {steps_per_epoch} batches...")
        
        inference_start_time = time.time()
        
        # Adjust prediction call to unpack the generator output if necessary (already done in sequence_generator)
        y_proba = model.predict(
            sequence_generator(data_scaled, WINDOW_SIZE, BATCH_SIZE),
            steps=steps_per_epoch,
            verbose=1 
        ).flatten()

        inference_end_time = time.time()
        print(f"Prediction complete. Total inference time: {inference_end_time - inference_start_time:.2f} seconds.")

        # Interpretation
        y_pred_binary = (y_proba >= BEST_THRESHOLD).astype(int)
        prediction_labels = np.where(y_pred_binary == 1, 'Fault (Panne)', 'Normal')

        # Create Initial Results Table
        start_index = WINDOW_SIZE - 1
        datetime_end = df_raw['datetime'].iloc[start_index:].reset_index(drop=True)

        results_df = pd.DataFrame({
            'TimeWindow_Ends_At': datetime_end.iloc[:len(y_proba)],
            'Fault_Probability': y_proba,
            'Prediction': prediction_labels
        })

        # ----------------------------------------------------
        # 4. FAULT EVENT ANALYSIS LOGIC 
        # ----------------------------------------------------
        
        print("Analyzing fault events...")
        time_step_duration = df_raw['datetime'].iloc[1] - df_raw['datetime'].iloc[0]
        time_offset = (WINDOW_SIZE - 1) * time_step_duration
        results_df['TimeWindow_Start_At'] = results_df['TimeWindow_Ends_At'] - time_offset
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
            fault_summary = pd.DataFrame(columns=['Fault_Start_Time', 'Fault_End_Time', 'Max_Probability', 'Duration_Windows', 'Duration'])
        
        print("\nInference Complete. Results generated.")

        results = {
            'logs': log_stream.getvalue(),
            'summary_data': fault_summary.to_json(orient='split', date_format='iso'),
            # Limit detailed data to prevent massive JSON payloads
            'detailed_data': results_df.head(100).to_json(orient='split', date_format='iso'),
            'status': 'success'
        }
        return results

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {
            'logs': log_stream.getvalue(),
            'status': 'error',
            'error_message': str(e)
        }
    finally:
        sys.stdout = old_stdout


# --- API ENDPOINTS ---

@app.route('/api/upload-preview', methods=['POST'])
def upload_preview():
    """Handles CSV upload and returns table preview data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            # Read file stream directly to StringIO
            stream = io.StringIO(file.stream.read().decode("utf-8"))
            df = pd.read_csv(stream)
            preview_df = df.fillna('N/A')
            
            return jsonify({
                'status': 'preview_ready',
                'filename': file.filename,
                'preview_data': preview_df.to_json(orient='split', date_format='iso')
            })
        except Exception as e:
            return jsonify({'error': f"Failed to read CSV: {str(e)}"}), 500


@app.route('/api/run-model', methods=['POST'])
def run_model():
    """Handles the final file upload and runs the ML inference."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    filename = secure_filename(file.filename)
    # CRITICAL FIX: Use the UPLOAD_FOLDER from app.config
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        results = run_inference(filepath)
    except Exception as e:
        results = {
            'logs': f"Error saving or running inference: {str(e)}",
            'status': 'error',
            'error_message': str(e)
        }
    finally:
        # Ensure the temporary file is removed
        if os.path.exists(filepath):
            os.remove(filepath)
    
    return jsonify(results)


# --- ROUTE TO SERVE REACT FRONTEND (CORRECTED PATTERN) ---

# 1. Dedicated route for the root (/) to explicitly serve index.html
@app.route('/')
def index():
    """Serves the main index.html file for the root path."""
    return send_from_directory(app.static_folder, 'index.html')

# 2. Catch-all for all other paths (assets or client-side routes)
@app.route('/<path:path>')
def serve_assets(path):
    """
    Tries to serve a specific static file (CSS, JS, etc.). If the file is 
    not found, it tries a fallback location ('public/') before serving 
    index.html for client-side routing.
    """
    # Print the path Flask is looking for to help with debugging
    print(f"Attempting to serve static asset: {path} from directory: {app.static_folder}")
    
    mimetype_override = None
    # 1. MIME Type Fix (for module scripts)
    if path.endswith('.jsx') or path.endswith('.js'):
        mimetype_override = 'application/javascript'
        print(f"Applying MIME type override: {mimetype_override} for {path}")
    
    # 2. Check 1: Direct path (e.g., /src/main.jsx)
    full_path = os.path.join(app.static_folder, path)
    if os.path.exists(full_path) and not os.path.isdir(full_path):
        return send_from_directory(app.static_folder, path, mimetype=mimetype_override)
    
    # 3. Check 2: Fallback path in 'public' directory (e.g., /Plogo2.png -> /public/Plogo2.png)
    public_path = os.path.join('public', path)
    full_public_path = os.path.join(app.static_folder, public_path)
    if os.path.exists(full_public_path) and not os.path.isdir(full_public_path):
        print(f"Asset not found directly. Serving from fallback path: {public_path}")
        return send_from_directory(app.static_folder, public_path, mimetype=mimetype_override)
        
    # 4. Check 3: Not an asset, serve index.html for client-side routing (404/Not Found fallback)
    print(f"Asset {path} not found. Assuming client-side route. Serving index.html.")
    return send_from_directory(app.static_folder, 'index.html')