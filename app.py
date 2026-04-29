from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
import random
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('models/ddos_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load cleaned data for simulation
df = pd.read_parquet('data/cleaned_data.parquet')
X = df.drop('Label', axis=1)
y = df['Label']

# Store recent predictions for dashboard
recent_traffic = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_traffic')
def get_traffic():
    # Pick 10 random records from dataset
    sample = X.sample(10)
    predictions = model.predict(sample)
    
    traffic_data = []
    for i, (idx, row) in enumerate(sample.iterrows()):
        pred = int(predictions[i])
        traffic_data.append({
            'id': int(idx),
            'prediction': 'DDoS' if pred == 1 else 'Benign',
            'status': 'danger' if pred == 1 else 'success',
            'flow_duration': round(float(row['Flow Duration']), 2),
            'flow_bytes': round(float(row['Flow Bytes/s']), 2),
            'flow_packets': round(float(row['Flow Packets/s']), 2),
            'packet_length': round(float(row['Packet Length Mean']), 2),
        })
    
    # Count attacks vs benign
    attack_count = sum(1 for t in traffic_data if t['prediction'] == 'DDoS')
    benign_count = len(traffic_data) - attack_count
    
    return jsonify({
        'traffic': traffic_data,
        'attack_count': attack_count,
        'benign_count': benign_count,
        'total': len(traffic_data)
    })

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'accuracy': 99.98,
        'total_records': 680095,
        'ddos_records': 128014,
        'benign_records': 552081,
        'model': 'Random Forest (100 trees)'
    })

if __name__ == '__main__':
    app.run(debug=True)