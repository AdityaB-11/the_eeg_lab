from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import json
import subprocess
from datetime import datetime
import sqlite3
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure required directories exist
os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/datasets', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    
    # Create models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            accuracy REAL,
            loss REAL,
            dataset TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_path TEXT,
            description TEXT
        )
    ''')
    
    # Create datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            size INTEGER,
            description TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models ORDER BY created_at DESC')
    models_data = cursor.fetchall()
    conn.close()
    return render_template('models.html', models=models_data)

@app.route('/datasets')
def datasets():
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM datasets ORDER BY uploaded_at DESC')
    datasets_data = cursor.fetchall()
    conn.close()
    return render_template('datasets.html', datasets=datasets_data)

@app.route('/testing')
def testing():
    # Get available models for testing
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, type, accuracy FROM models')
    available_models = cursor.fetchall()
    conn.close()
    return render_template('testing.html', models=available_models)

@app.route('/notebooks')
def notebooks():
    # List available Jupyter notebooks
    notebook_files = []
    notebooks_dir = 'notebooks'
    if os.path.exists(notebooks_dir):
        for file in os.listdir(notebooks_dir):
            if file.endswith('.ipynb'):
                notebook_files.append(file)
    return render_template('notebooks.html', notebooks=notebook_files)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Save to database
        conn = sqlite3.connect('eeg_lab.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO datasets (name, file_path, size, description)
            VALUES (?, ?, ?, ?)
        ''', (filename, file_path, file_size, request.form.get('description', '')))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Dataset uploaded successfully'})

@app.route('/add_model', methods=['POST'])
def add_model():
    data = request.json
    
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO models (name, type, accuracy, loss, dataset, description, file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['name'],
        data['type'],
        data.get('accuracy'),
        data.get('loss'),
        data.get('dataset'),
        data.get('description'),
        data.get('file_path')
    ))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Model added successfully'})

@app.route('/open_notebook/<notebook_name>')
def open_notebook(notebook_name):
    # This will attempt to open Jupyter notebook
    notebook_path = os.path.join('notebooks', notebook_name)
    if os.path.exists(notebook_path):
        try:
            # Try to start jupyter notebook server
            subprocess.Popen(['jupyter', 'notebook', notebook_path])
            return jsonify({'success': True, 'message': f'Opening {notebook_name}'})
        except Exception as e:
            return jsonify({'error': f'Could not open notebook: {str(e)}'})
    else:
        return jsonify({'error': 'Notebook not found'}), 404

@app.route('/create_notebook', methods=['POST'])
def create_notebook():
    notebook_name = request.json.get('name')
    if not notebook_name.endswith('.ipynb'):
        notebook_name += '.ipynb'
    
    notebook_path = os.path.join('notebooks', notebook_name)
    
    # Create a basic notebook structure
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# EEG Analysis Notebook\n",
                    "\n",
                    "This notebook is for EEG data analysis and model development.\n",
                    "\n",
                    "## Import Required Libraries"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.metrics import accuracy_score, classification_report\n",
                    "\n",
                    "# EEG specific libraries\n",
                    "import mne\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.optim as optim\n",
                    "\n",
                    "print(\"Libraries imported successfully!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    return jsonify({'success': True, 'message': f'Notebook {notebook_name} created successfully'})

@app.route('/api/model_results/<int:model_id>')
def get_model_results(model_id):
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models WHERE id = ?', (model_id,))
    model_data = cursor.fetchone()
    conn.close()
    
    if model_data:
        return jsonify({
            'id': model_data[0],
            'name': model_data[1],
            'type': model_data[2],
            'accuracy': model_data[3],
            'loss': model_data[4],
            'dataset': model_data[5],
            'created_at': model_data[6]
        })
    else:
        return jsonify({'error': 'Model not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
