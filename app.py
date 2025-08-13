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
    
    # Create citations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            year INTEGER NOT NULL,
            journal TEXT,
            category TEXT NOT NULL,
            doi TEXT,
            url TEXT,
            abstract TEXT,
            notes TEXT,
            tags TEXT,
            relevance_score INTEGER,
            citations_count INTEGER,
            bookmarked BOOLEAN DEFAULT FALSE,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

@app.route('/api/stats')
def get_dashboard_stats():
    """Get real statistics for dashboard"""
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    
    # Count models
    cursor.execute('SELECT COUNT(*) FROM models')
    models_count = cursor.fetchone()[0]
    
    # Count datasets
    cursor.execute('SELECT COUNT(*) FROM datasets')
    datasets_count = cursor.fetchone()[0]
    
    # Count notebooks (check if notebooks directory exists and count .ipynb files)
    notebooks_count = 0
    if os.path.exists('notebooks'):
        notebooks_count = len([f for f in os.listdir('notebooks') if f.endswith('.ipynb')])
    
    # Get best accuracy from models
    cursor.execute('SELECT MAX(accuracy) FROM models WHERE accuracy IS NOT NULL')
    best_accuracy_result = cursor.fetchone()[0]
    best_accuracy = best_accuracy_result if best_accuracy_result is not None else None
    
    conn.close()
    
    return jsonify({
        'models_count': models_count,
        'datasets_count': datasets_count,
        'notebooks_count': notebooks_count,
        'best_accuracy': best_accuracy
    })

@app.route('/api/recent_models')
def get_recent_models():
    """Get recent models for dashboard"""
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    
    # Get last 3 models
    cursor.execute('SELECT name, type, accuracy FROM models ORDER BY created_at DESC LIMIT 3')
    models_data = cursor.fetchall()
    conn.close()
    
    models = []
    for model in models_data:
        models.append({
            'name': model[0],
            'type': model[1],
            'accuracy': model[2]
        })
    
    return jsonify({'models': models})

@app.route('/api/citation_stats')
def get_citation_stats():
    """Get real citation statistics"""
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    
    # Total citations
    cursor.execute('SELECT COUNT(*) FROM citations')
    total_citations = cursor.fetchone()[0]
    
    # Recent papers (2024)
    cursor.execute('SELECT COUNT(*) FROM citations WHERE year = 2024')
    recent_papers_2024 = cursor.fetchone()[0]
    
    # Count unique categories actually used
    cursor.execute('SELECT COUNT(DISTINCT category) FROM citations')
    categories_used = cursor.fetchone()[0]
    
    # Bookmarked papers
    cursor.execute('SELECT COUNT(*) FROM citations WHERE bookmarked = 1')
    bookmarked_count = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_citations': total_citations,
        'recent_papers_2024': recent_papers_2024,
        'categories_used': categories_used,
        'bookmarked_count': bookmarked_count
    })

# Citations routes
@app.route('/citations')
def citations():
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM citations ORDER BY date_added DESC')
    citations = cursor.fetchall()
    conn.close()
    
    return render_template('citations.html', citations=citations)

@app.route('/add_citation', methods=['POST'])
def add_citation():
    data = request.get_json()
    
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO citations (title, authors, year, journal, category, doi, url, 
                             abstract, notes, tags, relevance_score, citations_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['title'], data['authors'], data['year'], data.get('journal'),
        data['category'], data.get('doi'), data.get('url'), data.get('abstract'),
        data.get('notes'), data.get('tags'), data.get('relevance_score'),
        data.get('citations_count', 0)
    ))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Citation added successfully'})

@app.route('/fetch_citation_from_url', methods=['POST'])
def fetch_citation_from_url():
    """Fetch citation data from URL by extracting metadata"""
    from utils.url_extractor import extract_citation_from_url
    
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'success': False, 'message': 'URL is required'})
    
    try:
        citation_data = extract_citation_from_url(url)
        
        if citation_data:
            return jsonify({'success': True, 'citation': citation_data})
        else:
            return jsonify({'success': False, 'message': 'Could not extract citation data from URL'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error extracting citation: {str(e)}'})

@app.route('/fetch_citation_from_doi', methods=['POST'])
def fetch_citation_from_doi():
    """Fetch citation data from DOI using CrossRef API"""
    import requests
    
    data = request.get_json()
    doi = data.get('doi')
    
    if not doi:
        return jsonify({'success': False, 'message': 'DOI is required'})
    
    try:
        # Use CrossRef API to fetch citation data
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, headers={'Accept': 'application/json'})
        
        if response.status_code == 200:
            work_data = response.json()['message']
            
            # Extract citation information
            title = work_data.get('title', [''])[0] if work_data.get('title') else ''
            authors = []
            if 'author' in work_data:
                for author in work_data['author']:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    authors.append(f"{family}, {given}" if family and given else family or given)
            
            authors_str = ', '.join(authors) if authors else ''
            
            year = None
            if 'published-print' in work_data:
                year = work_data['published-print']['date-parts'][0][0]
            elif 'published-online' in work_data:
                year = work_data['published-online']['date-parts'][0][0]
            
            journal = ''
            if 'container-title' in work_data and work_data['container-title']:
                journal = work_data['container-title'][0]
            
            abstract = work_data.get('abstract', '')
            url = work_data.get('URL', '')
            
            citation_data = {
                'title': title,
                'authors': authors_str,
                'year': year,
                'journal': journal,
                'abstract': abstract,
                'url': url
            }
            
            return jsonify({'success': True, 'citation': citation_data})
        else:
            return jsonify({'success': False, 'message': 'DOI not found'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error fetching citation: {str(e)}'})

@app.route('/delete_citation/<int:citation_id>', methods=['DELETE'])
def delete_citation(citation_id):
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM citations WHERE id = ?', (citation_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Citation deleted successfully'})

@app.route('/bookmark_citation/<int:citation_id>', methods=['POST'])
def bookmark_citation(citation_id):
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE citations SET bookmarked = NOT bookmarked WHERE id = ?', (citation_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Citation bookmark toggled'})

@app.route('/get_citation_formatted/<int:citation_id>')
def get_citation_formatted(citation_id):
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM citations WHERE id = ?', (citation_id,))
    citation = cursor.fetchone()
    conn.close()
    
    if citation:
        # Format citation in APA style
        formatted = f"{citation[2]} ({citation[3]}). {citation[1]}. "
        if citation[4]:  # journal
            formatted += f"{citation[4]}. "
        if citation[6]:  # doi
            formatted += f"doi:{citation[6]}"
        
        return jsonify({'success': True, 'citation': formatted})
    else:
        return jsonify({'success': False, 'message': 'Citation not found'})

@app.route('/export_citation/<int:citation_id>')
def export_citation(citation_id):
    format_type = request.args.get('format', 'bibtex')
    
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM citations WHERE id = ?', (citation_id,))
    citation = cursor.fetchone()
    conn.close()
    
    if not citation:
        return jsonify({'error': 'Citation not found'}), 404
    
    if format_type == 'bibtex':
        # Generate BibTeX format
        bibtex = f"""@article{{{citation[1].replace(' ', '').lower()}{citation[3]},
  title={{{citation[1]}}},
  author={{{citation[2]}}},
  year={{{citation[3]}}},"""
        
        if citation[4]:  # journal
            bibtex += f"\n  journal={{{citation[4]}}},"
        if citation[6]:  # doi
            bibtex += f"\n  doi={{{citation[6]}}},"
        if citation[7]:  # url
            bibtex += f"\n  url={{{citation[7]}}},"
        
        bibtex += "\n}"
        
        response = app.response_class(
            response=bibtex,
            status=200,
            mimetype='text/plain'
        )
        response.headers['Content-Disposition'] = f'attachment; filename=citation_{citation_id}.bib'
        return response
    
    return jsonify({'error': 'Unsupported format'}), 400

@app.route('/export_bibliography')
def export_bibliography():
    format_type = request.args.get('format', 'bibtex')
    
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM citations ORDER BY year DESC, authors ASC')
    citations = cursor.fetchall()
    conn.close()
    
    if format_type == 'bibtex':
        bibliography = ""
        for citation in citations:
            bibliography += f"""@article{{{citation[1].replace(' ', '').lower()}{citation[3]},
  title={{{citation[1]}}},
  author={{{citation[2]}}},
  year={{{citation[3]}}},"""
            
            if citation[4]:  # journal
                bibliography += f"\n  journal={{{citation[4]}}},"
            if citation[6]:  # doi
                bibliography += f"\n  doi={{{citation[6]}}},"
            if citation[7]:  # url
                bibliography += f"\n  url={{{citation[7]}}},"
            
            bibliography += "\n}\n\n"
        
        response = app.response_class(
            response=bibliography,
            status=200,
            mimetype='text/plain'
        )
        response.headers['Content-Disposition'] = 'attachment; filename=bibliography.bib'
        return response
    
    return jsonify({'error': 'Unsupported format'}), 400

@app.route('/generate_citation_report')
def generate_citation_report():
    conn = sqlite3.connect('eeg_lab.db')
    cursor = conn.cursor()
    
    # Get citation statistics
    cursor.execute('SELECT COUNT(*) FROM citations')
    total_citations = cursor.fetchone()[0]
    
    cursor.execute('SELECT category, COUNT(*) FROM citations GROUP BY category')
    category_stats = cursor.fetchall()
    
    cursor.execute('SELECT year, COUNT(*) FROM citations GROUP BY year ORDER BY year DESC')
    year_stats = cursor.fetchall()
    
    cursor.execute('SELECT COUNT(*) FROM citations WHERE bookmarked = 1')
    bookmarked_count = cursor.fetchone()[0]
    
    conn.close()
    
    # Generate HTML report
    report_html = f"""
    <html>
    <head>
        <title>EEG Lab Citation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 30px 0; }}
            .stat-box {{ text-align: center; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .chart {{ margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>EEG Lab Citation Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>{total_citations}</h3>
                <p>Total Citations</p>
            </div>
            <div class="stat-box">
                <h3>{len(category_stats)}</h3>
                <p>Categories</p>
            </div>
            <div class="stat-box">
                <h3>{bookmarked_count}</h3>
                <p>Bookmarked</p>
            </div>
        </div>
        
        <h2>Citations by Category</h2>
        <table>
            <tr><th>Category</th><th>Count</th></tr>
    """
    
    for category, count in category_stats:
        report_html += f"<tr><td>{category.replace('_', ' ').title()}</td><td>{count}</td></tr>"
    
    report_html += """
        </table>
        
        <h2>Citations by Year</h2>
        <table>
            <tr><th>Year</th><th>Count</th></tr>
    """
    
    for year, count in year_stats:
        report_html += f"<tr><td>{year}</td><td>{count}</td></tr>"
    
    report_html += """
        </table>
    </body>
    </html>
    """
    
    response = app.response_class(
        response=report_html,
        status=200,
        mimetype='text/html'
    )
    response.headers['Content-Disposition'] = 'attachment; filename=citation_report.html'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
