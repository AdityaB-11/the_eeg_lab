# EEG Lab - A Comparative Analysis and Novel Development Platform

A comprehensive Flask web application for EEG research, featuring self-supervised foundation models for universal EEG representation and disease prediction.

## Features

### 🧠 Model Management
- **Model Repository**: Store and manage trained EEG models
- **Performance Tracking**: Monitor accuracy, loss, and other metrics
- **Model Comparison**: Compare different model architectures and approaches
- **Real-time Testing**: Test models with live EEG data

### 📊 Data Repository
- **Dataset Management**: Centralized storage for EEG datasets
- **Format Support**: EDF, CSV, MAT, HDF5, FIF, SET files
- **Metadata Tracking**: Store dataset information and experimental details
- **Data Preview**: Visualize EEG signals before analysis

### 📓 Jupyter Integration
- **Interactive Notebooks**: Create and manage Jupyter notebooks
- **Template Library**: Pre-built templates for common EEG analysis tasks
- **Live Execution**: Run notebooks directly from the web interface
- **Collaborative Environment**: Share and collaborate on research

### 🧪 Testing & Validation
- **Model Testing**: Comprehensive model evaluation tools
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Confusion Matrix**: Visual representation of classification results
- **Real-time Monitoring**: Live EEG signal analysis and prediction

## Project Structure

```
the_eeg_lab/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html        # Base template with navigation
│   ├── index.html       # Home page
│   ├── models.html      # Model management
│   ├── datasets.html    # Data repository
│   ├── testing.html     # Model testing interface
│   └── notebooks.html   # Jupyter notebook management
├── data/                # Data storage
│   ├── uploads/         # Uploaded datasets
│   └── datasets/        # Processed datasets
├── notebooks/           # Jupyter notebooks
├── models/             # Trained model files
├── results/            # Test results and outputs
└── eeg_lab.db          # SQLite database
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone www.github.com/AdityaB-11/the_eeg_lab
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**:
   The database will be automatically created when you first run the application.

## Usage

1. **Start the Flask application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and navigate to `http://localhost:5000`

3. **Start Jupyter server** (optional):
   ```bash
   jupyter lab --port=8888
   ```

## Key Features Explained

### Model Management
- Upload and register new EEG models
- Track training metrics and performance
- Compare model architectures
- Export model results

### Data Repository
- Upload EEG datasets in various formats
- Automatic metadata extraction
- Dataset preview and visualization
- Bulk data management tools

### Jupyter Notebooks
- Create notebooks from templates
- EEG analysis workflows
- Model training pipelines
- Data preprocessing tools
- Visualization notebooks

### Testing Interface
- Real-time model evaluation
- Comprehensive metrics dashboard
- Confusion matrix visualization
- ROC curve analysis
- Test history tracking

## Supported EEG Formats

- **EDF/EDF+**: European Data Format
- **CSV**: Comma-separated values
- **MAT**: MATLAB files
- **HDF5**: Hierarchical Data Format
- **FIF**: Functional Imaging Format (MNE)
- **SET**: EEGLAB dataset format

## Research Applications

This platform is designed for:

1. **Self-supervised Learning**: Develop foundation models for EEG representation
2. **Disease Prediction**: Create models for neurological condition detection
3. **Comparative Analysis**: Benchmark different model architectures
4. **Data Exploration**: Visualize and analyze EEG datasets
5. **Collaborative Research**: Share notebooks and results with team members

## API Endpoints

- `GET /` - Home page
- `GET /models` - Model repository
- `GET /datasets` - Data repository
- `GET /testing` - Model testing interface
- `GET /notebooks` - Jupyter notebook management
- `POST /upload_dataset` - Upload new dataset
- `POST /add_model` - Register new model
- `POST /create_notebook` - Create new notebook
- `GET /api/model_results/<id>` - Get model details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue on the repository or contact the development team.

---

**EEG Lab** - Advancing EEG research through innovative computational tools and collaborative platforms.
