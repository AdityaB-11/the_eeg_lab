# EEG Lab Configuration

import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'
    
    # Database settings
    DATABASE_PATH = 'eeg_lab.db'
    
    # Upload settings
    UPLOAD_FOLDER = 'data/uploads'
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024 * 1024  # 200GB max file size for large EEG datasets
    ALLOWED_EXTENSIONS = {'.csv', '.edf', '.bdf', '.gdf', '.mat', '.h5', '.hdf5', '.fif', '.set', '.fdt', '.npz'}
    
    # Chunk size for large file uploads (100MB chunks)
    CHUNK_SIZE = 100 * 1024 * 1024
    
    # Jupyter settings
    JUPYTER_PORT = 8888
    JUPYTER_HOST = 'localhost'
    
    # Model settings
    MODELS_FOLDER = 'models'
    RESULTS_FOLDER = 'results'
    
    # EEG processing settings
    DEFAULT_SAMPLING_RATE = 256
    DEFAULT_CHANNELS = 19
    
    # Supported EEG formats
    SUPPORTED_FORMATS = {
        '.edf': 'European Data Format',
        '.bdf': 'BioSemi Data Format',
        '.gdf': 'General Data Format',
        '.csv': 'Comma Separated Values',
        '.mat': 'MATLAB File',
        '.h5': 'HDF5 Format',
        '.hdf5': 'HDF5 Format',
        '.fif': 'Functional Imaging Format (MNE)',
        '.set': 'EEGLAB Dataset',
        '.fdt': 'EEGLAB Data File',
        '.npz': 'NumPy Compressed Array'
    }
    
    # Default model types
    MODEL_TYPES = [
        'transformer',
        'cnn',
        'rnn',
        'lstm',
        'gru',
        'hybrid',
        'classical_ml',
        'other'
    ]
    
    # Frequency bands for EEG analysis
    FREQUENCY_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
