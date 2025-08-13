"""
Utilities for handling large EEG datasets (1GB - 150GB).

This module provides functions for efficient processing of large EEG files,
including memory-mapped loading, chunked processing, and progress tracking.
"""

import os
import h5py
import numpy as np
from typing import Generator, Tuple, Optional
import mmap
import gc
from pathlib import Path

class LargeDatasetHandler:
    """Handler for processing large EEG datasets efficiently."""
    
    def __init__(self, chunk_size_mb: int = 100):
        """
        Initialize the handler.
        
        Args:
            chunk_size_mb: Size of processing chunks in MB
        """
        self.chunk_size = chunk_size_mb * 1024 * 1024  # Convert to bytes
        
    def get_file_size_gb(self, filepath: str) -> float:
        """Get file size in gigabytes."""
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 ** 3)
    
    def estimate_memory_usage(self, filepath: str, dtype=np.float64) -> float:
        """
        Estimate memory usage for loading the entire file.
        
        Returns:
            Estimated memory usage in GB
        """
        file_size_bytes = os.path.getsize(filepath)
        # Assume the file contains mostly numeric data
        if filepath.endswith(('.h5', '.hdf5')):
            # HDF5 files are compressed, estimate 2-4x expansion
            estimated_memory = file_size_bytes * 3
        elif filepath.endswith('.npz'):
            # NPZ files are compressed, estimate 3-5x expansion
            estimated_memory = file_size_bytes * 4
        else:
            # For other formats, assume minimal compression
            estimated_memory = file_size_bytes * 1.5
            
        return estimated_memory / (1024 ** 3)  # Convert to GB
    
    def memory_mapped_load(self, filepath: str, shape: Tuple[int, ...], 
                          dtype=np.float64, mode='r') -> np.memmap:
        """
        Create a memory-mapped array for large files.
        
        Args:
            filepath: Path to the file
            shape: Shape of the data array
            dtype: Data type
            mode: File access mode ('r', 'r+', 'w+', 'c')
        
        Returns:
            Memory-mapped array
        """
        return np.memmap(filepath, dtype=dtype, mode=mode, shape=shape)
    
    def chunked_hdf5_reader(self, filepath: str, dataset_name: str = 'data',
                           chunk_size: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """
        Read HDF5 file in chunks to manage memory usage.
        
        Args:
            filepath: Path to HDF5 file
            dataset_name: Name of the dataset within the HDF5 file
            chunk_size: Number of samples per chunk (if None, uses default)
        
        Yields:
            Chunks of data as numpy arrays
        """
        if chunk_size is None:
            chunk_size = self.chunk_size // 8  # Assume float64 (8 bytes per element)
            
        with h5py.File(filepath, 'r') as f:
            dataset = f[dataset_name]
            total_samples = dataset.shape[0]
            
            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                yield dataset[start_idx:end_idx]
    
    def process_large_edf(self, filepath: str, 
                         processing_func=None) -> Generator[np.ndarray, None, None]:
        """
        Process large EDF files in chunks.
        
        Args:
            filepath: Path to EDF file
            processing_func: Function to apply to each chunk
        
        Yields:
            Processed chunks of EEG data
        """
        try:
            import pyedflib
        except ImportError:
            raise ImportError("pyedflib is required for EDF file processing")
        
        with pyedflib.EdfReader(filepath) as f:
            n_channels = f.signals_in_file
            sample_frequency = f.getSampleFrequencies()[0]  # Assume same for all channels
            
            # Calculate chunk size in samples
            chunk_samples = int(self.chunk_size / (n_channels * 8))  # 8 bytes per float64
            
            # Read signal lengths
            signal_lengths = [f.getNSamples()[i] for i in range(n_channels)]
            max_length = max(signal_lengths)
            
            for start_sample in range(0, max_length, chunk_samples):
                end_sample = min(start_sample + chunk_samples, max_length)
                
                # Read chunk from all channels
                chunk_data = np.zeros((n_channels, end_sample - start_sample))
                for ch in range(n_channels):
                    if end_sample <= signal_lengths[ch]:
                        chunk_data[ch, :] = f.readSignal(ch)[start_sample:end_sample]
                
                if processing_func:
                    chunk_data = processing_func(chunk_data)
                
                yield chunk_data
                
                # Force garbage collection to manage memory
                gc.collect()
    
    def estimate_processing_time(self, file_size_gb: float, 
                                processing_speed_gbps: float = 0.1) -> float:
        """
        Estimate processing time for a large dataset.
        
        Args:
            file_size_gb: File size in GB
            processing_speed_gbps: Processing speed in GB per second
        
        Returns:
            Estimated processing time in seconds
        """
        return file_size_gb / processing_speed_gbps
    
    def create_progress_tracker(self, total_size: int, description: str = "Processing"):
        """
        Create a simple progress tracker for large file processing.
        
        Args:
            total_size: Total size to process
            description: Description of the process
        """
        processed = 0
        
        def update_progress(chunk_size: int):
            nonlocal processed
            processed += chunk_size
            percentage = (processed / total_size) * 100
            print(f"\r{description}: {percentage:.1f}% complete", end='', flush=True)
            
        return update_progress
    
    def optimize_chunk_size(self, available_memory_gb: float, 
                           file_size_gb: float, n_channels: int) -> int:
        """
        Optimize chunk size based on available memory and file characteristics.
        
        Args:
            available_memory_gb: Available RAM in GB
            file_size_gb: File size in GB
            n_channels: Number of EEG channels
        
        Returns:
            Optimized chunk size in samples
        """
        # Use up to 50% of available memory for processing
        max_memory_bytes = available_memory_gb * 0.5 * (1024 ** 3)
        
        # Account for channel count and data type (float64 = 8 bytes)
        bytes_per_sample = n_channels * 8
        
        optimal_chunk_samples = int(max_memory_bytes / bytes_per_sample)
        
        # Ensure minimum chunk size for efficiency
        min_chunk_samples = 1000  # At least 1000 samples per chunk
        
        return max(optimal_chunk_samples, min_chunk_samples)


def create_hdf5_from_large_edf(edf_path: str, hdf5_path: str, 
                              compression: str = 'gzip', 
                              compression_level: int = 9):
    """
    Convert large EDF file to compressed HDF5 format.
    
    Args:
        edf_path: Path to input EDF file
        hdf5_path: Path to output HDF5 file
        compression: Compression algorithm ('gzip', 'lzf', 'szip')
        compression_level: Compression level (0-9 for gzip)
    """
    handler = LargeDatasetHandler()
    
    print(f"Converting {edf_path} to HDF5 format...")
    print(f"File size: {handler.get_file_size_gb(edf_path):.2f} GB")
    
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib is required for EDF file processing")
    
    # Get file info
    with pyedflib.EdfReader(edf_path) as edf:
        n_channels = edf.signals_in_file
        sample_frequencies = edf.getSampleFrequencies()
        signal_labels = edf.getSignalLabels()
        signal_lengths = [edf.getNSamples()[i] for i in range(n_channels)]
        max_length = max(signal_lengths)
    
    # Create HDF5 file
    with h5py.File(hdf5_path, 'w') as hdf5:
        # Create dataset with compression
        dataset = hdf5.create_dataset(
            'eeg_data', 
            shape=(n_channels, max_length),
            dtype=np.float64,
            compression=compression,
            compression_opts=compression_level,
            chunks=True  # Enable chunking for better compression
        )
        
        # Store metadata
        hdf5.attrs['n_channels'] = n_channels
        hdf5.attrs['sample_frequencies'] = sample_frequencies
        hdf5.attrs['signal_labels'] = [label.encode('utf-8') for label in signal_labels]
        hdf5.attrs['original_file'] = edf_path.encode('utf-8')
        
        # Process in chunks
        progress_tracker = handler.create_progress_tracker(max_length, "Converting")
        
        for i, chunk in enumerate(handler.process_large_edf(edf_path)):
            start_idx = i * chunk.shape[1]
            end_idx = start_idx + chunk.shape[1]
            dataset[:, start_idx:end_idx] = chunk
            progress_tracker(chunk.shape[1])
    
    print(f"\nConversion complete! HDF5 file saved to {hdf5_path}")
    print(f"Compressed size: {handler.get_file_size_gb(hdf5_path):.2f} GB")


if __name__ == "__main__":
    # Example usage
    handler = LargeDatasetHandler()
    
    # Example: Check file size
    test_file = "path/to/large/eeg/file.edf"
    if os.path.exists(test_file):
        size_gb = handler.get_file_size_gb(test_file)
        memory_estimate = handler.estimate_memory_usage(test_file)
        processing_time = handler.estimate_processing_time(size_gb)
        
        print(f"File size: {size_gb:.2f} GB")
        print(f"Estimated memory usage: {memory_estimate:.2f} GB")
        print(f"Estimated processing time: {processing_time:.1f} seconds")
