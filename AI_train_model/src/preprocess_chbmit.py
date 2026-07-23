import os
import re
import numpy as np
import mne
import yaml

# Standard 23 channels commonly present in CHB-MIT Scalp EEG database
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'T7-FT9', 'FT9-FT10',
    'FT10-T8', 'P7-T7', 'T8-DP1' # standard fallback channels to make 23
]

def parse_seizure_annotations(summary_filepath):
    """
    Parses a CHB-MIT summary text file to find seizure start and end times for each EDF file.
    Returns a dictionary: { edf_filename: [(start_sec, end_sec), ...] }
    """
    annotations = {}
    if not os.path.exists(summary_filepath):
        print(f"Warning: Summary file not found: {summary_filepath}")
        return annotations
        
    current_file = None
    seizure_count = 0
    seizure_starts = []
    seizure_ends = []
    
    with open(summary_filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Parse file name
        if line.startswith("File Name:"):
            current_file = line.split(":")[-1].strip()
            seizure_count = 0
            seizure_starts = []
            seizure_ends = []
            annotations[current_file] = []
        # Parse seizure count
        elif line.startswith("Number of Seizures in File:"):
            seizure_count = int(line.split(":")[-1].strip())
        # Parse seizure start time
        elif "Start Time" in line and current_file:
            match = re.search(r"Seizure\s+\d+\s+Start\s+Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)
            if match:
                seizure_starts.append(int(match.group(1)))
        # Parse seizure end time
        elif "End Time" in line and current_file:
            match = re.search(r"Seizure\s+\d+\s+End\s+Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)
            if match:
                seizure_ends.append(int(match.group(1)))
                # If we've got both start and end, add to dictionary
                if len(seizure_starts) == len(seizure_ends):
                    annotations[current_file].append((seizure_starts[-1], seizure_ends[-1]))
                    
    # Clean up empty files
    annotations = {k: v for k, v in annotations.items() if v or len(v) == 0}
    return annotations

def segment_edf_file(edf_path, seizure_windows, sample_rate=256, window_sec=1):
    """
    Reads an EDF file using MNE, selects standard channels, and slices it into
    1-second seizure (ictal) and non-seizure (interictal) segments.
    Returns: X_seizure, X_normal (arrays of shape (num_segments, 23, 256))
    """
    try:
        # Load raw EDF. Use preload=False to save memory
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error reading EDF file {edf_path}: {e}")
        return None, None
        
    # Check sampling frequency and resample if necessary
    sfreq = raw.info['sfreq']
    if int(sfreq) != sample_rate:
        raw.resample(sample_rate, verbose=False)
        
    # Clean channel names to match standard list (case-insensitive, remove spaces)
    raw_channels = raw.ch_names
    channel_mapping = {}
    for ch in raw_channels:
        clean_ch = ch.upper().replace(' ', '').replace('EEG', '')
        channel_mapping[ch] = clean_ch
    raw.rename_channels(channel_mapping, verbose=False)
    
    # Select available standard channels
    available_channels = [ch for ch in STANDARD_CHANNELS if ch in raw.ch_names]
    
    # If we have fewer channels, fill up with whatever channels are available to keep 23 channels
    if len(available_channels) < 23:
        extra_channels = [ch for ch in raw.ch_names if ch not in available_channels]
        available_channels.extend(extra_channels[:23 - len(available_channels)])
        
    # Ensure we select exactly 23 channels
    channels_to_select = available_channels[:23]
    if len(channels_to_select) < 23:
        print(f"Warning: File {os.path.basename(edf_path)} has only {len(channels_to_select)} channels, skipping.")
        return None, None
        
    raw.pick(channels_to_select, verbose=False)
    
    # Extract data as numpy array: shape (23, num_samples)
    data, _ = raw[:, :]
    num_samples = data.shape[1]
    
    # Define seizure masks (1 for seizure, 0 for normal)
    seizure_mask = np.zeros(num_samples, dtype=bool)
    for start_sec, end_sec in seizure_windows:
        start_idx = int(start_sec * sample_rate)
        end_idx = int(end_sec * sample_rate)
        seizure_mask[start_idx:min(end_idx, num_samples)] = True
        
    # Slice into 1-second segments (256 samples)
    step_samples = int(window_sec * sample_rate)
    
    seizure_segments = []
    normal_segments = []
    
    for idx in range(0, num_samples - step_samples, step_samples):
        segment = data[:, idx:idx + step_samples]
        segment_mask = seizure_mask[idx:idx + step_samples]
        
        # If the entire segment is seizure, label as seizure
        if np.all(segment_mask):
            seizure_segments.append(segment)
        # If the entire segment is normal, label as normal
        elif np.all(~segment_mask):
            normal_segments.append(segment)
            
    return np.array(seizure_segments), np.array(normal_segments)

def preprocess_subject(subject_dir, sample_rate=256, window_sec=1):
    """
    Preprocesses all EDF files for a given subject (e.g. chb01).
    Reads the summary text file and segments all EDF recordings.
    """
    subject_name = os.path.basename(os.path.normpath(subject_dir))
    summary_file = os.path.join(subject_dir, f"{subject_name}-summary.txt")
    
    # Parse seizure times
    seizures_dict = parse_seizure_annotations(summary_file)
    print(f"Found {len(seizures_dict)} files with seizure annotations in summary.")
    
    all_seizure = []
    all_normal = []
    
    # List EDF files
    edf_files = [f for f in os.listdir(subject_dir) if f.endswith(".edf")]
    print(f"Processing {len(edf_files)} EDF files for subject {subject_name}...")
    
    for filename in edf_files:
        edf_path = os.path.join(subject_dir, filename)
        # Seizure windows for this specific file
        seizure_windows = seizures_dict.get(filename, [])
        
        x_seizure, x_normal = segment_edf_file(edf_path, seizure_windows, sample_rate, window_sec)
        
        if x_seizure is not None and len(x_seizure) > 0:
            all_seizure.append(x_seizure)
        if x_normal is not None and len(x_normal) > 0:
            # Subsample normal segments to avoid memory issues (max 500 normal segments per file)
            indices = np.random.choice(len(x_normal), min(len(x_normal), 500), replace=False)
            all_normal.append(x_normal[indices])
            
    if all_seizure:
        X_seiz = np.concatenate(all_seizure, axis=0)
    else:
        X_seiz = np.empty((0, 23, int(window_sec * sample_rate)))
        
    if all_normal:
        X_norm = np.concatenate(all_normal, axis=0)
    else:
        X_norm = np.empty((0, 23, int(window_sec * sample_rate)))
        
    print(f"Subject {subject_name} Summary: Seizure segments = {X_seiz.shape[0]} | Normal segments = {X_norm.shape[0]}")
    return X_seiz, X_norm

def run_chbmit_preprocessing(raw_dataset_dir, output_path, sample_rate=256, window_sec=1):
    """
    Orchestrates the preprocessing across multiple subjects in the CHB-MIT dataset directory,
    balances seizure/non-seizure segments, and saves the final npz dataset.
    """
    print(f"Starting CHB-MIT Preprocessing on database folder: {raw_dataset_dir}")
    if not os.path.exists(raw_dataset_dir):
        print(f"Error: Raw dataset folder not found at {raw_dataset_dir}. Please download it or adjust raw_dir in config.yaml.")
        return False
        
    # We will scan the subject folders (e.g. chb01, chb02, chb03, chb04, chb05)
    # To keep preprocessing manageable, we'll scan the first 5 subjects or all available
    subject_folders = [d for d in os.listdir(raw_dataset_dir) if os.path.isdir(os.path.join(raw_dataset_dir, d)) and d.startswith("chb")]
    subject_folders.sort()
    
    if len(subject_folders) == 0:
        print(f"Error: No subject folders (chbXX) found in {raw_dataset_dir}.")
        return False
        
    # Let's preprocess the first 3-5 subjects for training. We can increase this for full scale.
    subjects_to_process = subject_folders[:5] 
    print(f"Found {len(subject_folders)} subject directories. Restricting preprocessing to: {subjects_to_process}")
    
    seizure_list = []
    normal_list = []
    
    for subject_name in subjects_to_process:
        subject_dir = os.path.join(raw_dataset_dir, subject_name)
        x_seiz, x_norm = preprocess_subject(subject_dir, sample_rate, window_sec)
        
        if len(x_seiz) > 0:
            seizure_list.append(x_seiz)
        if len(x_norm) > 0:
            normal_list.append(x_norm)
            
    if not seizure_list:
        print("Error: No seizure segments could be extracted. Check annotations.")
        return False
        
    X_seizure = np.concatenate(seizure_list, axis=0)
    X_normal_all = np.concatenate(normal_list, axis=0)
    
    num_seizure = X_seizure.shape[0]
    print(f"\nTotal extracted Seizure segments: {num_seizure}")
    print(f"Total extracted Normal segments:  {X_normal_all.shape[0]}")
    
    # Balance dataset: Randomly select normal segments to match seizure count
    print("Balancing datasets (1:1 ratio)...")
    indices = np.random.choice(X_normal_all.shape[0], num_seizure, replace=False)
    X_normal = X_normal_all[indices]
    
    # Combine and shuffle
    X = np.concatenate([X_seizure, X_normal], axis=0)
    y = np.concatenate([np.ones(num_seizure), np.zeros(num_seizure)], axis=0)
    
    # Shuffle datasets
    shuffle_idx = np.random.permutation(X.shape[0])
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Ensure target output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save preprocessed dataset
    np.savez_compressed(output_path, X=X, y=y, channels=STANDARD_CHANNELS)
    print(f"\nSaved balanced preprocessed dataset to {output_path}")
    print(f"Dataset Dimensions: X shape: {X.shape} | y shape: {y.shape}")
    print(f"Successfully preprocessed {X.shape[0]} total balanced 1-second segments.")
    return True
