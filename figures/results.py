import librosa
import numpy as np
import re
import os
import matplotlib as mpl
import g2p_en

def extract_id(filename):
    """ Extracts the ID from a filename. """
    parts = filename.split('_')
    if parts[-1].endswith(('.txt', '.npy', '.wav')):
        return '_'.join(parts[:-1]), parts[-1].split('.')[0]
        
def load_data(file_path, file_type):
    """ Loads data from a file. """
    if file_type.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.read()
    elif file_type.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    elif file_type.endswith('.wav'):
        return librosa.load(file_path, sr=None)
    return None

def clean_string(text):
    "lowercase, remove quotes, remove punctuation, and strip leading and trailing whitespace"
    text = text.lower().replace("n/a", "").replace("inaudible", "").replace("[", "").replace("]", "")
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = text.replace("'", "").replace('"', '')
    return text

def unstress_phonemes(phonemes):
    """
    Removes numbers (stress markers) from each phoneme in a string of phonemes.
    """
    # Use a regular expression to remove digits from each phoneme
    return re.sub(r'\d', '', clean_string(phonemes))

def convert_to_unstressed_phonemes(sentence):
    """Converts a sentence into its unstressed phonemes"""
    converter = g2p_en.G2p()
    phonemes = converter(sentence)
    unstressed_phonemes = [phoneme.rstrip("0123456789") for phoneme in phonemes]
    return unstressed_phonemes

def process_result_directories(base_dirs, exclude_entry_prefixes=[]):
    """ Loads data from a set of directories.
    
    Example usage:
    base_dirs = {
        'data1': 'path/to/data1',
        'data2': 'path/to/data2'
    }
    
    Returns a dictionary with the following structure:
    {
        'data1': {
            'modality1': {
                'data_type1': {
                    'id1': data1,
                    'id2': data2,
                    ...
                },
                'data_type2': {
                    'id1': data1,
                    'id2': data2,
                    ...
                },
                ...
            },
            'modality2': {
                ...
            },
            ...
        },
        'data2': {
            ...
        },
        ...
    }
    """
    # Initialize results dictionary
    results = {}

    # Process each base directory
    for base_name, data_dir in base_dirs.items():
        results[base_name] = {}
        
        # Iterate through the folders (modalities or experiments)
        for folder in os.listdir(data_dir):
            if "ablat" in folder.lower() or 's24' in folder.lower() or 'demoday' in folder.lower() or 'heldout' in folder.lower() or 'mask' in folder.lower() or "crop" in folder.lower():
                continue
            print("Processing ", folder)
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue  # Skip if not a directory
            
            # Initialize data structure for the folder
            folder_key = folder.replace(' ', '_').lower()  # Normalize folder name
            results[base_name][folder_key] = {}
            
            # Iterate through data types (subdirectories)
            for data_type in os.listdir(folder_path):
                if "ckpts" in data_type:
                    continue
                if any(data_type.startswith(prefix) for prefix in exclude_entry_prefixes):
                    continue
                data_type_path = os.path.join(folder_path, data_type)
                if not os.path.isdir(data_type_path):
                    continue  # Skip if not a directory
                
                # Initialize data structure for the data type
                data_type_key = data_type.replace(' ', '_').lower()  # Normalize data type name
                results[base_name][folder_key][data_type_key] = {}
                
                # Iterate through files
                for filename in os.listdir(data_type_path):
                    file_path = os.path.join(data_type_path, filename)
                    _, file_id = extract_id(filename)  # Includes modality prefix for rnnt results
                    
                    # Load and store data based on its type
                    data = load_data(file_path, filename)
                    if data is not None:
                        # Accommodate for audio files returning a tuple (audio, rate)
                        if isinstance(data, tuple):
                            data, _ = data
                        if "asr_transcript" in data_type_key:
                            data = clean_string(str(data))
                        results[base_name][folder_key][data_type_key][file_id] = data
    return results 


def plotting_defaults(font='Arial', fontsize=7, linewidth=1):
    """ Sets plotting settings for our figures. """
    mpl.rcParams.update({'font.size': fontsize})
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [font]
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['xtick.major.width'] = linewidth
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['ytick.major.width'] = linewidth
    mpl.rcParams['lines.linewidth'] = linewidth
    mpl.rcParams['axes.linewidth'] = linewidth

    
def create_custom_subplots(fig, positions, titles, panel_letters):
    """
    Creates custom subplots within a given figure.

    Parameters:
    - fig: matplotlib.figure.Figure, the figure object in which to create the subplots.
    - positions: list, a list of positions where each subplot will be placed within the figure.
    - titles: list, titles for each subplot.
    - panel_letters: list, a list of letters used to label each subplot.

    Returns:
    - dict: A dictionary mapping panel letters to subplot axes.
    """
    # Initialize dictionary to store axis objects keyed by panel letters
    fig_axes = {}

    # Loop through and create each subplot with adjusted positions
    for i, pos in enumerate(positions):
        ax = fig.add_subplot(pos)
        ax.set_title(titles[i])
        ax.text(-0.1, 1.15, f"{panel_letters[i]}", transform=ax.transAxes, fontsize=8, va='top', ha='right', fontdict={'weight': 'bold'})
        ax.grid(False)  # Disabling the grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Store the axis in the dictionary
        fig_axes[panel_letters[i]] = ax
    
    return fig_axes