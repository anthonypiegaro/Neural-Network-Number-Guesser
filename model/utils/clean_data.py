import numpy as np

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data values to be between 0 and 1
    """
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_norm

def add_channel(data: np.ndarray) -> np.ndarray:
    """
    Add a channel to the data.
    
    input: (28, 28)
    output: (28, 28, 1)
    """
    reshaped_data = data.reshape((data.shape + (1,)))
    return reshaped_data

def clean_data(data: np.ndarray) -> np.ndarray:
    """
    Entire cleaning process for data.
    
    Normalize data to be in [0, 1], then
    reshape to (28, 28, 1).
    """
    normalized_data = normalize_data(data)
    fully_cleaned_data = add_channel(normalized_data)
    return fully_cleaned_data
