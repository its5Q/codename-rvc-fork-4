import numpy as np
import torch
import os

def load_file(file_name):
    """Load the file based on its extension."""
    if file_name.endswith('.npy'):
        data = np.load(file_name)
        return data, 'npy'
    elif file_name.endswith('.pt'):
        data = torch.load(file_name)
        return data, 'pt'
    else:
        print(f"Unsupported file type: {file_name}")
        return None, None

def trim_frames(data, num_frames_to_trim, file_type):
    """Trim frames from the data."""
    if file_type == 'npy':
        # For numpy array (F0), we just slice it
        trimmed_data = data[:-num_frames_to_trim]
    elif file_type == 'pt':
        # For PyTorch tensor (spectrogram), we assume it's a 2D tensor (time x frequency)
        trimmed_data = data[:, :-num_frames_to_trim]
    else:
        raise ValueError("Unsupported file type for trimming.")
    
    return trimmed_data

def save_file(data, file_name, file_type):
    """Save the trimmed data back to the file or a new file."""
    if file_type == 'npy':
        np.save(file_name, data)
    elif file_type == 'pt':
        torch.save(data, file_name)
    else:
        raise ValueError("Unsupported file type for saving.")

def main():
    # Ask for the file name
    file_name = input("Enter the name of the file (e.g., file.npy or file.pt): ").strip()
    
    # Check if the file exists
    if not os.path.exists(file_name):
        print(f"File '{file_name}' not found.")
        return
    
    # Load the data from the file
    data, file_type = load_file(file_name)
    
    if data is None:
        return
    
    # Ask how many frames to trim from the end
    try:
        num_frames_to_trim = int(input("How many frames to trim from the end? "))
    except ValueError:
        print("Invalid number of frames.")
        return
    
    if num_frames_to_trim <= 0:
        print("Please provide a positive integer to trim frames.")
        return
    
    # Ensure we don't trim more frames than are available
    if num_frames_to_trim >= data.shape[-1]:
        print(f"You cannot trim more than {data.shape[-1]} frames from the end.")
        return
    
    # Trim the frames
    trimmed_data = trim_frames(data, num_frames_to_trim, file_type)
    
    # Save the trimmed data to a new file (or overwrite original)
    save_file(trimmed_data, file_name, file_type)
    
    print(f"Successfully trimmed {num_frames_to_trim} frames from the end.")
    print(f"Updated file saved as {file_name}")

if __name__ == '__main__':
    main()
