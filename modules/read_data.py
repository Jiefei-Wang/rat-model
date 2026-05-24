import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

# The force threshold that is considered a valid press
low_force_threshold = 5

# The force range (+/-) that is considered a constant force
constant_force_threshold = 2
# The minimum number of consecutive constant force values to be considered a constant run
constant_run_threshold = 50


def collapse_zeros_alike(data, threshold=1):
    """
    Given a list of numbers, replace any consecutive zeros 
    with a single zero in the output list.
    
    Example:
        [1, 0, 0, 0, 2, 0, 0, 3] --> [1, 0, 2, 0, 3]
    """
    if not data:
        return []
    
    output = []
    for value in data:
        if value <= threshold:
            # Only add this zero if the last element in output isn't zero.
            if not output or output[-1] > threshold:
                output.append(0)
        else:
            output.append(value)
    
    return output


def extract_raw_data(file_path):
    data = []
    start_processing = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

    subject_id = None
    for line in lines:
        # format: Subject: xx
        if "Subject:" in line:
            subject_id = line.split("Subject:")[1].strip()
        if "P:" in line:
            start_processing = True
            continue
        
        if start_processing:
            if len(line) > 1 and line[0].isalpha() and line[1] == ':':
                break
            if not line:
                break
            parts = line.split()
            if len(parts) > 1:
                numbers = parts[1:]
                data.extend(map(float, numbers))
    
    
    low_force_mask = [x < low_force_threshold for x in data]
    new_data = [x if x >= low_force_threshold else 0 for x in data]
    # Remove trailing zeros
    while new_data and new_data[-1] == 0:
        new_data.pop()
        low_force_mask.pop()
    
    if subject_id is None:
        raise ValueError(f"Subject ID not found in file: {file_path}")
    return subject_id, new_data, low_force_mask


def to_bar_press(data):
    bar_presses = []
    mask = [False] * len(data)
    index = []
    has_large_val = False
    found_medium_force = False  # Track if we've encountered a value between 5 and 20

    press_start = None
    for i, value in enumerate(data):
        if value != 0:
            if press_start is None:
                press_start = i
            if value >= 20:
                has_large_val = True
            if 5 <= value <= 20:
                found_medium_force = True
        elif press_start is not None:
            if has_large_val and found_medium_force:
                index.append((press_start, i))
                bar_presses.append(data[press_start:i])
                mask[press_start:i] = [True] * (i - press_start)
                
            press_start = None
            has_large_val = False
            found_medium_force = False
    ## The last bar press
    if press_start is not None and has_large_val and found_medium_force:
        index.append((press_start, len(data)))
        bar_presses.append(data[press_start:])
        mask[press_start:] = [True] * (len(data) - press_start)
    return bar_presses, mask, index


def mask_constant_values(data):
    """
    data: 1D array-like of numeric values (Python list)
    
    return: True if constant values is found, False otherwise
    """
    n = len(data)
    i = 0
    while i < n - constant_run_threshold + 1:
        end_idx = i + constant_run_threshold
        
        seq_data = data[i:end_idx]
        data_range = max(seq_data) - min(seq_data)
        if data_range <= 2 * constant_force_threshold:
            return True, [i, end_idx]
        
        i += 1  # Move to the next index and check again
        
    return False, None




def process_raw_data(raw_data):
    # To bar press data: list of lists
    bar_presses, bar_press_mask, bar_press_index = to_bar_press(raw_data)
    # If a bar press has constant values, remove it
    constant_value_masks = [mask_constant_values(press) for press in bar_presses]
    bar_presses_filtered = [press for press, mask in zip(bar_presses, constant_value_masks) if not mask[0]]
    bar_presses_filtered_index = [index for index, mask in zip(bar_press_index, constant_value_masks) if not mask[0]]
    bar_presses_constant = [press for press, mask in zip(bar_presses, constant_value_masks) if mask[0]]
    bar_presses_constant_index = [index for index, mask in zip(bar_press_index, constant_value_masks) if mask[0]]
    return bar_presses_filtered, bar_presses_filtered_index, bar_press_mask, bar_presses_constant, bar_presses_constant_index, constant_value_masks



def process_raw_datas(raw_data_list):
    dt = []
    ## progress bar
    for i in tqdm(range(len(raw_data_list))):
        raw_data = raw_data_list[i]
        ## file base name with no folder
        bar_presses_filtered, bar_presses_filtered_index, bar_press_mask, bar_presses_constant, bar_presses_constant_index, constant_value_masks = process_raw_data(raw_data)
        
        ## combine the data
        rat = {
            'data': bar_presses_filtered,
            'data_index': bar_presses_filtered_index,
            'raw_data': raw_data,
            'bar_presses_constant': bar_presses_constant,
            'raw_bar_press_mask': bar_press_mask,
            'bar_presses_constant_index': bar_presses_constant_index,
            'constant_value_masks': constant_value_masks
            }
        dt += [rat]
    
    df = pd.DataFrame(dt)
    return df



def list_all_files(PATH):
    """
    List all files path in the folder and subfolders
    """
    files_path = []
    for root, subFolder, all_files in os.walk(PATH):
        for item in all_files:
            subject_tail = item.split("Subject ", 1)[-1]
            if "Subject" in item and "." not in subject_tail:
                fileNamePath = str(os.path.join(root, item))
                files_path += [fileNamePath]
    return files_path



def read_category_data(folder):
    """
    Read data from the folder and return a data frame
    folder name format: {data_category}/*/{data_file}
    !2024-07-15_15h30m.Subject 14M
    data name format: !{date}_{time}.Subject {rat_id}{gender}
    """    
    
    files = list_all_files(folder)
    category = os.path.basename(folder)
    
    raw_data_list = []
    id_list = []
    file_name_list = []
    low_force_mask_list = []
    for i in tqdm(range(len(files))):
        file_path = files[i]
        ## file base name with no folder
        fileName = os.path.basename(file_path)
        subject_id, raw_data, low_force_mask = extract_raw_data(file_path)
        raw_data_list += [raw_data]
        id_list += [subject_id]
        file_name_list += [fileName]
        low_force_mask_list += [low_force_mask]
    
    df_meta = pd.DataFrame({
        'id': id_list,
        'file_name': file_name_list,
        'low_force_mask': low_force_mask_list
    })
    
    df_data = process_raw_datas(raw_data_list)
    
    df = pd.concat([df_meta, df_data], axis=1)
    df['category'] = category
    return df
    
    

# data_root = 'data/01 Sucrose FR1 vs EXT 8_2024'
def read_cohort_type1(data_root):
    """
    Read data from the data root folder and return a data frame
    """
    ## folders only
    folders = [str(f) for f in Path(data_root).iterdir() if f.is_dir()]
    
    dt = []
    for i in range(len(folders)):
        print(f"Processing {folders[i]}")
        folder = folders[i]
        df = read_category_data(folder)
        dt += [df]
    
    df = pd.concat(dt)
    df.reset_index(drop=True, inplace=True)
    return df


def read_cohort_type2(path, time_cutoff = 5*60*100):
    """
    Read data from the cohort 2 folder and return a data frame
    folder name format: {data_category}/*/{data_file}
    !2024-07-15_15h30m.Subject 14M
    data name format: !{date}_{time}.Subject {rat_id}{gender}
    """    
    
    files = list_all_files(path)
    
    raw_data_list = []
    id_list = []
    file_name_list = []
    category_list = []
    low_force_mask_list = []
    for i in tqdm(range(len(files))):
        file_path = files[i]
        ## file base name with no folder
        fileName = os.path.basename(file_path)
        subject_id, raw_data, low_force_mask = extract_raw_data(file_path)
        raw_data_fr1 = raw_data[:time_cutoff]
        raw_data_ext = raw_data[time_cutoff:]
        low_force_mask_fr1 = low_force_mask[:time_cutoff]
        low_force_mask_ext = low_force_mask[time_cutoff:]
        
        
        raw_data_list += [raw_data_fr1, raw_data_ext]
        low_force_mask_list += [low_force_mask_fr1, low_force_mask_ext]
        
        id_list += [subject_id, subject_id]
        file_name_list += [fileName, fileName]
        category_list += ['FR1', 'EXT']
    
    df_meta = pd.DataFrame({
        'id': id_list,
        'file_name': file_name_list,
        'category': category_list,
        'low_force_mask': low_force_mask_list
    })
    
    df_data = process_raw_datas(raw_data_list)
    
    df = pd.concat([df_meta, df_data], axis=1)
    return df