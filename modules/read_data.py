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
            if not output or output[-1] <= threshold:
                output.append(0)
        else:
            output.append(value)
    
    return output


def extractData(file_path):
    data = []
    start_processing = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
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

    ## Remove excessive zeros
    data = collapse_zeros_alike(data)
    return data

def filter_low_force(data): 
    new_data = [x if x >= low_force_threshold else 0 for x in data]
    mask = [x < low_force_threshold for x in data]
    return new_data, mask

def to_bar_press(data):
    bar_presses = []
    mask = [False] * len(data)
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
                bar_presses.append(data[press_start:i])
                mask[press_start:i] = [True] * (i - press_start)
            press_start = None
            has_large_val = False
            found_medium_force = False
    ## The last bar press
    if press_start is not None and has_large_val and found_medium_force:
        bar_presses.append(data[press_start:])
        mask[press_start:] = [True] * (len(data) - press_start)
    return bar_presses, mask


def remove_constant_values(data):
    """
    data: 1D array-like of numeric values (Python list)
    
    returns: a new list where constant segments are replaced with 0
    """
    new_data = data[:]  # make a copy to avoid modifying the original
    mask = [False] * len(data)
    n = len(new_data)
    
    i = 0
    while i < n:
        # We'll consider a potential "constant" segment starting at index i
        start_val = new_data[i]
        
        ## If the value is 0, skip
        if start_val == 0:
            i += 1
            continue
        
        current_min = start_val
        current_max = start_val
        
        
        j = i
        # Try to extend the segment until we exceed the threshold
        while j < n:
            val = new_data[j]
            
            ## If the value is 0, stop
            if val == 0:
                break

            potential_min = min(current_min, val)
            potential_max = max(current_max, val)
            
            if (potential_max - potential_min) <= 2 * constant_force_threshold:
                # Update current_min and current_max because we can include val
                current_min, current_max = potential_min, potential_max
                j += 1
            else:
                # We exceeded the allowed variation, so break
                break
        
        # Now, j is just past the end of the potential constant segment
        run_length = j - i
        
        # If run is long enough, set all those elements to 0
        if run_length >= constant_run_threshold:
            for k in range(i, j):
                new_data[k] = 0
                mask[k] = True
        
        # Jump i past this entire segment 
        # Ideally we should increment i by 1
        # we do this for simplicity
        i = j
    
    return new_data, mask


def process_file(file_path):
    raw_data = extractData(file_path)
    data, low_force_mask = filter_low_force(raw_data)
    data, constant_force_mask = remove_constant_values(data)
    bar_presses, bar_press_mask = to_bar_press(data)
    return bar_presses, raw_data, low_force_mask, constant_force_mask, bar_press_mask

def list_all_files(PATH):
    """
    List all files path in the folder and subfolders
    """
    files_path = []
    for root, subFolder, all_files in os.walk(PATH):
        for item in all_files:
            if item.startswith("!"):
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
    
    dt = []
    ## progress bar
    for i in tqdm(range(len(files))):
        file_path = files[i]
        ## file base name with no folder
        fileName = os.path.basename(file_path)
        data, raw_data, low_force_mask, constant_force_mask, bar_press_mask = process_file(file_path)
        
        rat_identifier = file_path.split('Subject ')[1]
        ## get integer part of the rat id
        id = int(''.join(filter(str.isdigit, rat_identifier)))
        ## gender
        sex = ''.join(filter(str.isalpha, rat_identifier))
        ## F,M to 0,1
        sex = 1 if sex == 'M' else 0
        ## combine the data
        rat = {
            'id': id,
            'category': category,
            'file': fileName,
            'sex': sex,
            'data': data,
            'raw_data': raw_data,
            'low_force_mask': low_force_mask,
            'constant_force_mask': constant_force_mask,
            'bar_press_mask': bar_press_mask
            }
        dt += [rat]
    
    df = pd.DataFrame(dt)
    return df

# data_root = 'data/01 Sucrose FR1 vs EXT 8_2024'
def read_data(data_root):
    """
    Read data from the data root folder and return a data frame
    """
    ## list folders within the data folder
    folders = os.listdir(data_root)
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
