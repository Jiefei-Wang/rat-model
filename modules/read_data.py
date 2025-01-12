import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm


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

    return data

def lessThan5(data): 
    return [x if x >= 5 else 0 for x in data]

def to_bar_press(data):
    bar_presses = []
    current_press = []
    hasLargeVal = False
    for value in data:
        if value != 0:
            current_press.append(value)
            if value >= 20: #change to 20 
                hasLargeVal = True 
        elif current_press:
            if hasLargeVal: 
                bar_presses.append(current_press)
            current_press = []
            hasLargeVal = False 
    if current_press and hasLargeVal:
        bar_presses.append(current_press)
    return bar_presses

def process_file(file_path):
    data = extractData(file_path)
    data = lessThan5(data)
    bar_presses = to_bar_press(data)
    return bar_presses

def list_all_files(PATH):
    """
    List all files path in the folder and subfolders
    """
    files_path = []
    for root, subFolder, all_files in os.walk(PATH):
        for item in all_files:
            if item.startswith("!") :
                fileNamePath = str(os.path.join(root,item))
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
        file = files[i]
        ## file base name with no folder
        fileName = os.path.basename(file)
        data = process_file(file)
        rat_identifier = file.split('Subject ')[1]
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
            'data': data}
        dt += [rat]
    
    
    df = pd.DataFrame(dt)
    return df

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
