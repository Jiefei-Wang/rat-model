import pandas as pd
import os


def extractData(file_path):
    data = []
    start_processing = False
    is_frustrated = 0
    ratID = 1

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "Subject" in line: 
            if "L" in line: 
                ratID = 2
        if "EXT" in line:
            is_frustrated = 1
        
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

    data.append(is_frustrated)
    data.append(ratID)
    return data

def lessThan5(data): 
    return [x if x >= 5 else 0 for x in data]

def process_data(data):
    bar_presses = []
    current_press = []
    hasLargeVal = False
    for value in data:
        if value >= 5.0:
            current_press.append(value)
            if value >= 20: #change to 20 
                hasLargeVal = True 
        elif current_press:
            if hasLargeVal: 
                bar_presses.append(current_press)
            current_press = []
            hasLargeVal = False 
    if current_press:
        bar_presses.append(current_press)
    return bar_presses

def process_file(file_path):
    data = extractData(file_path)
    rat_id = data[-1]
    is_frustrated = data[-2]
    data = data[:-2]
    data = lessThan5(data)
    bar_presses = process_data(data)
    return rat_id, is_frustrated, bar_presses


def read_data(folder):
    files = os.listdir(folder)
    files = [i for i in files if not i.startswith('.')]

    all_data = []

    for file in files:
        print(f"Processing {file}")
        ratid, frustration, data = process_file(f"{folder}/"+file)
        all_data.append([ratid, frustration, data])


    ## Data frame
    ## id: rat id
    ## frustration: 1 if the rat is frustrated, 0 otherwise
    ## data: A list of bar presses data. List element is a single bar press.
    df = pd.DataFrame(all_data, columns=['id', 'frustration', 'data'])
    return df

