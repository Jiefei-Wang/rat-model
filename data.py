import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from scipy.signal import find_peaks 
from imblearn.over_sampling import SMOTE

import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import seaborn as sns
from statistics import mean
os.listdir('rawData')
chunkSize = 12

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



def filter_presses(bar_presses):
    return [press for press in bar_presses if len(press) >= 12]

def calculate_max_and_mean(filtered_presses):
    max_press = [max(press) for press in filtered_presses]
    mean_press = [np.mean(press) for press in filtered_presses]
    return max_press, mean_press

def calculate_average_force(filtered_presses):
    max_length = max(len(press) for press in filtered_presses)
    average_force = np.zeros(max_length)
    count = np.zeros(max_length)
        
    for press in filtered_presses:
        for i, force in enumerate(press):
            average_force[i] += force
            count[i] += 1

    average_force = np.divide(average_force, count, out=np.zeros_like(average_force), where=count!=0)
    return average_force

def calculate_peak_durations(bar_press_forces, threshold):
    threshold = 10
    peaks, _ = find_peaks(bar_press_forces)
    peak_durations = []
    
    for peak in peaks:
        peak_value = bar_press_forces[peak]
        lower_bound = peak_value - threshold
        upper_bound = peak_value + threshold
        
        # Find the start of the peak
        start = peak
        while start > 0 and bar_press_forces[start] >= lower_bound:
            start -= 1
        
        # Find the end of the peak
        end = peak
        while end < len(bar_press_forces) - 1 and bar_press_forces[end] >= lower_bound:
            end += 1
        
        # Calculate duration
        duration = end - start
        peak_durations.append(duration)
    
    return peak_durations


files = os.listdir("data")
files = [i for i in files if not i.startswith('.')]

all_data = []

for file in files:
    ratid, frustration, data = process_file("data/"+file)
    all_data.append([ratid, frustration, data])

df = pd.DataFrame(all_data, columns=['id', 'frustration', 'data'])
df

with open("output/raw.pkl", "wb") as f: 
    pickle.dump(df,f)

