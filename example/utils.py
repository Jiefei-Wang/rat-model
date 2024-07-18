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

    if not data:
        print("No data found in file.")
    else:
        print("Data extracted successfully.")

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
            if value >= 10: 
                hasLargeVal = True 
        elif current_press:
            if hasLargeVal: 
                bar_presses.append(current_press)
            current_press = []
            hasLargeVal = False 
    if current_press:
        bar_presses.append(current_press)

    return bar_presses


def filter_presses(bar_presses):
    return [press for press in bar_presses if len(press) >= 12]

def calculate_max_and_mean(filtered_presses):
    max_press = [max(press) for press in filtered_presses]
    mean_press = [np.mean(press) for press in filtered_presses]
    return max_press, mean_press

def calculate_average_force(filtered_presses):
    max_length = max(len(press) for press in filtered_presses)
        #ensures that we create arrays (average_force and count) that are large enough to store data for the longest bar press.
    average_force = np.zeros(max_length)
    count = np.zeros(max_length)
        #average_force will store the sum of forces at each time point.
        #count will store how many times a force has been added at each time point (to later compute the average).

    for press in filtered_presses:
        for i, force in enumerate(press):
            average_force[i] += force
            count[i] += 1

    # Avoids division by zero
    average_force = np.divide(average_force, count, out=np.zeros_like(average_force), where=count!=0)

    return average_force



def chunk_data(data, chunk_size=100):
    """
    data: List of data
    chunk_size: Size of each chunk
    """
    chunk = []
    for i in range(0, len(data), chunk_size):
        if i + chunk_size > len(data):
            break
        chunk.append(data[i:i+chunk_size])
    return chunk



def concate_data(list_of_lists):
    concatenated_list = []
    # Iterate over the list of lists
    for i, sublist in enumerate(list_of_lists):
        concatenated_list.extend(sublist)  # Add the elements of the sublist to the result list
        if i < len(list_of_lists) - 1:
            concatenated_list.append(0)  # Add 0 between the sublists
            
    return concatenated_list



def truncate_or_padding(data, length=100):
    if len(data) > length:
        return data[:length]
    else:
        return data + [0]*(length-len(data))
    