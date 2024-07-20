import pandas as pd

def chunk_data(df, chunk_size):
   if chunk_size <= 0:
      raise ValueError("Chunk size must be a positive integer")

   new_rows = []
    
   
   for _, row in df.iterrows():
      id_val = row['id']
      frustration_val = row['frustration']
      data_list = row['data']
        
        
      num_chunks = len(data_list) // chunk_size
      for i in range(num_chunks):
         chunk = data_list[i*chunk_size:(i+1)*chunk_size]
         new_rows.append({'id': id_val, 'frustration': frustration_val, 'data': chunk})
    
   new_df = pd.DataFrame(new_rows)
   return new_df


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
    