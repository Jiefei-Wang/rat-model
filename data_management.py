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

