This project implement ML algorithms for classifying rat frustration status

## Checklist

- [] Data reading
    - [] Read all data from the data folders
    - [] Make a dataframe with three columns: `rat_id`, `frustration`, `data`
        - `rat_id`: the id of the rat
        - `frustration`: the frustration status of the rat(0: not frustrated, 1: frustrated)
        - `data`: list, the raw bar press data of the rat
- [] Data preprocessing
    - [] Segment the data into individual bar press sequences
    - [] Chunk the data. Each chunk should contain k bar presses
    - [] create a new dataframe with each row being a chunk of data
    - [] Normalize the data
- [] Machine learning Models
    - [] Feature extraction
    - [] Logistic Regression
    - [] Random Forest
    - [] Gradient Boosting Trees
    - [] Recurrent Neural Network
    - [] Long Short Term Memory Network
- [] Model evaluation
  - [] True Positive Rate
  - [] True Negative Rate
  - [] AUC