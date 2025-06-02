Outline:
- It is known that frustration intolerance is a risk factor for various mental health issues, including anxiety and depression.
- Link to animal study
- Lack: There is no direct measurement of frustration level in animal
- Link to our research: We applied ML models to predict frustration level in rats based on the Barpress data.

Data Management:
- 19 rats
- Number of sessions: 2 sucrose reinforcement and 2 extinction phases per rat
- Raw force reading: 1,00 Hz sampling rate
- Data cleaning: removed low-force values (below 5), a bar press event is defined as the press force starting below 5, having a value between 5 and 20, and then reaching a force above 20
- Bar press with a constant force are removed as it is not a valid bar press event. The first and last 3 presses are removed as the rat is still learning the task.
- The force is capped at 1000, and the data is standardized to ensure consistency.

Feature Extraction:
- For each bar press event, we extract the following features:
  - Total press duration
  - Maximum force applied
  - Number of peaks
  - Peak and valley sharpness
  - Force variation rate
  - Statistical properties (skewness, kurtosis)
  - Average force during the first and last five presses




flagged problematic bar press events (force < 20), and segmented data into smaller chunks.

- Number of bar press: 11,216 bar presses in total

The raw bar press data collected from rats during the sucrose reinforcement and extinction phases were processed using a series of data management techniques to ensure data quality. First, the data were extracted from individual rat files and cleaned by removing low-force values (below 5) and collapsing consecutive zeros to reduce noise. Bar press events that never reached a force of 20 were flagged as problematic and excluded from further analysis. Valid bar press sequences were identified by detecting force peaks, and the data were segmented into smaller, manageable chunks. To prevent extreme values from skewing the results, bar press forces were further capped, and the data were standardized where necessary to ensure consistency. 


After data preprocessing, key features were extracted to characterize bar press behavior. These included total press duration, maximum force applied, number of peaks, peak and valley sharpness, force variation rate, and statistical properties such as skewness and kurtosis. Additionally, the average force during the first and last five presses was calculated to assess changes in behavior over time. A feature matrix was then created & utilized to train and test three machine learning models: Logistic Regression for baseline comparison, Random Forest for ensemble-based decision-making, and Gradient Boosting for iterative improvement. These models were assessed based on the Area Under the Receiver Operating Characteristic Curve (AUROC) scores, 10-fold stratified cross-validation was conducted to ensure the validity of the stability and generalizability of the models.