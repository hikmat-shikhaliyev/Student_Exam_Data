# Student_Exam_Data
This repository contains code for analyzing student exam data and building machine learning models to predict student pass/fail outcomes. The code is written in Python and utilizes various libraries for data manipulation, visualization, and modeling.

#### Libraries
The code begins by importing the necessary libraries, including:

pandas: for data manipulation and analysis
numpy: for numerical operations
matplotlib: for data visualization
seaborn: for advanced statistical visualization 
#### Data Loading and Preprocessing
The code then loads the student exam data from a CSV file using the pandas library. It displays the loaded data, provides a summary of the data using the describe() function, checks for missing values using the isnull().sum() function, and displays the data types of each column using the dtypes attribute.

#### Checking Multicollinearity between Independent Variables with VIF
Next, the code checks for multicollinearity between the independent variables "Study Hours" and "Previous Exam Score" using the variance inflation factor (VIF). The VIF is calculated using the variance_inflation_factor() function from the statsmodels library. The results are displayed in a table, showing the VIF values for each independent variable.

#### Outlier Checking
The code then performs outlier checking for the "Study Hours" and "Previous Exam Score" variables using box plots. It displays a box plot for each variable, showing the distribution of values and any potential outliers.

#### WOE Transformation for Logistic Regression
The code performs a Weight of Evidence (WOE) transformation for logistic regression. It creates categorical variables based on the "Study Hours" and "Previous Exam Score" variables using the cut() function from pandas. It then calculates the WOE values for each category and merges them with the original data. The resulting dataset includes the WOE-transformed variables [[14]].

#### Data Splitting and Modeling
The code splits the data into training and testing sets using the train_test_split() function from the sklearn library. It then builds several machine learning models, including logistic regression, decision tree, random forest, LightGBM, XGBoost, and CatBoost. Each model is trained on the training data and evaluated on the testing data using various performance metrics, including Gini score, accuracy score, and confusion matrix. The evaluate() function is used to calculate and display metrics.

#### Results and Visualization
The code calculates the Gini score and plots the Receiver Operating Characteristic (ROC) curve for each model. The Gini score is a measure of the model's predictive power, with higher values indicating better performance. The ROC curve visualizes the trade-off between the true positive rate and the false positive rate for different classification thresholds.

#### Variable Importance
Finally, the code calculates the Gini score for each individual variable in the dataset. It iterates over each variable, trains a CatBoost model using only that variable, and calculates the Gini score for both the training and testing data. The results are displayed in a table, showing the variables ranked by their test Gini score.
