# Prediction-of-Expected-Salary-using-Machine-Learning
Here is my first Project Repository where I have made a Machine Learning Project using Python.


The Problem Statemnent of the Project was: 
To ensure there is no discrimination between employees, it is imperative for the Human 
Resources department of Company X to maintain a salary range for each employee with a similar profile. 
Apart from the existing salary, a considerable number of factors, such as an employee’s experience and 
other abilities, are evaluated during interviews. Given the data related to individuals who applied to 
Company X, models can be built that automatically determine the salary to be offered if a prospective 
candidate is selected. This model seeks to minimize human judgment in salary decisions.

Goal and Objective of the Project: 
The objective of this exercise is to build a model, using historical data, that will determine 
the salary to be offered to an employee, minimizing manual judgment in the selection process. The 
approach aims to be robust and eliminate any discrimination in salary among employees with similar 
profiles.


Key Concepts: 
As I had to Predict the Expected Salary (which is a Continuous Value) I used two Regression Algorithms:
  1> Linear Regression
  2> Random Forest Regressor



Quick Glimpse of the concepts: 

Linear Regression: The Goal is to Model a relationship between Features X and Target Variable y.
  Prediction:  ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θnxn  where Theta represents the Coefficients. We need to solve for these Coefficents using OLS (Ordinary Least Squares) to minimize the suqared errors.

  Key Assumptions:
  -> Linearity
  -> Homoscedasticity
  -> Independent Errors
  -> Normal Residual
  -> Low Multicollinearity

  Pre-processing Notes: 
  -> Scaling and Normallization of Dataset must be done before feedin it to the algorithm so as to avoid Numerical Instability
  -> One-Hot encoding to be done for converting the Categorical Columns into numerical columns

  Metrics used: 
  -> RMSE: Root Mean Square Error (Less is the Value better is the Model's Performance)
  -> R-square score: Proportion of Variance Explained

  Advantages of Linear Regression:
  => Simple
  => Fast
  => Interpretability of Coefficients

  Disadvantages of Linear Regression:
  => Cannot handle non-linear patterns
  => Is not robust to outliers


Random Forest: It's an ensemble model which means, it evaluates the output of n-number of Decision Trees and then takes the Average result of all the Outputs from each tree. It gets trained on many bootstraped samples to generate the average of the outputs.

  HyperParameters: Some of the hyperparameters that we need to pass while training on this model are-> n_estimators (number of     
  trees), max_depth, max_leaf ...and there are  several other parameters too.

  Pre-processing Notes: 
  -> Random Forest does not care about whether the Data is Scaled and Normalized. It can work easily in either cases.
  -> One-Hot encoding is to be done before feeding the dataset to the Algorithm

  Advantages of Random Forest:
  => It can handle Non-linear patterns
  => It is robust to outliers

  Disadvantages of Random Forest:
  => It is not Fast as the output of each of the n-number of decision trees are calculated first and then it calcultes the average from it.
  => Its is not simple and interpretable


Tools/Libraries I have used in my Project: Pandas | Numpy | Matplotlib | Seaborn | Scikit-Learn | Pathlib | JobLib 


Step-wise process of what I have done in my Project:

-> Step 1: Loading and Inspecting the Dataset
    -Initial Inspection on the Dataset like (checking the total number of rows and columns, total number of null values)
    -Printing the first 5 records
    -Getting the Description using .describe() method. (which returns a table having count, uniq, top, min, 25%, 50%, 75%, max, std,       mean)
    
-> Step 2: Handling Missing Values and Duplicates: 
    -Dropping Duplicate Entries
    -Imputing Numerical Column Null values with: 
     => Mean (when absolute skewness value is less than equal to 0.5)
     => Median (when absolute skewness value is greater than 0.5 and this is also robust to outliers)
    -Imputing Categorical Column Null values with:
     => Mode (By taking the most frequent value from the column)
    
-> Step 3: EDA (Exploratory Data Analysis): 
    -Consisting of Histograms of Categorical Columns
    -Univariate analysis using Frequency Distribution Graphs
    -Bivariate analysis using Pairplots and Histograms (as diagonal graphs instead of Kernel Density Estimation)
    -Correlation Heatmap

-> Step 4: Iterative Outlier Removal:
    -This section will also show us the Boxplots of the Columns Before and After removal of the Outliers

-> Step 5: One-Hot Encoding:
    -Helps to convert categorical columns to numrerical columns
    -In each coolumn, for n categories, (n-1) number of dummies are formed. 

-> Step 6: Train-Test split:
    -A split of 80:20 has been used (80% Training and 20% testing)

-> Step 7: Normalization and Scaling:
    -To avoid Numerical Instability

-> Step 8: Training and Evaluating by Linear Regression:
    -We train the model with Encoded Features and the Target column
    -During Making Predictions I have checked for fit quality and Genralization
    -Then calculation of RMSE and R-square score for Linear Regression model has been done

-> Step 9: Training and Evaluating by Random Forest Regressor:
    -Again We train the model with Encoded Features and the Target column
    -Also During Making Predictions I have checked for fit quality and Genralization
    -Then calculation of RMSE and R-square score for Random Forest Regressor model has been done

Then we could clearly see that Random Forest Regressor has Out-performed the Linear Regression Model with slightly higher R-square score and very less RMSE score than Linear Regression Model, both indicating that Random Forest Regressor is a better model.

-> Step 10: Visual Comparisons for both Models
    -I have generated bar graphs that clearly shows that Random Forest has outperformed Linear Regression Model
    -I have plotted other graphs also which shows the Top 15 most important Features that were taken during each split for Random         Forest
    -I have also written codes that will help me to preserve the values of the final model (Random Forest Regressor) in a csv file        and have also saved the model using Joblib Library.

-> Step 11: Final visual comparison and Residual analysis:
    -Here we can see the Scatter plot that shows all the datapoints will be near to the zero-line and no noticeable patterns found        after  training it with Random Forest Regressor, so this indicates we have not missed anything.

I have attached my Google Colaboratory file that includes Everything (Code, Graphs and Comparison after Model training).
Also I will be attaching the dataset 'expected_ctc.csv' which I have used there.
