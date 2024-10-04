# Exploratory Data Analysis for Data Science and Machine Learning 
### IBM Guided Project

## Importing Required Libraries


```python
import random
import missingno as msno
import numpy as np
import pandas as pd

import seaborn as sns
sns.set_context('notebook') # Configures the aesthetics of the plots for jupyter notebook
sns.set_style('white') # Sets background style of plots to white

import matplotlib.pyplot as plt
%matplotlib inline 
# ensures that inline plotting works correctly (newer versions of jypyter notbook does not need this)

from scipy.stats import shapiro

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from fasteda import fast_eda
```

## Regression
The aim is to predict a numeric score indicating diabetes progression one year after bloor pressure, BMI and bloor sugar level are recorder using Regression. 

## Load the diabetes data set (from sklearn)


```python
# About the data
print(load_diabetes()['DESCR'])
```


```python
# Load the data from sklearn as two pandas.DataFrame - features (X) and target variable (y)
diabetes_X, diabetes_y = load_diabetes(return_X_y = True, as_frame = True, scaled = False)

#Renaming columns
diabetes_X.columns= ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']
```


```python
diabetes_X.head()
```


```python
diabetes_y.head()
```


```python
# Combine both diabetes_X (features) and diabetes_y (target) into one pandas.DataFrame
diabetes = pd.concat([diabetes_X, pd.Series(diabetes_y)], axis=1)

#Renaming the column with target value
diabetes.rename(columns={0: 'target'}, inplace=True)
```


```python
# Looking into the data
diabetes.sample(5)
```

## Add some missing values
The original dataset does not contain any missing value hence for the sake of EDA missing values are introduced to 3 columns and 5% of the rows at random


```python
# Verifying that the data set has no missing values
diabetes.isna().max(axis=0).max()
```


```python
# Intializing seed value to 2000 to make sure that the random value is same each time the code is executed
random.seed(2024)

# Selecting 3 columns at random
missing_cols = random.sample(range(len(diabetes.columns)), 3)

# Selecting 5% of row index at random
missing_rows = random.sample(diabetes.index.tolist(), int(np.round(len(diabetes.index.tolist())/20)))

# Setting missing values to the randomly selected rows and columns
diabetes.iloc[missing_rows, missing_cols] = np.nan
```


```python
# Having a look at the columns which has been selected in random
print(diabetes.columns[missing_cols])
```


```python
# Now verifying that the data set has missing values
diabetes.isna().max(axis=0).max()
```

## Initial Data Preprocessing
Note: In a typical workflow data preprocessing comes after conducting EDA

## One-Hot Encoding
In diabetes dataset `sex` is encoded as 1 and 2 for female and male, this is not ideal for predictive models as it may consided that the column has some ordering to it. Hence we use One-Hot encoding to create two different columns for each category of `sex` with binary values in it.


```python
# Initializing OneHotEncoder (ignore unknown categories in dataset, no categories are dropped)
enc1 = OneHotEncoder(handle_unknown='ignore', drop=None)

# One-hot encode 'sex'. 
# Double square brackets are used to ensure that the extracted sex data is in DataFrame format which is required by One-hot encoder
# The output from OneHotEncoder is sparse matrix (stores only non-zero elements to save memory) which is converted to numpy array
encoded_sex = enc1.fit_transform(diabetes[['sex']]).toarray()

# Convert numpy array to pandas DataFrame with column names corresponding to its sex category 
encoded_sex = pd.DataFrame(encoded_sex, columns=['sex' + str(int(x)) for x in enc1.categories_[0]])

# Horrizontally concatenate dataframes'diabetes' and 'encoded_sex'
diabetes = pd.concat([diabetes, encoded_sex], axis=1)

# Looking into the modified diabetes DataFrame
diabetes.sample(5)
```

From above sex is indicated through `sex`, `sex1` and `sex2`, two of which is redundant hence `sex` and `sex2` can be dropped 


```python
# Drop 'sex' and 'sex2' from diabetes DataFrame
diabetes = diabetes.drop(['sex', 'sex2'], axis=1)

# Rename 'sex1' to 'sex'
diabetes = diabetes.rename(columns={'sex1': 'sex'})

# Reorder renamed 'sex' to the previous 'sex' position
diabetes = diabetes.loc[:, ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu', 'target']]

# Looking into a sample of the modified diabetes DataFrame
diabetes.sample(5)
```

## Make a Train-Test Split
Below code will randomly assign 33% of the rows to test set and the remaining 67% to training set.
Training set is used to train the predictive models and the test set is the unseen data on which predictions are made.


```python
# Make a Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.iloc[:,:-1], # Features data (all columns except the last)
    diabetes.iloc[:,-1], # Target data (last column)
    test_size=0.33, # 33% for testing
    random_state=2024 # for reproducibility
)

# `X_train` are the feature columns in the training set.
# `X_test` are the feature columns in the test set.
# `y_train` is the target column for the training set.
# `y_test` is the target column for the test set.
```

## Perform EDA

### A look at the beginning and end of the data set


```python
diabetes.head()
```


```python
diabetes.tail()
```

### Describe the DataFrame


```python
# Having a look at the general statistical summaries for the diabetes DataFrame
diabetes.describe()
```

### Missing Values


```python
# We know that the dataframe has missing values which can be verified below
diabetes.isna().max(axis=1).max()
```


```python
# To see the summary of missing values in each column
diabetes.isna().sum()
```


```python
# Visualizing the missing values in diabetes dataframe
msno.matrix(diabetes)
```

It can be easily observed how the missing values occur over the three columns bmi, s1 & s4. There are typically three approaces in dealing with the missing values,
- Dropping the observation with missing values
- Filling the observations with missing values with the mean
- Filling the observations with missing values with the median

## Dropping the observations with missing values


```python
# Linear refression with dropping NANs

# Getting the Non-NANs indices (observations/rows) of X_train and X_test
nonnan_train_indices = X_train.index[~X_train.isna().max(axis=1)]
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Fit an instance of Linear Regression with train dataset
reg = LinearRegression().fit(X_train.loc[nonnan_train_indices], y_train.loc[nonnan_train_indices])

# Generate predictions for the test dataset
pred = reg.predict(X_test.loc[nonnan_test_indices])

# Finding the root mean squared error between prediction vs test target (y_test)
root_mean_squared_error(y_test.loc[nonnan_test_indices], pred)
```

## Filling the observations with missing values with the mean


```python
# Linear regressing with mean fill

# Getting the Non-NAN indices (observations/rows) of X_test as only missing values in train dataset will be filled with mean 
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Initializing simple imputer with 'mean' strategy. 
# Note: Simple imputer supports mean, median, most_frequest and constant strategies
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# Fit the simple imputer using the training data
imp_mean.fit(X_train)

# Transforming X_train to mean filled dataset and converting it to a pandas DataFrame
X_train_mean_fill = pd.DataFrame(imp_mean.transform(X_train))

# Assigning column names to the above dataframe
X_train_mean_fill.columns= ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

# Fit an instance of Linear Regression with mean filled train dataset
reg = LinearRegression().fit(X_train_mean_fill, y_train)

# Generate predictions for the test dataset
pred = reg.predict(X_test.loc[nonnan_test_indices])

# Finding the root mean squared error between prediction vs test target (y_test)
root_mean_squared_error(y_test.loc[nonnan_test_indices], pred)
```

## Filling the observations with missing values with the median


```python
# Linear regressing with median fill

# Getting the Non-NAN indices (observations/rows) of X_test as only missing values in train dataset will be filled with median
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Initializing simple imputer with 'median' strategy. 
# Note: Simple imputer supports mean, median, most_frequest and constant strategies
imp_median = SimpleImputer(missing_values = np.nan, strategy = 'median')

# Fit the simple imputer using the training data
imp_median.fit(X_train)

# Transforming X_train to median filled dataset and converting it to a pandas DataFrame
X_train_median_fill = pd.DataFrame(imp_median.transform(X_train))

# Assigning column names to the above dataframe
X_train_median_fill.columns= ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

# Fit an instance of Linear Regression with median filled train dataset
reg = LinearRegression().fit(X_train_median_fill, y_train)

# Generate predictions for the test dataset
pred = reg.predict(X_test.loc[nonnan_test_indices])

# Finding the root mean squared error between prediction vs test target (y_test)
root_mean_squared_error(y_test.loc[nonnan_test_indices], pred)
```

### The root mean squared error is minimum for linear regressing with missing values filled with median. Looking into ways to impove this.

## Histograms and Boxplots


```python
# Define a function that takes columns_toplt as an argument
def plot_hist_and_box(diabetes, columns_toplt):
    for idx, col in enumerate(columns_toplt): 
        # Creates two subplots (2 plots in a row)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6)) 
        
        # Creating a histogram in first subplot (ax1) with KDE overlay 
        sns.histplot(diabetes, x=diabetes[col], kde=True,
                     color=sns.color_palette('hls', len(columns_toplt))[idx], ax=ax1)
        
        # Creating a boxplot in second subplot (ax2) with the same color as histogram
        sns.boxplot(diabetes, x=diabetes[col], width=0.4, linewidth=3, fliersize=2.5, 
                    color=sns.color_palette('hls', len(columns_toplt))[idx], ax=ax2)
        
        # Adding title to the figure
        fig.suptitle(f"Histogram and Boxplot of {col}", size=20, y=1.02)
        plt.show()
```


```python
# Assigning column names of all columns in diabetes dataframe except 'sex'
columns_toplt = [i for i in diabetes.columns if i != 'sex']

# Call the function
plot_hist_and_box(diabetes, columns_toplt)
```


```python
# looking closely into hdl column
# Assigning column name 'hdl'
columns_toplt = ['hdl']

# Call the function
plot_hist_and_box(diabetes, columns_toplt)
```

## Normality Test on 'hdl'


```python
# Normality test on 'hdl'
stat, p = shapiro(X_train['hdl'])
print('Statistics = %.3f, p = %.3f' % (stat, p))

# Interpret
alpha = 0.05
if p > alpha:
    print("Sample is normally distributes (Fail to reject null hypothesis)")
else:
    print("Sample is not normally distributes(reject null hypothesis)")
```

## Normality Test on log of 'hdl'


```python
# Normality test on 'hdl'
stat, p = shapiro(np.log(X_train['hdl']))
print('Statistics = %.3f, p = %.3f' % (stat, p))

# Interpret
alpha = 0.05
if p > alpha:
    print("Sample is normally distributes (Fail to reject null hypothesis)")
else:
    print("Sample is not normally distributes(reject null hypothesis)")
```

## Linear Regression with missing observations filled with median and log of 'hdl'


```python
# Replacing 'hdl' column in X_train and X_test with log of 'hdl'
X_train['hdl'] = np.log(X_train['hdl'])
X_test['hdl'] = np.log(X_test['hdl'])

# Getting Non-NAN index values for X_test
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Initializing simple imputer with 'median' strategy
imp_median = SimpleImputer(missing_values = np.nan, strategy = 'median')

# Fit simple imputer with training data (with log 'hdl')
imp_median.fit(X_train)

# Transforming X_train to median filled dataset and converting it to a pandas DataFrame
X_train_median_log_fill = pd.DataFrame(imp_median.transform(X_train))

# Assigning column names to the above dataframe
X_train_median_log_fill.columns= ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

# Fit an instance of Linear Regression
reg = LinearRegression().fit(X_train_median_log_fill, y_train)

# Generate prediction for X_test dataset
pred = reg.predict(X_test.loc[nonnan_test_indices])

# Calculate Root Mean Squared error 
root_mean_squared_error(y_test.loc[nonnan_test_indices], pred)
```

### Root mean squared error has improved. Looking into column 'ldl' for more improvements


```python
# looking closely into ldl column
# Assigning column name 'ldl'
columns_toplt = ['ldl']

# Call the function
plot_hist_and_box(diabetes, columns_toplt)
```

## Linear Regression with missing observations filled with median, log of 'hdl' and removal of oultiers in 'ldl'


```python
# Removing outlier index from 'idl'
X_train_nonoutlier_idx = X_train.index[X_train.ldl < X_train.ldl.quantile(0.999)]
X_train = X_train.loc[X_train_nonoutlier_idx]
y_train = y_train.loc[X_train_nonoutlier_idx]

# Getting Non-NAN index values for X_test
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Initializing simple imputer with 'median' strategy
imp_median = SimpleImputer(missing_values = np.nan, strategy = 'median')

# Fit simple imputer with training data (with log 'hdl')
imp_median.fit(X_train)

# Transforming X_train to median filled dataset and converting it to a pandas DataFrame
X_train_median_log_fill = pd.DataFrame(imp_median.transform(X_train))

# Assigning column names to the above dataframe
X_train_median_log_fill.columns= ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

# Fit an instance of Linear Regression
reg = LinearRegression().fit(X_train_median_log_fill, y_train)

# Generate prediction for X_test dataset
pred = reg.predict(X_test.loc[nonnan_test_indices])

# Calculate Root Mean Squared error 
root_mean_squared_error(y_test.loc[nonnan_test_indices], pred)
```

## Correlation Matrix


```python
plt.figure(figsize = (12, 8))
sns.heatmap(diabetes.corr(), annot = True, cmap = 'Spectral', linewidth = 2, linecolor = '#000000', fmt = '.3f')
plt.show()
```

It can be observed that the correlation of 'tc' and 'idl' to 'target' is very low. Hence the we might be able to improve the regression model by dropping the column 'tc'

## Linear Regression with meadian filled, log of 'hdl', removed outliers in 'idl' and dropping 'tc'


```python
# Removing outlier index from 'idl'
X_train_nonoutlier_idx = X_train.index[X_train.ldl < X_train.ldl.quantile(0.999)]
X_train = X_train.loc[X_train_nonoutlier_idx]
y_train = y_train.loc[X_train_nonoutlier_idx]

# Getting Non-NAN index values for X_test
nonnan_test_indices = X_test.index[~X_test.isna().max(axis=1)]

# Getting column names except 'tc'
col_no_tc = [i for i in X_train.columns if i != 'tc']

# Initializing simple imputer with 'median' strategy
imp_median = SimpleImputer(missing_values = np.nan, strategy = 'median')

# Fit simple imputer with training data (with log 'hdl')
imp_median.fit(X_train.loc[:, col_no_tc])

# Transforming X_train to median filled dataset and converting it to a pandas DataFrame
X_train_median_log_fill = pd.DataFrame(imp_median.transform(X_train.loc[:, col_no_tc]))

# Assigning column names to the above dataframe
X_train_median_log_fill.columns= ['age', 'sex', 'bmi', 'bp', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

# Fit an instance of Linear Regression
reg = LinearRegression().fit(X_train_median_log_fill, y_train)

# Generate prediction for X_test dataset
pred = reg.predict(X_test.loc[nonnan_test_indices, col_no_tc])

# Calculate Root Mean Squared error 
root_mean_squared_error(y_test.loc[nonnan_test_indices], pred)
```

Removal of 'tc' column has lead to worse performance

## Pair Plots


```python
sns.pairplot(diabetes)
plt.show()
```

## A Simple function to perform EDA - fasteda

The fast_eda from fasteda package does all the above EDA analysis in single step


```python
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

# Now run fast_eda(diabetes) function
fast_eda(diabetes)

```

# Classification

## Import Iris Data Set


```python
# Load the data set from sklearn
iris_sklearn = load_iris()

# Extract the data and target labels as a numpy array
iris_npy = np.concatenate([iris_sklearn['data'], np.atleast_2d(iris_sklearn['target']).T], axis=1)

# Define column names
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# Convert the numpy array to a pandas dataframe with column names
iris = pd.DataFrame(iris_npy, columns=col_names)

# Print a description of the dataset
print(iris_sklearn['DESCR'])
```


```python
iris['target'].sample(5)
```


```python
class_names = dict(zip(list(map(float, range(len(iris_sklearn['target_names'])))), iris_sklearn['target_names']))
print(class_names)
```

## Performing EDA for classification using fasteda


```python
fast_eda(iris, target = 'target')
```


```python
plt.axis('equal')
sns.scatterplot(iris, x='petal_width', y='sepal_width', hue='target', palette=sns.color_palette("hls", iris['target'].nunique()))
plt.show()
```


```python
# Define a function to format value counts into percentages
def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

# Get value counts
vc = iris['target'].value_counts()

# Draw a pie chart using value counts and the `autopct_format` format
_ = plt.pie(vc, labels = vc.rename(class_names).index, autopct=autopct_format(vc))
```
