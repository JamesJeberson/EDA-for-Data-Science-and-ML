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

    .. _diabetes_dataset:
    
    Diabetes dataset
    ----------------
    
    Ten baseline variables, age, sex, body mass index, average blood
    pressure, and six blood serum measurements were obtained for each of n =
    442 diabetes patients, as well as the response of interest, a
    quantitative measure of disease progression one year after baseline.
    
    **Data Set Characteristics:**
    
    :Number of Instances: 442
    
    :Number of Attributes: First 10 columns are numeric predictive values
    
    :Target: Column 11 is a quantitative measure of disease progression one year after baseline
    
    :Attribute Information:
        - age     age in years
        - sex
        - bmi     body mass index
        - bp      average blood pressure
        - s1      tc, total serum cholesterol
        - s2      ldl, low-density lipoproteins
        - s3      hdl, high-density lipoproteins
        - s4      tch, total cholesterol / HDL
        - s5      ltg, possibly log of serum triglycerides level
        - s6      glu, blood sugar level
    
    Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times the square root of `n_samples` (i.e. the sum of squares of each column totals 1).
    
    Source URL:
    https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
    
    For more information see:
    Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
    (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
    
    


```python
# Load the data from sklearn as two pandas.DataFrame - features (X) and target variable (y)
diabetes_X, diabetes_y = load_diabetes(return_X_y = True, as_frame = True, scaled = False)

#Renaming columns
diabetes_X.columns= ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']
```


```python
diabetes_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>2.0</td>
      <td>32.1</td>
      <td>101.0</td>
      <td>157.0</td>
      <td>93.2</td>
      <td>38.0</td>
      <td>4.0</td>
      <td>4.8598</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.0</td>
      <td>1.0</td>
      <td>21.6</td>
      <td>87.0</td>
      <td>183.0</td>
      <td>103.2</td>
      <td>70.0</td>
      <td>3.0</td>
      <td>3.8918</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72.0</td>
      <td>2.0</td>
      <td>30.5</td>
      <td>93.0</td>
      <td>156.0</td>
      <td>93.6</td>
      <td>41.0</td>
      <td>4.0</td>
      <td>4.6728</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24.0</td>
      <td>1.0</td>
      <td>25.3</td>
      <td>84.0</td>
      <td>198.0</td>
      <td>131.4</td>
      <td>40.0</td>
      <td>5.0</td>
      <td>4.8903</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.0</td>
      <td>1.0</td>
      <td>23.0</td>
      <td>101.0</td>
      <td>192.0</td>
      <td>125.4</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>4.2905</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
diabetes_y.head()
```




    0    151.0
    1     75.0
    2    141.0
    3    206.0
    4    135.0
    Name: target, dtype: float64




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43</th>
      <td>54.0</td>
      <td>1.0</td>
      <td>24.2</td>
      <td>74.0</td>
      <td>204.0</td>
      <td>109.0</td>
      <td>82.0</td>
      <td>2.0</td>
      <td>4.1744</td>
      <td>109.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>44.0</td>
      <td>1.0</td>
      <td>25.4</td>
      <td>95.0</td>
      <td>162.0</td>
      <td>92.6</td>
      <td>53.0</td>
      <td>3.0</td>
      <td>4.4067</td>
      <td>83.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>394</th>
      <td>58.0</td>
      <td>1.0</td>
      <td>28.1</td>
      <td>111.0</td>
      <td>198.0</td>
      <td>80.6</td>
      <td>31.0</td>
      <td>6.0</td>
      <td>6.0684</td>
      <td>93.0</td>
      <td>273.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>34.0</td>
      <td>2.0</td>
      <td>22.6</td>
      <td>75.0</td>
      <td>166.0</td>
      <td>91.8</td>
      <td>60.0</td>
      <td>3.0</td>
      <td>4.2627</td>
      <td>108.0</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>361</th>
      <td>60.0</td>
      <td>1.0</td>
      <td>25.7</td>
      <td>103.0</td>
      <td>158.0</td>
      <td>84.6</td>
      <td>64.0</td>
      <td>2.0</td>
      <td>3.8501</td>
      <td>97.0</td>
      <td>182.0</td>
    </tr>
  </tbody>
</table>
</div>



## Add some missing values
The original dataset does not contain any missing value hence for the sake of EDA missing values are introduced to 3 columns and 5% of the rows at random


```python
# Verifying that the data set has no missing values
diabetes.isna().max(axis=0).max()
```




    False




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

    Index(['tch', 'bmi', 'tc'], dtype='object')
    


```python
# Now verifying that the data set has missing values
diabetes.isna().max(axis=0).max()
```




    True



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
      <th>sex1</th>
      <th>sex2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>411</th>
      <td>50.0</td>
      <td>1.0</td>
      <td>31.8</td>
      <td>82.0</td>
      <td>136.0</td>
      <td>69.2</td>
      <td>55.0</td>
      <td>2.0</td>
      <td>4.0775</td>
      <td>85.0</td>
      <td>136.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>50.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>88.0</td>
      <td>140.0</td>
      <td>71.8</td>
      <td>35.0</td>
      <td>4.0</td>
      <td>5.1120</td>
      <td>71.0</td>
      <td>252.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>403</th>
      <td>43.0</td>
      <td>1.0</td>
      <td>35.4</td>
      <td>93.0</td>
      <td>185.0</td>
      <td>100.2</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>5.3181</td>
      <td>101.0</td>
      <td>275.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>264</th>
      <td>58.0</td>
      <td>2.0</td>
      <td>29.0</td>
      <td>85.0</td>
      <td>156.0</td>
      <td>109.2</td>
      <td>36.0</td>
      <td>4.0</td>
      <td>3.9890</td>
      <td>86.0</td>
      <td>145.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23.0</td>
      <td>1.0</td>
      <td>22.6</td>
      <td>89.0</td>
      <td>139.0</td>
      <td>64.8</td>
      <td>61.0</td>
      <td>2.0</td>
      <td>4.1897</td>
      <td>68.0</td>
      <td>97.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>139</th>
      <td>55.0</td>
      <td>1.0</td>
      <td>32.1</td>
      <td>110.0</td>
      <td>164.0</td>
      <td>84.2</td>
      <td>42.0</td>
      <td>4.0</td>
      <td>5.2417</td>
      <td>90.0</td>
      <td>281.0</td>
    </tr>
    <tr>
      <th>307</th>
      <td>67.0</td>
      <td>0.0</td>
      <td>23.5</td>
      <td>96.0</td>
      <td>207.0</td>
      <td>138.2</td>
      <td>42.0</td>
      <td>5.0</td>
      <td>4.8978</td>
      <td>111.0</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>282</th>
      <td>68.0</td>
      <td>1.0</td>
      <td>25.9</td>
      <td>93.0</td>
      <td>253.0</td>
      <td>181.2</td>
      <td>53.0</td>
      <td>5.0</td>
      <td>4.5433</td>
      <td>98.0</td>
      <td>230.0</td>
    </tr>
    <tr>
      <th>433</th>
      <td>41.0</td>
      <td>1.0</td>
      <td>20.8</td>
      <td>86.0</td>
      <td>223.0</td>
      <td>128.2</td>
      <td>83.0</td>
      <td>3.0</td>
      <td>4.0775</td>
      <td>89.0</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>275</th>
      <td>47.0</td>
      <td>0.0</td>
      <td>25.3</td>
      <td>98.0</td>
      <td>173.0</td>
      <td>105.6</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>4.7622</td>
      <td>108.0</td>
      <td>94.0</td>
    </tr>
  </tbody>
</table>
</div>



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>0.0</td>
      <td>32.1</td>
      <td>101.0</td>
      <td>157.0</td>
      <td>93.2</td>
      <td>38.0</td>
      <td>4.0</td>
      <td>4.8598</td>
      <td>87.0</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.0</td>
      <td>1.0</td>
      <td>21.6</td>
      <td>87.0</td>
      <td>183.0</td>
      <td>103.2</td>
      <td>70.0</td>
      <td>3.0</td>
      <td>3.8918</td>
      <td>69.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72.0</td>
      <td>0.0</td>
      <td>30.5</td>
      <td>93.0</td>
      <td>156.0</td>
      <td>93.6</td>
      <td>41.0</td>
      <td>4.0</td>
      <td>4.6728</td>
      <td>85.0</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24.0</td>
      <td>1.0</td>
      <td>25.3</td>
      <td>84.0</td>
      <td>198.0</td>
      <td>131.4</td>
      <td>40.0</td>
      <td>5.0</td>
      <td>4.8903</td>
      <td>89.0</td>
      <td>206.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.0</td>
      <td>1.0</td>
      <td>23.0</td>
      <td>101.0</td>
      <td>192.0</td>
      <td>125.4</td>
      <td>52.0</td>
      <td>4.0</td>
      <td>4.2905</td>
      <td>80.0</td>
      <td>135.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
diabetes.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>437</th>
      <td>60.0</td>
      <td>0.0</td>
      <td>28.2</td>
      <td>112.00</td>
      <td>185.0</td>
      <td>113.8</td>
      <td>42.0</td>
      <td>4.00</td>
      <td>4.9836</td>
      <td>93.0</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>438</th>
      <td>47.0</td>
      <td>0.0</td>
      <td>24.9</td>
      <td>75.00</td>
      <td>225.0</td>
      <td>166.0</td>
      <td>42.0</td>
      <td>5.00</td>
      <td>4.4427</td>
      <td>102.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>439</th>
      <td>60.0</td>
      <td>0.0</td>
      <td>24.9</td>
      <td>99.67</td>
      <td>162.0</td>
      <td>106.6</td>
      <td>43.0</td>
      <td>3.77</td>
      <td>4.1271</td>
      <td>95.0</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>440</th>
      <td>36.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>95.00</td>
      <td>201.0</td>
      <td>125.2</td>
      <td>42.0</td>
      <td>4.79</td>
      <td>5.1299</td>
      <td>85.0</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>36.0</td>
      <td>1.0</td>
      <td>19.6</td>
      <td>71.00</td>
      <td>250.0</td>
      <td>133.2</td>
      <td>97.0</td>
      <td>3.00</td>
      <td>4.5951</td>
      <td>92.0</td>
      <td>57.0</td>
    </tr>
  </tbody>
</table>
</div>



### Describe the DataFrame


```python
# Having a look at the general statistical summaries for the diabetes DataFrame
diabetes.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>442.000000</td>
      <td>442.000000</td>
      <td>420.000000</td>
      <td>442.000000</td>
      <td>420.000000</td>
      <td>442.000000</td>
      <td>442.000000</td>
      <td>420.000000</td>
      <td>442.000000</td>
      <td>442.000000</td>
      <td>442.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48.518100</td>
      <td>0.531674</td>
      <td>26.358095</td>
      <td>94.647014</td>
      <td>188.830952</td>
      <td>115.439140</td>
      <td>49.788462</td>
      <td>4.071595</td>
      <td>4.641411</td>
      <td>91.260181</td>
      <td>152.133484</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.109028</td>
      <td>0.499561</td>
      <td>4.404820</td>
      <td>13.831283</td>
      <td>34.690827</td>
      <td>30.413081</td>
      <td>12.934202</td>
      <td>1.296942</td>
      <td>0.522391</td>
      <td>11.496335</td>
      <td>77.093005</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>62.000000</td>
      <td>97.000000</td>
      <td>41.600000</td>
      <td>22.000000</td>
      <td>2.000000</td>
      <td>3.258100</td>
      <td>58.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>38.250000</td>
      <td>0.000000</td>
      <td>23.175000</td>
      <td>84.000000</td>
      <td>164.000000</td>
      <td>96.050000</td>
      <td>40.250000</td>
      <td>3.000000</td>
      <td>4.276700</td>
      <td>83.250000</td>
      <td>87.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>1.000000</td>
      <td>25.700000</td>
      <td>93.000000</td>
      <td>186.000000</td>
      <td>113.000000</td>
      <td>48.000000</td>
      <td>4.000000</td>
      <td>4.620050</td>
      <td>91.000000</td>
      <td>140.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>59.000000</td>
      <td>1.000000</td>
      <td>29.325000</td>
      <td>105.000000</td>
      <td>209.000000</td>
      <td>134.500000</td>
      <td>57.750000</td>
      <td>5.000000</td>
      <td>4.997200</td>
      <td>98.000000</td>
      <td>211.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79.000000</td>
      <td>1.000000</td>
      <td>42.200000</td>
      <td>133.000000</td>
      <td>301.000000</td>
      <td>242.400000</td>
      <td>99.000000</td>
      <td>9.090000</td>
      <td>6.107000</td>
      <td>124.000000</td>
      <td>346.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Missing Values


```python
# We know that the dataframe has missing values which can be verified below
diabetes.isna().max(axis=1).max()
```




    True




```python
# To see the summary of missing values in each column
diabetes.isna().sum()
```




    age        0
    sex        0
    bmi       22
    bp         0
    tc        22
    ldl        0
    hdl        0
    tch       22
    ltg        0
    glu        0
    target     0
    dtype: int64




```python
# Visualizing the missing values in diabetes dataframe
msno.matrix(diabetes)
```




    <Axes: >




    
![png](output_32_1.png)
    


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




    55.962919546725054



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




    55.95122410079265



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




    55.9148764740674



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


    
![png](output_43_0.png)
    



    
![png](output_43_1.png)
    



    
![png](output_43_2.png)
    



    
![png](output_43_3.png)
    



    
![png](output_43_4.png)
    



    
![png](output_43_5.png)
    



    
![png](output_43_6.png)
    



    
![png](output_43_7.png)
    



    
![png](output_43_8.png)
    



    
![png](output_43_9.png)
    



```python
# looking closely into hdl column
# Assigning column name 'hdl'
columns_toplt = ['hdl']

# Call the function
plot_hist_and_box(diabetes, columns_toplt)
```


    
![png](output_44_0.png)
    


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

    Statistics = 0.962, p = 0.000
    Sample is not normally distributes(reject null hypothesis)
    

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

    Statistics = 0.996, p = 0.700
    Sample is normally distributes (Fail to reject null hypothesis)
    

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




    55.685863090763554



### Root mean squared error has improved. Looking into column 'ldl' for more improvements


```python
# looking closely into ldl column
# Assigning column name 'ldl'
columns_toplt = ['ldl']

# Call the function
plot_hist_and_box(diabetes, columns_toplt)
```


    
![png](output_52_0.png)
    


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




    55.53368308287885



## Correlation Matrix


```python
plt.figure(figsize = (12, 8))
sns.heatmap(diabetes.corr(), annot = True, cmap = 'Spectral', linewidth = 2, linecolor = '#000000', fmt = '.3f')
plt.show()
```


    
![png](output_56_0.png)
    


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




    55.619929987460424



Removal of 'tc' column has lead to worse performance

## Pair Plots


```python
sns.pairplot(diabetes)
plt.show()
```


    
![png](output_62_0.png)
    


## A Simple function to perform EDA - fasteda

The fast_eda from fasteda package does all the above EDA analysis in single step


```python
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

# Now run fast_eda(diabetes) function
fast_eda(diabetes)

```

    [32m[1mDataFrame Head:[0m
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>0.0</td>
      <td>32.1</td>
      <td>101.0</td>
      <td>157.0</td>
      <td>93.2</td>
      <td>38.0</td>
      <td>4.0</td>
      <td>4.8598</td>
      <td>87.0</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.0</td>
      <td>1.0</td>
      <td>21.6</td>
      <td>87.0</td>
      <td>183.0</td>
      <td>103.2</td>
      <td>70.0</td>
      <td>3.0</td>
      <td>3.8918</td>
      <td>69.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72.0</td>
      <td>0.0</td>
      <td>30.5</td>
      <td>93.0</td>
      <td>156.0</td>
      <td>93.6</td>
      <td>41.0</td>
      <td>4.0</td>
      <td>4.6728</td>
      <td>85.0</td>
      <td>141.0</td>
    </tr>
  </tbody>
</table>
</div>


    [32m[1mDataFrame Tail:[0m
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>tc</th>
      <th>ldl</th>
      <th>hdl</th>
      <th>tch</th>
      <th>ltg</th>
      <th>glu</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>439</th>
      <td>60.0</td>
      <td>0.0</td>
      <td>24.9</td>
      <td>99.67</td>
      <td>162.0</td>
      <td>106.6</td>
      <td>43.0</td>
      <td>3.77</td>
      <td>4.1271</td>
      <td>95.0</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>440</th>
      <td>36.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>95.00</td>
      <td>201.0</td>
      <td>125.2</td>
      <td>42.0</td>
      <td>4.79</td>
      <td>5.1299</td>
      <td>85.0</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>36.0</td>
      <td>1.0</td>
      <td>19.6</td>
      <td>71.00</td>
      <td>250.0</td>
      <td>133.2</td>
      <td>97.0</td>
      <td>3.00</td>
      <td>4.5951</td>
      <td>92.0</td>
      <td>57.0</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    [31m[1mMissing values:[0m
    


<style type="text/css">
#T_db6d0_row0_col0, #T_db6d0_row1_col0, #T_db6d0_row2_col0 {
  background-color: #000000;
  color: #ff0000;
  font-weight: bold;
}
</style>
<table id="T_db6d0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_db6d0_level0_col0" class="col_heading level0 col0" >0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_db6d0_level0_row0" class="row_heading level0 row0" >bmi</th>
      <td id="T_db6d0_row0_col0" class="data row0 col0" >22</td>
    </tr>
    <tr>
      <th id="T_db6d0_level0_row1" class="row_heading level0 row1" >tc</th>
      <td id="T_db6d0_row1_col0" class="data row1 col0" >22</td>
    </tr>
    <tr>
      <th id="T_db6d0_level0_row2" class="row_heading level0 row2" >tch</th>
      <td id="T_db6d0_row2_col0" class="data row2 col0" >22</td>
    </tr>
  </tbody>
</table>



    ----------------------------------------------------------------------------------------------------
    [31m[1mMSNO Matrix:[0m
    
    


    
![png](output_65_7.png)
    


    ----------------------------------------------------------------------------------------------------
    [33m[1mShape of DataFrame:[0m
    
    (442, 11)
    
    ----------------------------------------------------------------------------------------------------
    [32m[1mDataFrame Info:[0m
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 442 entries, 0 to 441
    Data columns (total 11 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   age     442 non-null    float64
     1   sex     442 non-null    float64
     2   bmi     420 non-null    float64
     3   bp      442 non-null    float64
     4   tc      420 non-null    float64
     5   ldl     442 non-null    float64
     6   hdl     442 non-null    float64
     7   tch     420 non-null    float64
     8   ltg     442 non-null    float64
     9   glu     442 non-null    float64
     10  target  442 non-null    float64
    dtypes: float64(11)
    memory usage: 38.1 KB
    ----------------------------------------------------------------------------------------------------
    [34m[1mDescribe DataFrame:[0m
    
    


<style type="text/css">
#T_617d1_row0_col0, #T_617d1_row1_col0, #T_617d1_row3_col0, #T_617d1_row4_col1, #T_617d1_row4_col2, #T_617d1_row4_col4, #T_617d1_row4_col5, #T_617d1_row4_col6, #T_617d1_row5_col0, #T_617d1_row6_col0, #T_617d1_row8_col0, #T_617d1_row9_col0, #T_617d1_row10_col0, #T_617d1_row10_col3, #T_617d1_row10_col7, #T_617d1_row10_col8 {
  background-color: #5e4fa2;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row0_col1, #T_617d1_row6_col2, #T_617d1_row6_col6, #T_617d1_row10_col4 {
  background-color: #f99153;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row0_col2, #T_617d1_row0_col6 {
  background-color: #fa9656;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row0_col3 {
  background-color: #e95c47;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row0_col4 {
  background-color: #f36b43;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row0_col5 {
  background-color: #f7814c;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row0_col7 {
  background-color: #fb9d59;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row0_col8 {
  background-color: #f67c4a;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row0_col9, #T_617d1_row0_col10, #T_617d1_row1_col9, #T_617d1_row1_col10, #T_617d1_row3_col10, #T_617d1_row8_col10, #T_617d1_row10_col10 {
  color: #ff0000;
  background-color: #000000;
  font-weight: bold;
}
#T_617d1_row1_col1, #T_617d1_row1_col2, #T_617d1_row1_col3, #T_617d1_row1_col4, #T_617d1_row1_col5, #T_617d1_row1_col6, #T_617d1_row1_col7, #T_617d1_row1_col8, #T_617d1_row2_col0, #T_617d1_row4_col0, #T_617d1_row7_col0, #T_617d1_row8_col3 {
  background-color: #9e0142;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row2_col1 {
  background-color: #e1504b;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row2_col2, #T_617d1_row2_col6, #T_617d1_row2_col7 {
  background-color: #df4e4b;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row2_col3 {
  background-color: #ba2049;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row2_col4 {
  background-color: #ef6645;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row2_col5, #T_617d1_row9_col3 {
  background-color: #e2514a;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row2_col8 {
  background-color: #da464d;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row2_col9, #T_617d1_row2_col10, #T_617d1_row3_col9, #T_617d1_row4_col9, #T_617d1_row4_col10, #T_617d1_row5_col9, #T_617d1_row5_col10, #T_617d1_row6_col9, #T_617d1_row6_col10, #T_617d1_row7_col9, #T_617d1_row7_col10, #T_617d1_row8_col9, #T_617d1_row9_col9, #T_617d1_row9_col10, #T_617d1_row10_col9 {
  color: #00ff00;
  background-color: #000000;
  font-weight: bold;
}
#T_617d1_row3_col1, #T_617d1_row3_col2, #T_617d1_row3_col6 {
  background-color: #fffebe;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row3_col3 {
  background-color: #eb6046;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row3_col4 {
  background-color: #cfec9d;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row3_col5 {
  background-color: #fcfeba;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row3_col7 {
  background-color: #fffdbc;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row3_col8 {
  background-color: #fed683;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row4_col3 {
  background-color: #feefa3;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row4_col7 {
  background-color: #5956a5;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row4_col8 {
  background-color: #4199b6;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row5_col1 {
  background-color: #dff299;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row5_col2, #T_617d1_row5_col6 {
  background-color: #e4f498;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row5_col3 {
  background-color: #feda86;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row5_col4 {
  background-color: #fee999;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row5_col5 {
  background-color: #eaf79e;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row5_col7 {
  background-color: #d1ed9c;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row5_col8 {
  background-color: #aadca4;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row6_col1 {
  background-color: #f99355;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row6_col3 {
  background-color: #e85b48;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row6_col4 {
  background-color: #f67f4b;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row6_col5 {
  background-color: #f88950;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row6_col7 {
  background-color: #fa9b58;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row6_col8 {
  background-color: #fba35c;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row7_col1, #T_617d1_row7_col2, #T_617d1_row7_col5, #T_617d1_row7_col6, #T_617d1_row7_col7, #T_617d1_row8_col7 {
  background-color: #a70b44;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row7_col3 {
  background-color: #a20643;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row7_col4, #T_617d1_row8_col1, #T_617d1_row8_col2, #T_617d1_row8_col6 {
  background-color: #a90d45;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row7_col8, #T_617d1_row8_col5 {
  background-color: #ab0f45;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row8_col4 {
  background-color: #af1446;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row8_col8 {
  background-color: #a40844;
  color: #f1f1f1;
  font-weight: bold;
}
#T_617d1_row9_col1 {
  background-color: #fffab6;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row9_col2, #T_617d1_row9_col6 {
  background-color: #fffbb8;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row9_col4 {
  background-color: #e6f598;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row9_col5 {
  background-color: #fefebd;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row9_col7 {
  background-color: #fff2aa;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row9_col8 {
  background-color: #feca79;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row10_col1 {
  background-color: #62bda7;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row10_col2, #T_617d1_row10_col6 {
  background-color: #84cea5;
  color: #000000;
  font-weight: bold;
}
#T_617d1_row10_col5 {
  background-color: #f8fcb4;
  color: #000000;
  font-weight: bold;
}
</style>
<table id="T_617d1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_617d1_level0_col0" class="col_heading level0 col0" >count</th>
      <th id="T_617d1_level0_col1" class="col_heading level0 col1" >mean</th>
      <th id="T_617d1_level0_col2" class="col_heading level0 col2" >median</th>
      <th id="T_617d1_level0_col3" class="col_heading level0 col3" >std</th>
      <th id="T_617d1_level0_col4" class="col_heading level0 col4" >min</th>
      <th id="T_617d1_level0_col5" class="col_heading level0 col5" >25%</th>
      <th id="T_617d1_level0_col6" class="col_heading level0 col6" >50%</th>
      <th id="T_617d1_level0_col7" class="col_heading level0 col7" >75%</th>
      <th id="T_617d1_level0_col8" class="col_heading level0 col8" >max</th>
      <th id="T_617d1_level0_col9" class="col_heading level0 col9" >skewness</th>
      <th id="T_617d1_level0_col10" class="col_heading level0 col10" >kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_617d1_level0_row0" class="row_heading level0 row0" >age</th>
      <td id="T_617d1_row0_col0" class="data row0 col0" >442</td>
      <td id="T_617d1_row0_col1" class="data row0 col1" >48.518</td>
      <td id="T_617d1_row0_col2" class="data row0 col2" >50</td>
      <td id="T_617d1_row0_col3" class="data row0 col3" >13.109</td>
      <td id="T_617d1_row0_col4" class="data row0 col4" >19</td>
      <td id="T_617d1_row0_col5" class="data row0 col5" >38.25</td>
      <td id="T_617d1_row0_col6" class="data row0 col6" >50</td>
      <td id="T_617d1_row0_col7" class="data row0 col7" >59</td>
      <td id="T_617d1_row0_col8" class="data row0 col8" >79</td>
      <td id="T_617d1_row0_col9" class="data row0 col9" >-0.219726</td>
      <td id="T_617d1_row0_col10" class="data row0 col10" >-0.714041</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row1" class="row_heading level0 row1" >sex</th>
      <td id="T_617d1_row1_col0" class="data row1 col0" >442</td>
      <td id="T_617d1_row1_col1" class="data row1 col1" >0.532</td>
      <td id="T_617d1_row1_col2" class="data row1 col2" >1</td>
      <td id="T_617d1_row1_col3" class="data row1 col3" >0.5</td>
      <td id="T_617d1_row1_col4" class="data row1 col4" >0</td>
      <td id="T_617d1_row1_col5" class="data row1 col5" >0</td>
      <td id="T_617d1_row1_col6" class="data row1 col6" >1</td>
      <td id="T_617d1_row1_col7" class="data row1 col7" >1</td>
      <td id="T_617d1_row1_col8" class="data row1 col8" >1</td>
      <td id="T_617d1_row1_col9" class="data row1 col9" >-0.085793</td>
      <td id="T_617d1_row1_col10" class="data row1 col10" >-1.992640</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row2" class="row_heading level0 row2" >bmi</th>
      <td id="T_617d1_row2_col0" class="data row2 col0" >420</td>
      <td id="T_617d1_row2_col1" class="data row2 col1" >26.358</td>
      <td id="T_617d1_row2_col2" class="data row2 col2" >25.7</td>
      <td id="T_617d1_row2_col3" class="data row2 col3" >4.405</td>
      <td id="T_617d1_row2_col4" class="data row2 col4" >18</td>
      <td id="T_617d1_row2_col5" class="data row2 col5" >23.175</td>
      <td id="T_617d1_row2_col6" class="data row2 col6" >25.7</td>
      <td id="T_617d1_row2_col7" class="data row2 col7" >29.325</td>
      <td id="T_617d1_row2_col8" class="data row2 col8" >42.2</td>
      <td id="T_617d1_row2_col9" class="data row2 col9" >0.582185</td>
      <td id="T_617d1_row2_col10" class="data row2 col10" >0.059985</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row3" class="row_heading level0 row3" >bp</th>
      <td id="T_617d1_row3_col0" class="data row3 col0" >442</td>
      <td id="T_617d1_row3_col1" class="data row3 col1" >94.647</td>
      <td id="T_617d1_row3_col2" class="data row3 col2" >93</td>
      <td id="T_617d1_row3_col3" class="data row3 col3" >13.831</td>
      <td id="T_617d1_row3_col4" class="data row3 col4" >62</td>
      <td id="T_617d1_row3_col5" class="data row3 col5" >84</td>
      <td id="T_617d1_row3_col6" class="data row3 col6" >93</td>
      <td id="T_617d1_row3_col7" class="data row3 col7" >105</td>
      <td id="T_617d1_row3_col8" class="data row3 col8" >133</td>
      <td id="T_617d1_row3_col9" class="data row3 col9" >0.271068</td>
      <td id="T_617d1_row3_col10" class="data row3 col10" >-0.531885</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row4" class="row_heading level0 row4" >tc</th>
      <td id="T_617d1_row4_col0" class="data row4 col0" >420</td>
      <td id="T_617d1_row4_col1" class="data row4 col1" >188.831</td>
      <td id="T_617d1_row4_col2" class="data row4 col2" >186</td>
      <td id="T_617d1_row4_col3" class="data row4 col3" >34.691</td>
      <td id="T_617d1_row4_col4" class="data row4 col4" >97</td>
      <td id="T_617d1_row4_col5" class="data row4 col5" >164</td>
      <td id="T_617d1_row4_col6" class="data row4 col6" >186</td>
      <td id="T_617d1_row4_col7" class="data row4 col7" >209</td>
      <td id="T_617d1_row4_col8" class="data row4 col8" >301</td>
      <td id="T_617d1_row4_col9" class="data row4 col9" >0.383226</td>
      <td id="T_617d1_row4_col10" class="data row4 col10" >0.226904</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row5" class="row_heading level0 row5" >ldl</th>
      <td id="T_617d1_row5_col0" class="data row5 col0" >442</td>
      <td id="T_617d1_row5_col1" class="data row5 col1" >115.439</td>
      <td id="T_617d1_row5_col2" class="data row5 col2" >113</td>
      <td id="T_617d1_row5_col3" class="data row5 col3" >30.413</td>
      <td id="T_617d1_row5_col4" class="data row5 col4" >41.6</td>
      <td id="T_617d1_row5_col5" class="data row5 col5" >96.05</td>
      <td id="T_617d1_row5_col6" class="data row5 col6" >113</td>
      <td id="T_617d1_row5_col7" class="data row5 col7" >134.5</td>
      <td id="T_617d1_row5_col8" class="data row5 col8" >242.4</td>
      <td id="T_617d1_row5_col9" class="data row5 col9" >0.430437</td>
      <td id="T_617d1_row5_col10" class="data row5 col10" >0.538215</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row6" class="row_heading level0 row6" >hdl</th>
      <td id="T_617d1_row6_col0" class="data row6 col0" >442</td>
      <td id="T_617d1_row6_col1" class="data row6 col1" >49.788</td>
      <td id="T_617d1_row6_col2" class="data row6 col2" >48</td>
      <td id="T_617d1_row6_col3" class="data row6 col3" >12.934</td>
      <td id="T_617d1_row6_col4" class="data row6 col4" >22</td>
      <td id="T_617d1_row6_col5" class="data row6 col5" >40.25</td>
      <td id="T_617d1_row6_col6" class="data row6 col6" >48</td>
      <td id="T_617d1_row6_col7" class="data row6 col7" >57.75</td>
      <td id="T_617d1_row6_col8" class="data row6 col8" >99</td>
      <td id="T_617d1_row6_col9" class="data row6 col9" >0.790610</td>
      <td id="T_617d1_row6_col10" class="data row6 col10" >0.987366</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row7" class="row_heading level0 row7" >tch</th>
      <td id="T_617d1_row7_col0" class="data row7 col0" >420</td>
      <td id="T_617d1_row7_col1" class="data row7 col1" >4.072</td>
      <td id="T_617d1_row7_col2" class="data row7 col2" >4</td>
      <td id="T_617d1_row7_col3" class="data row7 col3" >1.297</td>
      <td id="T_617d1_row7_col4" class="data row7 col4" >2</td>
      <td id="T_617d1_row7_col5" class="data row7 col5" >3</td>
      <td id="T_617d1_row7_col6" class="data row7 col6" >4</td>
      <td id="T_617d1_row7_col7" class="data row7 col7" >5</td>
      <td id="T_617d1_row7_col8" class="data row7 col8" >9.09</td>
      <td id="T_617d1_row7_col9" class="data row7 col9" >0.737344</td>
      <td id="T_617d1_row7_col10" class="data row7 col10" >0.444940</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row8" class="row_heading level0 row8" >ltg</th>
      <td id="T_617d1_row8_col0" class="data row8 col0" >442</td>
      <td id="T_617d1_row8_col1" class="data row8 col1" >4.641</td>
      <td id="T_617d1_row8_col2" class="data row8 col2" >4.62</td>
      <td id="T_617d1_row8_col3" class="data row8 col3" >0.522</td>
      <td id="T_617d1_row8_col4" class="data row8 col4" >3.258</td>
      <td id="T_617d1_row8_col5" class="data row8 col5" >4.277</td>
      <td id="T_617d1_row8_col6" class="data row8 col6" >4.62</td>
      <td id="T_617d1_row8_col7" class="data row8 col7" >4.997</td>
      <td id="T_617d1_row8_col8" class="data row8 col8" >6.107</td>
      <td id="T_617d1_row8_col9" class="data row8 col9" >0.300617</td>
      <td id="T_617d1_row8_col10" class="data row8 col10" >-0.160402</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row9" class="row_heading level0 row9" >glu</th>
      <td id="T_617d1_row9_col0" class="data row9 col0" >442</td>
      <td id="T_617d1_row9_col1" class="data row9 col1" >91.26</td>
      <td id="T_617d1_row9_col2" class="data row9 col2" >91</td>
      <td id="T_617d1_row9_col3" class="data row9 col3" >11.496</td>
      <td id="T_617d1_row9_col4" class="data row9 col4" >58</td>
      <td id="T_617d1_row9_col5" class="data row9 col5" >83.25</td>
      <td id="T_617d1_row9_col6" class="data row9 col6" >91</td>
      <td id="T_617d1_row9_col7" class="data row9 col7" >98</td>
      <td id="T_617d1_row9_col8" class="data row9 col8" >124</td>
      <td id="T_617d1_row9_col9" class="data row9 col9" >0.220172</td>
      <td id="T_617d1_row9_col10" class="data row9 col10" >0.253283</td>
    </tr>
    <tr>
      <th id="T_617d1_level0_row10" class="row_heading level0 row10" >target</th>
      <td id="T_617d1_row10_col0" class="data row10 col0" >442</td>
      <td id="T_617d1_row10_col1" class="data row10 col1" >152.133</td>
      <td id="T_617d1_row10_col2" class="data row10 col2" >140.5</td>
      <td id="T_617d1_row10_col3" class="data row10 col3" >77.093</td>
      <td id="T_617d1_row10_col4" class="data row10 col4" >25</td>
      <td id="T_617d1_row10_col5" class="data row10 col5" >87</td>
      <td id="T_617d1_row10_col6" class="data row10 col6" >140.5</td>
      <td id="T_617d1_row10_col7" class="data row10 col7" >211.5</td>
      <td id="T_617d1_row10_col8" class="data row10 col8" >346</td>
      <td id="T_617d1_row10_col9" class="data row10 col9" >0.430462</td>
      <td id="T_617d1_row10_col10" class="data row10 col10" >-0.876956</td>
    </tr>
  </tbody>
</table>



    ----------------------------------------------------------------------------------------------------
    [34m[1mDataFrame Correlation:[0m
    
    


    
![png](output_65_11.png)
    


    ----------------------------------------------------------------------------------------------------
    [34m[1mDataFrame Pairplot:[0m
    
    


    
![png](output_65_13.png)
    


    ----------------------------------------------------------------------------------------------------
    [33m[1mHistogram(s) & Boxplot(s):[0m
    
    


    
![png](output_65_15.png)
    



    
![png](output_65_16.png)
    



    
![png](output_65_17.png)
    



    
![png](output_65_18.png)
    



    
![png](output_65_19.png)
    



    
![png](output_65_20.png)
    



    
![png](output_65_21.png)
    



    
![png](output_65_22.png)
    



    
![png](output_65_23.png)
    



    
![png](output_65_24.png)
    


    ----------------------------------------------------------------------------------------------------
    [33m[1mCountplot(s):[0m
    
    


    
![png](output_65_26.png)
    


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

    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
    
    :Summary Statistics:
    
    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================
    
    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    |details-start|
    **References**
    |details-split|
    
    - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
      Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
      Mathematical Statistics" (John Wiley, NY, 1950).
    - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
      (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
    - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
      Structure and Classification Rule for Recognition in Partially Exposed
      Environments".  IEEE Transactions on Pattern Analysis and Machine
      Intelligence, Vol. PAMI-2, No. 1, 67-71.
    - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
      on Information Theory, May 1972, 431-433.
    - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
      conceptual clustering system finds 3 classes in the data.
    - Many, many more ...
    
    |details-end|
    
    


```python
iris['target'].sample(5)
```




    121    2.0
    16     0.0
    19     0.0
    23     0.0
    8      0.0
    Name: target, dtype: float64




```python
class_names = dict(zip(list(map(float, range(len(iris_sklearn['target_names'])))), iris_sklearn['target_names']))
print(class_names)
```

    {0.0: 'setosa', 1.0: 'versicolor', 2.0: 'virginica'}
    

## Performing EDA for classification using fasteda


```python
fast_eda(iris, target = 'target')
```

    [32m[1mDataFrame Head:[0m
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    [32m[1mDataFrame Tail:[0m
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>


    ----------------------------------------------------------------------------------------------------
    [31m[1mMissing values:[0m
    


<style type="text/css">
</style>
<table id="T_affcf">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_affcf_level0_col0" class="col_heading level0 col0" >0</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>



    ----------------------------------------------------------------------------------------------------
    [33m[1mShape of DataFrame:[0m
    
    (150, 5)
    
    ----------------------------------------------------------------------------------------------------
    [32m[1mDataFrame Info:[0m
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   target        150 non-null    float64
    dtypes: float64(5)
    memory usage: 6.0 KB
    ----------------------------------------------------------------------------------------------------
    [34m[1mDescribe DataFrame:[0m
    
    


<style type="text/css">
#T_eeda5_row0_col0, #T_eeda5_row1_col0, #T_eeda5_row1_col3, #T_eeda5_row2_col0, #T_eeda5_row3_col0, #T_eeda5_row3_col7, #T_eeda5_row4_col0, #T_eeda5_row4_col1, #T_eeda5_row4_col2, #T_eeda5_row4_col4, #T_eeda5_row4_col5, #T_eeda5_row4_col6, #T_eeda5_row4_col8 {
  background-color: #9e0142;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row0_col1, #T_eeda5_row0_col2, #T_eeda5_row0_col4, #T_eeda5_row0_col5, #T_eeda5_row0_col6, #T_eeda5_row0_col7, #T_eeda5_row0_col8, #T_eeda5_row2_col3 {
  background-color: #5e4fa2;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row0_col3 {
  background-color: #fcaa5f;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row0_col9, #T_eeda5_row1_col9, #T_eeda5_row1_col10 {
  color: #00ff00;
  background-color: #000000;
  font-weight: bold;
}
#T_eeda5_row0_col10, #T_eeda5_row2_col9, #T_eeda5_row2_col10, #T_eeda5_row3_col9, #T_eeda5_row3_col10, #T_eeda5_row4_col10 {
  color: #ff0000;
  background-color: #000000;
  font-weight: bold;
}
#T_eeda5_row1_col1 {
  background-color: #fee797;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row1_col2, #T_eeda5_row1_col6 {
  background-color: #fee593;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row1_col4 {
  background-color: #fff5ae;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row1_col5 {
  background-color: #f3faac;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row1_col7 {
  background-color: #fdbb6c;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row1_col8 {
  background-color: #fee28f;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row2_col1 {
  background-color: #eef8a4;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row2_col2, #T_eeda5_row2_col6 {
  background-color: #acdda4;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row2_col4 {
  background-color: #f7814c;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row2_col5 {
  background-color: #fdb567;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row2_col7 {
  background-color: #9fd8a4;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row2_col8 {
  background-color: #56b0ad;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row3_col1 {
  background-color: #b41947;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row3_col2, #T_eeda5_row3_col6 {
  background-color: #c1274a;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row3_col3 {
  background-color: #f88950;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row3_col4 {
  background-color: #a90d45;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row3_col5 {
  background-color: #be254a;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row3_col8 {
  background-color: #cb334d;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row4_col3 {
  background-color: #fca55d;
  color: #000000;
  font-weight: bold;
}
#T_eeda5_row4_col7 {
  background-color: #b61b48;
  color: #f1f1f1;
  font-weight: bold;
}
#T_eeda5_row4_col9 {
  color: #FFFFFF;
  background-color: #000000;
  font-weight: bold;
}
</style>
<table id="T_eeda5">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_eeda5_level0_col0" class="col_heading level0 col0" >count</th>
      <th id="T_eeda5_level0_col1" class="col_heading level0 col1" >mean</th>
      <th id="T_eeda5_level0_col2" class="col_heading level0 col2" >median</th>
      <th id="T_eeda5_level0_col3" class="col_heading level0 col3" >std</th>
      <th id="T_eeda5_level0_col4" class="col_heading level0 col4" >min</th>
      <th id="T_eeda5_level0_col5" class="col_heading level0 col5" >25%</th>
      <th id="T_eeda5_level0_col6" class="col_heading level0 col6" >50%</th>
      <th id="T_eeda5_level0_col7" class="col_heading level0 col7" >75%</th>
      <th id="T_eeda5_level0_col8" class="col_heading level0 col8" >max</th>
      <th id="T_eeda5_level0_col9" class="col_heading level0 col9" >skewness</th>
      <th id="T_eeda5_level0_col10" class="col_heading level0 col10" >kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_eeda5_level0_row0" class="row_heading level0 row0" >sepal_length</th>
      <td id="T_eeda5_row0_col0" class="data row0 col0" >150</td>
      <td id="T_eeda5_row0_col1" class="data row0 col1" >5.843</td>
      <td id="T_eeda5_row0_col2" class="data row0 col2" >5.8</td>
      <td id="T_eeda5_row0_col3" class="data row0 col3" >0.828</td>
      <td id="T_eeda5_row0_col4" class="data row0 col4" >4.3</td>
      <td id="T_eeda5_row0_col5" class="data row0 col5" >5.1</td>
      <td id="T_eeda5_row0_col6" class="data row0 col6" >5.8</td>
      <td id="T_eeda5_row0_col7" class="data row0 col7" >6.4</td>
      <td id="T_eeda5_row0_col8" class="data row0 col8" >7.9</td>
      <td id="T_eeda5_row0_col9" class="data row0 col9" >0.311753</td>
      <td id="T_eeda5_row0_col10" class="data row0 col10" >-0.573568</td>
    </tr>
    <tr>
      <th id="T_eeda5_level0_row1" class="row_heading level0 row1" >sepal_width</th>
      <td id="T_eeda5_row1_col0" class="data row1 col0" >150</td>
      <td id="T_eeda5_row1_col1" class="data row1 col1" >3.057</td>
      <td id="T_eeda5_row1_col2" class="data row1 col2" >3</td>
      <td id="T_eeda5_row1_col3" class="data row1 col3" >0.436</td>
      <td id="T_eeda5_row1_col4" class="data row1 col4" >2</td>
      <td id="T_eeda5_row1_col5" class="data row1 col5" >2.8</td>
      <td id="T_eeda5_row1_col6" class="data row1 col6" >3</td>
      <td id="T_eeda5_row1_col7" class="data row1 col7" >3.3</td>
      <td id="T_eeda5_row1_col8" class="data row1 col8" >4.4</td>
      <td id="T_eeda5_row1_col9" class="data row1 col9" >0.315767</td>
      <td id="T_eeda5_row1_col10" class="data row1 col10" >0.180976</td>
    </tr>
    <tr>
      <th id="T_eeda5_level0_row2" class="row_heading level0 row2" >petal_length</th>
      <td id="T_eeda5_row2_col0" class="data row2 col0" >150</td>
      <td id="T_eeda5_row2_col1" class="data row2 col1" >3.758</td>
      <td id="T_eeda5_row2_col2" class="data row2 col2" >4.35</td>
      <td id="T_eeda5_row2_col3" class="data row2 col3" >1.765</td>
      <td id="T_eeda5_row2_col4" class="data row2 col4" >1</td>
      <td id="T_eeda5_row2_col5" class="data row2 col5" >1.6</td>
      <td id="T_eeda5_row2_col6" class="data row2 col6" >4.35</td>
      <td id="T_eeda5_row2_col7" class="data row2 col7" >5.1</td>
      <td id="T_eeda5_row2_col8" class="data row2 col8" >6.9</td>
      <td id="T_eeda5_row2_col9" class="data row2 col9" >-0.272128</td>
      <td id="T_eeda5_row2_col10" class="data row2 col10" >-1.395536</td>
    </tr>
    <tr>
      <th id="T_eeda5_level0_row3" class="row_heading level0 row3" >petal_width</th>
      <td id="T_eeda5_row3_col0" class="data row3 col0" >150</td>
      <td id="T_eeda5_row3_col1" class="data row3 col1" >1.199</td>
      <td id="T_eeda5_row3_col2" class="data row3 col2" >1.3</td>
      <td id="T_eeda5_row3_col3" class="data row3 col3" >0.762</td>
      <td id="T_eeda5_row3_col4" class="data row3 col4" >0.1</td>
      <td id="T_eeda5_row3_col5" class="data row3 col5" >0.3</td>
      <td id="T_eeda5_row3_col6" class="data row3 col6" >1.3</td>
      <td id="T_eeda5_row3_col7" class="data row3 col7" >1.8</td>
      <td id="T_eeda5_row3_col8" class="data row3 col8" >2.5</td>
      <td id="T_eeda5_row3_col9" class="data row3 col9" >-0.101934</td>
      <td id="T_eeda5_row3_col10" class="data row3 col10" >-1.336067</td>
    </tr>
    <tr>
      <th id="T_eeda5_level0_row4" class="row_heading level0 row4" >target</th>
      <td id="T_eeda5_row4_col0" class="data row4 col0" >150</td>
      <td id="T_eeda5_row4_col1" class="data row4 col1" >1</td>
      <td id="T_eeda5_row4_col2" class="data row4 col2" >1</td>
      <td id="T_eeda5_row4_col3" class="data row4 col3" >0.819</td>
      <td id="T_eeda5_row4_col4" class="data row4 col4" >0</td>
      <td id="T_eeda5_row4_col5" class="data row4 col5" >0</td>
      <td id="T_eeda5_row4_col6" class="data row4 col6" >1</td>
      <td id="T_eeda5_row4_col7" class="data row4 col7" >2</td>
      <td id="T_eeda5_row4_col8" class="data row4 col8" >2</td>
      <td id="T_eeda5_row4_col9" class="data row4 col9" >0.000000</td>
      <td id="T_eeda5_row4_col10" class="data row4 col10" >-1.500000</td>
    </tr>
  </tbody>
</table>



    ----------------------------------------------------------------------------------------------------
    [34m[1mDataFrame Correlation:[0m
    
    


    
![png](output_72_9.png)
    


    ----------------------------------------------------------------------------------------------------
    [34m[1mDataFrame Pairplot:[0m
    
    


    
![png](output_72_11.png)
    


    ----------------------------------------------------------------------------------------------------
    [33m[1mHistogram(s) & Boxplot(s):[0m
    
    


    
![png](output_72_13.png)
    



    
![png](output_72_14.png)
    



    
![png](output_72_15.png)
    



    
![png](output_72_16.png)
    


    ----------------------------------------------------------------------------------------------------
    [33m[1mCountplot(s):[0m
    
    


    
![png](output_72_18.png)
    



```python
plt.axis('equal')
sns.scatterplot(iris, x='petal_width', y='sepal_width', hue='target', palette=sns.color_palette("hls", iris['target'].nunique()))
plt.show()
```


    
![png](output_73_0.png)
    



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


    
![png](output_74_0.png)
    

