https://www.kaggle.com/learn/overview
# Intro to machine learning
## Key words
- decision tree regression; random forest
- model validation; train test split
- overfitting and underfitting
- optimum value of hyper parameter
## Note
- use DecisionTreeRegressor and radom forest as an example to predict house pricing in Iowa
- underfitting and overfitting (hyper parameter tuning)
![](2019-12-30-12-23-40.png)
## Sample code
### Basic
- load packages
```python
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
```
- set up X and y
    - X should be a pd data frame. y should be a pd series
```python
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
```

- train test split
```python
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```

- specify model
```python
iowa_model = DecisionTreeRegressor(random_state=1)
```

- fit model
```python
iowa_model.fit(train_X, train_y)
```

- make validation predictions and calculate MAE
```python
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
```
## Hyper-parameter tuning
- use a utility function to help compare MAE scores from different values for max_leaf_nodes
```python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```
- compare MAE with differing values of max_leaf_nodes
```python
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```
- find the ideal tree size from candidate_max_leaf_nodes, a smart way to use dictionary
```python
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
# get the key for min value
best_tree_size = min(scores, key=scores.get)
```

## Fit model using all data
```python
# set up model using tuned hyper-parameter
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model
final_model.fit(X, y)
```

## Random forest
```python
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
```

# Intermediate machine learning
## Key words
- missing values; categorical variables
- pipelines
- cross validation
- XGBoost
- leakage
## Notes and Sample codes
### Missing values
#### drop
- drop columns or rows (or sample) with missing values
    - works if like 80% of values are missing for a column
    - lose information if most value of the column is not missing
- To detect NaN values numpy uses np.isnan(). 
- To detect NaN values pandas uses either .isna() or .isnull(). They're exactly the same

```python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
```
#### imputation
- simple imputation: fill in mean, median, most frequent values
- more complex ways: 
    - regression imputer
        - should be the IteractiveImputer in sklearn
        - A strategy for imputing missing values by modeling each feature with missing values as a function of other features 
    - KNN imputer
        - Each sample’s missing values are imputed using the mean value from n_neighbors nearest neighbors found in the training set. Two samples are close if the features that neither is missing are close.

```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
```

#### An extension to imputation
- intuition: imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing.
- method: add a new column to indicate which value is missing
![](2019-12-30-15-58-39.png)

```python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
```