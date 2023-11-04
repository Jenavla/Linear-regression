from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the parameters
data = pd.read_csv('student-mat.csv', sep=',')
target_col = 'G3'
feature_cols = ['G1', 'G2']
test_size = 0.8
zscore_threshold = 2.576 # 99% interval

def perform_linear_regression(data, target_col, feature_cols, random_state=None):
    # Identify and remove outliers using Z-score
    z_scores = np.abs(stats.zscore(data[feature_cols]))
    data = data[(z_scores < zscore_threshold).all(axis=1)]

    # Split data
    y = data[target_col]
    x = data[feature_cols]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Fit a linear regression model
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    # Make predictions
    predictions = lm.predict(x_test)

    # Evaluate model
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Plot all the features against the prediction
    for feature in feature_cols:
        sns.regplot(x=x_test[feature], y=predictions, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
        plt.xlabel(feature)
        plt.ylabel("Prediction")
        plt.title(f"{feature} vs. Prediction")
        plt.show()

    # Return results, model, and coefficients
    results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'coefficients': pd.DataFrame({'': x.columns, '': lm.coef_})
    }

    return results

# Create a correlation matrix
correlation_matrix = data[feature_cols].corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.title("Correlation Matrix")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

results = perform_linear_regression(data, target_col, feature_cols, random_state=None)
print(results)
