# GRNN model applied to CBOT SBO closing-prices
The GNN model code applied to CBOT Soybean oil listed contracts incorporates the walk-forward cross-validation with purging &amp; embargo, as well as the additional evaluation metrics. It evaluates the model's performance using various evaluation metrics and visualizes the predicted and actual closing prices.

**More technically.**
The provided code performs the following steps:

Imports the required libraries, including pandas, numpy, TensorFlow, matplotlib, seaborn, and scikit-learn modules.
Loads the dataset from the given URL using pandas read_csv function.
Filters the data to include only the desired period (2014-2016) based on the start and end dates.
Extracts the closing prices from the filtered data.
Normalizes the closing prices between 0 and 1 using the MinMaxScaler from scikit-learn.
Defines a function impute_missing_values that imputes missing values in the data using the k-nearest neighbors (KNN) algorithm.
Defines a function create_dataset that creates a time series dataset for training the neural network. The function takes a sequence of data and a window size, and returns input features (X) and corresponding target values (y).
Sets the window size for the time series data.
Defines the walk-forward cross-validation with purging and embargo using the TimeSeriesSplit function from scikit-learn.
Initializes lists for storing the evaluation metrics: RMSE, MAE, R2, MSLE, MPE, MAPE, and explained variance score.
Iterates over the cross-validation splits and performs the following steps for each split:
Applies purging and embargo to obtain the training and testing data.
Imputes missing values in the training and testing data using the impute_missing_values function.
Creates the training and testing datasets using the create_dataset function.
Defines a neural network model using the Sequential API of TensorFlow.
Compiles the model using the Adam optimizer and mean squared error loss function.
Trains the model on the training dataset.
Makes predictions on the testing dataset.
Denormalizes the predicted and actual closing prices using the inverse_transform method of the MinMaxScaler.
Calculates the evaluation metrics: RMSE, MAE, R2, MSLE, MPE, MAPE, and explained variance score.
Appends the scores to the corresponding lists.
Calculates the average scores across all cross-validation folds.
Visualizes the performance by plotting the predicted and actual closing prices using a line plot.
Prints the average scores.
