# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import frommean_squared_error, mean_absolute_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#  Lets Load the datasets
rainfall_data = pd.read_csv('path/to/kenya-climate-data.csv')
temperature_data = pd.read_csv('path/to/kenya-climate-data.csv')

# Lest analyse the data sets (EDA)
print("\nTemperature Data Exploration:")
print(temperature_data.info())
print(temperature_data.describe())
print(temperature_data.head())
print(temperature_data.columns)
print(temperature_data.isnull().sum())

print("Rainfall Data Exploration:")
print(rainfall_data.info())
print(rainfall_data.describe())
print(rainfall_data.head())
print(rainfall_data.columns)
print(rainfall_data.isnull().sum())


# Rename columns for proper merging
rainfall_data.columns = ['Year', 'Month', 'Avg_Rainfall_MM']
temperature_data.columns = ['Year', 'Month', 'Avg_Temp_Celsius']


# Remove duplicates
rainfall_data.drop_duplicates(inplace=True)
temperature_data.drop_duplicates(inplace=True)

# Handle missing values 
rainfall_data.dropna(inplace=True)
temperature_data.dropna(inplace=True)

# Merge datasets based on 'Year' and 'Month'
merged_data = pd.merge(rainfall_data, temperature_data, on=['Year', 'Month'])

# Separate features (X) and target variable (y)
features = merged_data[['Year', 'Month', 'Avg_Rainfall_MM']]
target = merged_data['Avg_Temp_Celsius']

# Encode 'Month' using one-hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
month_encoded = encoder.fit_transform(features[['Month']])

# Combine encoded month with other features
encoded_cols = encoder.get_feature_names_out(['Month'])
combined_features = pd.concat([features[['Year', 'Avg_Rainfall_MM']], pd.DataFrame(month_encoded, columns=encoded_cols)], axis=1)

# Standardize features for better model performance

from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Use StandardScaler for normalization

scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# Reshape data for LSTM (samples, timesteps, features)
reshaped_data = scaled_features.reshape(scaled_features.shape[0], scaled_features.shape[1], 1)

# We now Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reshaped_data, target, test_size=0.3, random_state=int(time.time()))

# Reshape X for KNN
flat_X_train = X_train.reshape(X_train.shape[0], -1)
flat_X_test = X_test.reshape(X_test.shape[0], -1)

# Train the KNN model with appropriate hyperparameters
knn_model = KNeighborsRegressor(n_neighbors=5)  # Adjust n_neighbors as needed
knn_model.fit(flat_X_train, y_train)

# Make predictions on the test data
knn_predictions = knn_model.predict(flat_X_test)

# Evaluate KNN model performance using MSE and MAE
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_mae = mean_absolute_error(y_test, knn_predictions)
print(f'KNN - Mean Squared Error: {knn_mse}')
print(f'KNN - Mean Absolute Error: {knn_mae}')

# Define a temperature threshold for classification (consider domain knowledge)
temp_threshold = y_train.mean()

# Convert KNN regression predictions to binary classes
knn_class_preds = (knn_predictions >= temp_threshold).astype(int)
y_test_class = (y_test >= temp_threshold
# Evaluate KNN model performance using classification metrics
print("\nKNN Report:")
print(classification_report(y_test_class, knn_class_preds))  # Use knn_class_preds for consistency

# Build the LSTM model architecture
lstm_model = Sequential([
    LSTM(50, activation='ReLu', input_shape=(X_train.shape[1], 1)),  # Specify input shape
    Dense(1)  # Output layer with 1 neuron for temperature prediction
])

# Configure the training process for LSTM
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model with validation split for monitoring performance
lstm_history = lstm_model.fit(X_train, y_train, epochs=50, verbose=1, validation_split=0.3)

# Assess LSTM model performance on test data
lstm_mse = lstm_model.evaluate(X_test, y_test)
print(f'LSTM Model Evaluation - Mean Squared Error: {lstm_mse}')


# comparing the evaluations
print(f'KNN - Mean Squared Error: {knn_mse}')
print(f'KNN  - Mean Absolute Error: {knn_mae}')
print(f'LSTM - Mean Squared Error: {lstm_mse}')
n tasks

# Generate predictions on test data using the trained LSTM model
lstm_predictions = lstm_model.predict(X_test)

# Convert regression predictions to binary classes for classification # Summarize KNN and LSTM model evaluation metrics

lstm_class_preds = (lstm_predictions >= temp_threshold).astype(int)  # Use temp_threshold for consistency

# Evaluate LSTM model performance using classification metrics
print("\nLSTM Report:")
print(classification_report(y_test_class, lstm_class_preds))  # Use consistent variable names

# Visualize the model's performance
plt.figure(figsize=(18, 12))

# Plot actual vs predicted temperatures for KNN
plt.subplot(4, 4, 2)
plt.plot(y_test.values, label='The Actual data')
plt.plot(knn_class_preds, label='The Predicted data')  # Use knn_class_preds for classification plot
plt.title('KNN: Actual vs Predicted Temperatures (Class)')  # Indicate classification
plt.xlabel('My temp Samples')
plt.ylabel('Average Temperature')
plt.legend(title="Data Lines", loc="upper left")

# Plot actual vs predicted temperatures for LSTM
plt.subplot(4, 4, 10)
plt.plot(y_test.values, label='The Actual data')
plt.plot(lstm_class_preds, label='The Predicted data')  # Use lstm_class_preds for classification plot
plt.title('LSTM: Actual vs Predicted Temperatures (Class)')  # Indicate classification
plt.xlabel('My temp Samples')
plt.ylabel('Average Temperature')
plt.legend(title="Data Lines", loc="center right")

# Plot training and validation loss over epochs for LSTM
plt.subplot(2, 2, 3)
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend(title="Data Lines", loc="best")

# Explore relationships between numerical features using correlation matrix
plt.figure(figsize=(15, 10))
numeric_features = data_combined.select_dtypes(include=[np.number])
sns.heatmap(numeric_features.corr(), annot=True, cmap='viridis')
plt.title('Numerical feature Correlation')
plt.show()

# Create a pair plot to visualize relationships between all features
sns.pairplot(numeric_features)
plt.show()

