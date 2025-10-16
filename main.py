import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.training import train_model
from scripts.evaluation import evaluate_model, plot_correlation_matrix, plot_distribution
from scripts.prediction import predict, save_predictions
from scripts.feature_engineering import preprocess_data, add_rolling_averages, shift_target, add_season_flags
from scripts.model import FantasyFootballLSTM
import torch
import numpy as np

# Define paths
data_path = 'data/cleaned_fantasy_football_data.xlsx'
results_path = 'results'
model_path = 'models/trained_model.pth'
predictions_path = os.path.join(results_path, 'predictions_2025.xlsx')

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load and preprocess the data
print("Loading and preprocessing data...")
df = pd.read_excel(data_path)
df = preprocess_data(df)
df = add_rolling_averages(df)
df = add_season_flags(df)
# Fill NaNs after feature engineering to prevent issues in training/evaluation
df.fillna(0, inplace=True)

# Save 2024 data before shifting the target
df_2024 = df[df['Year'] == 2024].copy()
player_names_2024 = df_2024['Player'].reset_index(drop=True)
df_2024 = df_2024.drop(columns=['Player'])

# Apply shift_target to remove rows with NaN target values
df = shift_target(df)

# Reset the index to ensure sequential indexing
df = df.reset_index(drop=True)

# Define features and target
feature_names = ['Year', 'Age', 'G', 'Tgt', 'Rec', 'RecYds', 'RecTD', 'TD/G', 'RecYds/G', 'FantPtHalf/G', 'Tgt/G', 'FantPtHalf/GLast2Y',  'Tgt/GLast2Y', 'RecYds/GLast2Y', '#ofY']
target = 'NextYearFantPt/G'

# Split the data into training, validation, and test sets
print("Splitting data into train/val/test sets...")
X = df[feature_names]
y = df[target]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_2024 = df_2024[feature_names]

# Standardize the data
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_2024 = scaler.transform(X_2024)

# Ensure X_train, X_val, X_test, and X_2024 are correctly shaped for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_2024 = X_2024.reshape(X_2024.shape[0], 1, X_2024.shape[1])

# Train the model and save the best model
print("Training LSTM model...")
input_dim = X_train.shape[2]
hidden_dim = 128  # Set hidden dimension
num_layers = 2  # Set number of LSTM layers
model = FantasyFootballLSTM(input_dim, hidden_dim, num_layers)

best_model_path = 'models/trained_model.pth'
train_model(model, X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.001, model_path=best_model_path)

# Load the best model
print("Loading best model...")
model.load_state_dict(torch.load(best_model_path))

# Evaluate the model
print("\nEvaluating model performance...")
evaluate_model(model, X_test, y_test, X_train, y_train)

# Plot correlation matrix
print("Generating correlation matrix...")
plot_correlation_matrix(df, feature_names + [target], results_path)

# Make predictions for the test data
y_test_pred = predict(model, X_test)

# Make predictions for 2024 data
print("\nGenerating predictions for 2025...")
predictions_2024 = predict(model, X_2024)
predictions_df = pd.DataFrame(predictions_2024, columns=['Predicted_FantPt/G'])

# Ensure player names are included in predictions
predictions_df['Player'] = player_names_2024

# Round predictions to 1 decimal place
predictions_df['Predicted_FantPt/G'] = predictions_df['Predicted_FantPt/G'].round(1)

# Sort by predicted fantasy points per game in descending order
predictions_df = predictions_df.sort_values(by='Predicted_FantPt/G', ascending=False).reset_index(drop=True)

# Save to file
save_predictions(predictions_df, predictions_path)
print(f"Predictions for 2025 saved to '{predictions_path}'")

# Plot distribution curve for actual vs predicted values
print("Generating distribution curve...")
plot_distribution(y_test, y_test_pred, results_path)

print(f"\nAll results saved to '{results_path}/'")
print("\nMVP execution complete!")