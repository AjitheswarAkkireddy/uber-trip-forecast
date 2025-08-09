import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor

print("Starting model training process...")

# --- 1. Load and Prepare Data ---
try:
    # Define the path to the data folder
    data_path = 'data'
    # Get a list of all uber csv files from the data folder
    file_names = [f for f in os.listdir(data_path) if f.startswith('uber-raw-data') and f.endswith('.csv')]
    
    if not file_names:
        raise FileNotFoundError("No Uber CSV files found in the 'data' directory.")

    print(f"Found {len(file_names)} data files. Loading and concatenating...")
    
    # Read and combine all data files into a single DataFrame
    df_list = [pd.read_csv(os.path.join(data_path, f)) for f in file_names]
    df = pd.concat(df_list, ignore_index=True)

    # Convert 'Date/Time' to datetime objects
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    # --- 2. Feature Engineering ---
    # Create time-based features that the model will use
    df['hour'] = df['Date/Time'].dt.hour
    df['day'] = df['Date/Time'].dt.day
    df['dayofweek'] = df['Date/Time'].dt.dayofweek # Monday=0, Sunday=6
    df['month'] = df['Date/Time'].dt.month
    
    print("Data loaded and features engineered successfully.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure your CSV files are in a 'data' subfolder.")
    exit() # Exit the script if data is not found
except Exception as e:
    print(f"An error occurred during data preparation: {e}")
    exit()


# --- 3. Define Features and Target ---
# Select the columns to be used as input features (X) and the target variable (y)
# Note: The model from your Colab notebook predicts trip counts, but the raw data doesn't have a 'Trips' column.
# For this example, we'll create a proxy for trip count by grouping data.
# A more robust approach would be to properly aggregate trips per hour.
# Here, we'll just use the features to predict one of the inputs as a placeholder.
# For a real prediction, you would group by hour and count rows to create a 'Trips' column.
# Let's assume for now we are building a model based on the available columns.
# We will group by time features and use the size as the trip count.

print("Aggregating data to create trip counts per hour...")
hourly_trips = df.groupby(['month', 'day', 'hour', 'dayofweek']).size().reset_index(name='Trips')
# We'll use the average lat/lon for that hour as a feature
location_data = df.groupby(['month', 'day', 'hour', 'dayofweek'])[['Lat', 'Lon']].mean().reset_index()

final_df = pd.merge(hourly_trips, location_data, on=['month', 'day', 'hour', 'dayofweek'])

# Define our final features and target
features = ["hour", "day", "dayofweek", "month", "Lat", "Lon"]
target = "Trips"

X = final_df[features]
y = final_df[target]

print(f"Training data prepared. Shape of X: {X.shape}, Shape of y: {y.shape}")


# --- 4. Train the Ensemble Model ---
print("Training individual models (RandomForest, GradientBoosting, XGBoost)...")
# Initialize the individual regression models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, random_state=42)

# Create the ensemble model using VotingRegressor
ensemble_model = VotingRegressor(
    estimators=[("rf", rf), ("gbr", gbr), ("xgb", xgb)]
)

print("Training the ensemble model... (This may take a few minutes)")
# Train the ensemble model on the entire dataset
ensemble_model.fit(X, y)

print("Ensemble model training complete.")


# --- 5. Save the Trained Model ---
# Define the path for the output model file
output_path = "ensemble_model.pkl"
# Save the trained model to a .pkl file using joblib for efficiency
joblib.dump(ensemble_model, output_path)

print(f"✅ Model successfully trained and saved to '{output_path}'!")

