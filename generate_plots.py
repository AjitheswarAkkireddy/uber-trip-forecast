import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting plot generation...")

# --- Create Directories for Plots ---
# Create a 'static' folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')
# Create an 'images' subfolder inside 'static'
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# --- 1. Load and Prepare Data ---
try:
    data_path = 'data'
    file_names = [f for f in os.listdir(data_path) if f.startswith('uber-raw-data') and f.endswith('.csv')]
    
    if not file_names:
        raise FileNotFoundError("No Uber CSV files found in the 'data' directory.")

    print(f"Found {len(file_names)} files. Loading data...")
    df_list = [pd.read_csv(os.path.join(data_path, f)) for f in file_names]
    df = pd.concat(df_list, ignore_index=True)

    # Convert 'Date/Time' to datetime objects
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    # Feature Engineering for plots
    df['Hour'] = df['Date/Time'].dt.hour
    df['Day'] = df['Date/Time'].dt.day
    df['DayOfWeek'] = df['Date/Time'].dt.dayofweek
    df['Month'] = df['Date/Time'].dt.month
    
    print("Data loaded successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- 2. Generate and Save Plots ---

# Plot 1: Trips per Hour
print("Generating 'Trips per Hour' plot...")
plt.figure(figsize=(12, 7))
sns.countplot(x='Hour', data=df, palette='viridis')
plt.title('Total Uber Trips per Hour', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.savefig('static/images/trips_per_hour.png')
plt.close()

# Plot 2: Trips by Day of the Week
print("Generating 'Trips by Day of Week' plot...")
plt.figure(figsize=(12, 7))
sns.countplot(x='DayOfWeek', data=df, palette='magma')
plt.title('Total Uber Trips by Day of the Week', fontsize=16)
plt.xlabel('Day of the Week (0=Mon, 6=Sun)', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.savefig('static/images/trips_by_dayofweek.png')
plt.close()

# Plot 3: Trips by Month
print("Generating 'Trips by Month' plot...")
plt.figure(figsize=(12, 7))
sns.countplot(x='Month', data=df, palette='plasma')
plt.title('Total Uber Trips by Month', fontsize=16)
plt.xlabel('Month (4=Apr, 9=Sep)', fontsize=12)
plt.ylabel('Number of Trips', fontsize=12)
plt.savefig('static/images/trips_by_month.png')
plt.close()

# Plot 4: Heatmap of Hour vs Day
print("Generating 'Hour vs Day' heatmap...")
df_hour_day = df.groupby(['Hour', 'Day']).size().unstack()
plt.figure(figsize=(12, 8))
sns.heatmap(df_hour_day, cmap='hot', linecolor='white', linewidths=0.1)
plt.title('Heatmap of Trips by Hour and Day', fontsize=16)
plt.xlabel('Day of Month', fontsize=12)
plt.ylabel('Hour of Day', fontsize=12)
plt.savefig('static/images/heatmap_hour_day.png')
plt.close()

print("✅ All plots have been generated and saved in the 'static/images' folder.")