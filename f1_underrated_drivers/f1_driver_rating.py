import fastf1
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the cache directory exists before enabling fastf1 cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

def fetch_driver_performance_data(year, rounds):
    """
    Fetch comprehensive driver performance metrics for specified races
    """
    all_data = []
    
    for round_num in rounds:
        try:
            print(f"Fetching round {round_num} of {year}...")
            
           
            race = fastf1.get_session(year, round_num, 'R')
            race.load()
            quali = fastf1.get_session(year, round_num, 'Q')
            quali.load()
            
           
            for _, driver_result in race.results.iterrows():
                try:
                    driver_number = driver_result.DriverNumber
                    team = driver_result.TeamName
                    driver_name = driver_result.FullName
                    
                 
                    teammate_data = race.results[
                        (race.results['TeamName'] == team) & 
                        (race.results['DriverNumber'] != driver_number)
                    ].iloc[0]
                    
                   
                    quali_data = quali.results[quali.results['DriverNumber'] == driver_number].iloc[0]
                    teammate_quali = quali.results[quali.results['DriverNumber'] == teammate_data['DriverNumber']].iloc[0]
                    
                    race_data = {
                        'Round': round_num,
                        'Driver': driver_name,
                        'Team': team,
                        'Points': float(driver_result.Points),
                        'QualiPosition': float(quali_data.Position),
                        'TeammateQualiPosition': float(teammate_quali.Position),
                        'GridPosition': float(driver_result.GridPosition),
                        'FinishPosition': float(driver_result.Position),
                        'TeammateFinishPosition': float(teammate_data.Position),
                        'PositionsGained': float(driver_result.GridPosition) - float(driver_result.Position),
                        'DNF': 1 if driver_result.Status != 'Finished' else 0,
                        'FastestLapPoint': 1 if getattr(driver_result, 'FastestLap', False) else 0,
                        'PitStops': len(race.laps.pick_driver(driver_number)['PitOutTime'].dropna())
                    }
                    
                    all_data.append(race_data)
                    print(f"Successfully processed data for {driver_name}")
                    
                except Exception as e:
                    print(f"Error processing driver data: {e}")
                    continue
                
        except Exception as e:
            print(f"Error processing round {round_num}: {e}")
            continue
    
    if not all_data:
        raise Exception("No data could be collected!")
        
    return pd.DataFrame(all_data)

def prepare_features(df):
    """
    Prepare features for the model, excluding team-dependent metrics and driver names
    """
    features = [
        'QualiPosition',
        'TeammateQualiPosition',
        'GridPosition',
        'TeammateFinishPosition',
        'PositionsGained',
        'DNF',
        'FastestLapPoint',
        'PitStops'
    ]
    
    X = df[features]
    y = df['Points']
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=features), y

def train_points_predictor(X, y):
    """
    Train a Random Forest model to predict points based on performance metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

def analyze_driver_ratings(df, model, X):
    """
    Compare predicted vs actual points with detailed analysis over multiple races
    """
    df['PredictedPoints'] = model.predict(X)
    df['PointsDifference'] = df['PredictedPoints'] - df['Points']
    

    driver_stats = df.groupby('Driver').agg({
        'Points': ['mean', 'sum'],
        'PredictedPoints': ['mean', 'sum'],
        'PointsDifference': ['mean', 'sum'],
        'Team': 'first'
    }).round(2)
    
    driver_stats.columns = ['AvgPoints', 'TotalPoints', 'AvgPredicted', 
                          'TotalPredicted', 'AvgDiff', 'TotalDiff', 'Team']

    driver_stats = driver_stats.sort_values('TotalDiff', ascending=False)
    
    print("\n📊 DRIVER PERFORMANCE ANALYSIS")
    print("=" * 100)
    

    print("\n🏆 TOP 3 MOST UNDERRATED DRIVERS:")
    print("-" * 50)
    for idx, row in driver_stats[driver_stats['TotalDiff'] > 0].head(3).iterrows():
        print(f"\n{idx} ({row['Team']}):")
        print(f"  Average Points per Race: {row['AvgPoints']:.1f}")
        print(f"  Average Predicted Points: {row['AvgPredicted']:.1f}")
        print(f"  Total Points: {row['TotalPoints']:.1f}")
        print(f"  Total Predicted: {row['TotalPredicted']:.1f}")
        print(f"  Underrated by: {row['TotalDiff']:.1f} points total")
        print(f"  ({row['AvgDiff']:.1f} points per race)")
    

    print("\n📉 TOP 3 MOST OVERRATED DRIVERS:")
    print("-" * 50)
    for idx, row in driver_stats[driver_stats['TotalDiff'] < 0].head(3).iterrows():
        print(f"\n{idx} ({row['Team']}):")
        print(f"  Average Points per Race: {row['AvgPoints']:.1f}")
        print(f"  Average Predicted Points: {row['AvgPredicted']:.1f}")
        print(f"  Total Points: {row['TotalPoints']:.1f}")
        print(f"  Total Predicted: {row['TotalPredicted']:.1f}")
        print(f"  Overrated by: {abs(row['TotalDiff']):.1f} points total")
        print(f"  ({abs(row['AvgDiff']):.1f} points per race)")
    

    plt.figure(figsize=(15, 10))
    

    plt.subplot(2, 1, 1)
    for driver in driver_stats.index[:5]:  # Top 5 drivers
        driver_data = df[df['Driver'] == driver]
        plt.plot(driver_data['Round'], driver_data['PointsDifference'], 
                marker='o', label=driver)
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Race Number')
    plt.ylabel('Points Difference')
    plt.title('Points Difference Trend (Top 5 Drivers)')
    plt.legend()
    

    plt.subplot(2, 1, 2)
    colors = ['g' if x > 0 else 'r' for x in driver_stats['TotalDiff']]
    plt.bar(driver_stats.index, driver_stats['TotalDiff'], color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title('Total Points Difference (Predicted - Actual)')
    plt.xlabel('Driver')
    plt.ylabel('Points Difference')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Fetching F1 training data...")
    all_data = []
    
    try:

        print("\nFetching 2023 race data (last 7 races)...")
        for round_num in range(16, 23):  # Races 16-22 from 2023
            try:
                data_2023 = fetch_driver_performance_data(2023, [round_num])
                if data_2023 is not None and not data_2023.empty:
                    all_data.append(data_2023)
                    print(f"Successfully added 2023 Round {round_num}")
            except Exception as e:
                print(f"Couldn't fetch 2023 Round {round_num}: {e}")
        

        print("\nFetching 2024 race data (first 3 races)...")
        for round_num in range(1, 4):
            try:
                data_2024 = fetch_driver_performance_data(2024, [round_num])
                if data_2024 is not None and not data_2024.empty:
                    all_data.append(data_2024)
                    print(f"Successfully added 2024 Round {round_num}")
            except Exception as e:
                print(f"Couldn't fetch 2024 Round {round_num}: {e}")
        
        if all_data:
            # Combine all the data
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal races collected: {len(combined_data['Round'].unique())}")
            print(f"Total driver entries: {len(combined_data)}")
            
            # Prepare features and train model
            X, y = prepare_features(combined_data)
            model = train_points_predictor(X, y)
            
            # Analyze driver ratings
            print("\nAnalyzing driver performance across all collected races...")
            analyze_driver_ratings(combined_data, model, X)
            
        else:
            print("Could not fetch enough data!")
            
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main() 