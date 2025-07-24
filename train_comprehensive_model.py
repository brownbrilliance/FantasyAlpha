# train_comprehensive_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available - will use ensemble")
except ImportError:
    print("⚠️ XGBoost not available - using Random Forest only")
    XGBOOST_AVAILABLE = False

def find_and_load_datasets():
    """Find and load all available datasets"""
    print("🔍 Scanning for available datasets...")
    
    datasets = {}
    
    # Main fantasy dataset
    main_files = ['fantasy_football_dataset_complete.csv', 'fantasy_training_data.csv', 'nfl_weekly_stats.csv']
    main_dataset = None
    
    for filename in main_files:
        if os.path.exists(filename):
            print(f"✅ Found main dataset: {filename}")
            main_dataset = pd.read_csv(filename)
            datasets['main'] = main_dataset
            break
    
    if main_dataset is None:
        print("❌ No main fantasy dataset found!")
        print("📁 Available CSV files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"   - {file}")
        return None
    
    # Look for additional datasets
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    for filename in csv_files:
        if filename not in main_files:
            try:
                df = pd.read_csv(filename)
                print(f"📊 Found additional dataset: {filename} ({len(df)} rows, {len(df.columns)} cols)")
                
                # Categorize based on filename patterns
                filename_lower = filename.lower()
                if 'defense' in filename_lower or 'def' in filename_lower:
                    datasets['defense'] = df
                elif 'team' in filename_lower and ('rating' in filename_lower or 'strength' in filename_lower):
                    datasets['team_ratings'] = df
                elif 'player' in filename_lower and ('avg' in filename_lower or 'average' in filename_lower):
                    datasets['player_averages'] = df
                elif 'schedule' in filename_lower or 'matchup' in filename_lower:
                    datasets['schedule'] = df
                else:
                    # Generic additional dataset
                    datasets[f'additional_{len(datasets)}'] = df
                    
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
    
    print(f"\n📋 Loaded datasets: {list(datasets.keys())}")
    
    # Show dataset info
    for name, df in datasets.items():
        if name == 'main':
            print(f"\n📊 Main Dataset ({name}):")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")
            if 'position' in df.columns:
                print(f"   Positions: {df['position'].value_counts().to_dict()}")
            if 'season' in df.columns:
                print(f"   Seasons: {sorted(df['season'].unique())}")
        else:
            print(f"\n📊 {name}: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Sample columns: {list(df.columns)[:5]}")
    
    return datasets

def merge_datasets(datasets):
    """Intelligently merge all datasets"""
    print("\n🔗 Merging datasets...")
    
    main_df = datasets['main'].copy()
    print(f"📊 Starting with main dataset: {len(main_df)} rows")
    
    # Standardize column names for main dataset
    column_mapping = {
        'player_display_name': 'player_name',
        'recent_team': 'team',
        'opponent_team': 'opponent'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in main_df.columns:
            main_df = main_df.rename(columns={old_col: new_col})
    
    # Merge defense ratings
    if 'defense' in datasets:
        defense_df = datasets['defense'].copy()
        print(f"🛡️ Merging defense data...")
        
        # Try different merge strategies
        merge_cols = []
        if 'team' in defense_df.columns and 'opponent' in main_df.columns:
            merge_cols = [('opponent', 'team')]
        elif 'defense_team' in defense_df.columns:
            merge_cols = [('opponent', 'defense_team')]
        
        if merge_cols:
            for main_col, def_col in merge_cols:
                try:
                    main_df = main_df.merge(
                        defense_df.add_prefix('def_'), 
                        left_on=main_col, 
                        right_on=f'def_{def_col}', 
                        how='left'
                    )
                    print(f"   ✅ Merged on {main_col} -> {def_col}")
                    break
                except Exception as e:
                    print(f"   ⚠️ Failed merge on {main_col}: {e}")
    
    # Merge team ratings
    if 'team_ratings' in datasets:
        team_df = datasets['team_ratings'].copy()
        print(f"🏈 Merging team ratings...")
        
        try:
            main_df = main_df.merge(
                team_df.add_prefix('team_'), 
                left_on='team', 
                right_on='team_team', 
                how='left'
            )
            print(f"   ✅ Merged team ratings")
        except Exception as e:
            print(f"   ⚠️ Team ratings merge failed: {e}")
    
    # Merge player averages (for historical context)
    if 'player_averages' in datasets:
        player_df = datasets['player_averages'].copy()
        print(f"👤 Merging player averages...")
        
        try:
            main_df = main_df.merge(
                player_df.add_prefix('hist_'), 
                left_on=['player_name', 'position'], 
                right_on=['hist_player_name', 'hist_position'], 
                how='left'
            )
            print(f"   ✅ Merged player historical data")
        except Exception as e:
            print(f"   ⚠️ Player averages merge failed: {e}")
    
    print(f"✅ Final merged dataset: {len(main_df)} rows, {len(main_df.columns)} columns")
    return main_df

def create_comprehensive_features(df):
    """Create comprehensive features from merged dataset"""
    print("\n🔧 Creating comprehensive features...")
    
    features_df = pd.DataFrame()
    
    # Basic offensive features
    features_df['avg_points_last_3'] = df['avg_points_last_3'].fillna(df['fantasy_points'])
    features_df['target_share'] = df['target_share'].fillna(0)
    features_df['snap_count_pct'] = df['snap_count_pct'].fillna(0.75)
    features_df['vegas_total'] = df['vegas_total'].fillna(47.0)
    
    # Environmental features
    features_df['temp'] = df['temp'].fillna(70)
    features_df['wind_speed'] = df['wind_speed'].fillna(5)
    features_df['is_dome'] = df['is_dome'].fillna(False).astype(int)
    
    # Week features (crucial for different predictions each week)
    features_df['week'] = df['week']
    features_df['week_sin'] = np.sin(2 * np.pi * df['week'] / 18)
    features_df['week_cos'] = np.cos(2 * np.pi * df['week'] / 18)
    features_df['early_season'] = (df['week'] <= 6).astype(int)
    features_df['mid_season'] = ((df['week'] > 6) & (df['week'] <= 12)).astype(int)
    features_df['late_season'] = (df['week'] > 12).astype(int)
    features_df['playoff_push'] = (df['week'] >= 15).astype(int)
    
    # Home/Away
    features_df['is_home'] = (df['home_away'] == 'home').astype(int)
    
    # Position encoding
    features_df['pos_QB'] = (df['position'] == 'QB').astype(int)
    features_df['pos_RB'] = (df['position'] == 'RB').astype(int)
    features_df['pos_WR'] = (df['position'] == 'WR').astype(int)
    features_df['pos_TE'] = (df['position'] == 'TE').astype(int)
    
    # Team strength (calculate from data)
    if 'team_rating' in df.columns:
        features_df['team_strength'] = df['team_rating'].fillna(0.5)
    else:
        team_performance = df.groupby('team')['fantasy_points'].mean()
        team_strength = (team_performance - team_performance.min()) / (team_performance.max() - team_performance.min())
        features_df['team_strength'] = df['team'].map(team_strength).fillna(0.5)
    
    # Defense ratings (from merged defense data or calculated)
    defense_cols = [col for col in df.columns if col.startswith('def_') and 'rating' in col.lower()]
    if defense_cols:
        features_df['opp_def_rating'] = df[defense_cols[0]].fillna(0.5)
        print(f"   ✅ Using defense rating: {defense_cols[0]}")
    else:
        # Calculate from data
        opp_defense = df.groupby('opponent')['fantasy_points'].mean()
        opp_def_normalized = 1 - ((opp_defense - opp_defense.min()) / (opp_defense.max() - opp_defense.min()))
        features_df['opp_def_rating'] = df['opponent'].map(opp_def_normalized).fillna(0.5)
        print(f"   📊 Calculated defense ratings from data")
    
    # Advanced defensive features (if available)
    def_pressure_cols = [col for col in df.columns if 'pressure' in col.lower() or 'sack' in col.lower()]
    if def_pressure_cols:
        features_df['def_pressure'] = df[def_pressure_cols[0]].fillna(0)
        print(f"   ✅ Added defensive pressure: {def_pressure_cols[0]}")
    
    def_coverage_cols = [col for col in df.columns if 'coverage' in col.lower() or 'int' in col.lower()]
    if def_coverage_cols:
        features_df['def_coverage'] = df[def_coverage_cols[0]].fillna(0)
        print(f"   ✅ Added defensive coverage: {def_coverage_cols[0]}")
    
    # Historical player performance (if available)
    hist_cols = [col for col in df.columns if col.startswith('hist_') and 'avg' in col.lower()]
    if hist_cols:
        features_df['hist_avg_performance'] = df[hist_cols[0]].fillna(features_df['avg_points_last_3'])
        print(f"   ✅ Added historical averages: {hist_cols[0]}")
    
    # Usage and efficiency features
    features_df['high_target_share'] = (features_df['target_share'] > 0.2).astype(int)
    features_df['high_snap_pct'] = (features_df['snap_count_pct'] > 0.8).astype(int)
    features_df['usage_score'] = features_df['target_share'] * features_df['snap_count_pct']
    
    # Game environment
    features_df['bad_weather'] = ((features_df['temp'] < 40) | 
                                 (features_df['wind_speed'] > 15) | 
                                 (features_df['is_dome'] == 0)).astype(int)
    features_df['game_environment'] = features_df['vegas_total'] * (1 - features_df['bad_weather'] * 0.1)
    
    # Matchup difficulty
    features_df['matchup_difficulty'] = features_df['opp_def_rating'] * (1 + features_df['bad_weather'] * 0.2)
    
    # Target variable and metadata
    features_df['fantasy_points'] = df['fantasy_points']
    features_df['player_name'] = df['player_name'] if 'player_name' in df.columns else df.get('player_display_name', 'Unknown')
    features_df['position'] = df['position']
    features_df['team'] = df['team'] if 'team' in df.columns else df.get('recent_team', 'Unknown')
    features_df['season'] = df['season']
    
    print(f"✅ Created {len(features_df.columns)} comprehensive features")
    
    # Show feature summary
    feature_cols = [col for col in features_df.columns 
                   if col not in ['fantasy_points', 'player_name', 'position', 'team', 'season']]
    print(f"📊 Feature categories:")
    print(f"   Basic stats: {len([c for c in feature_cols if any(x in c for x in ['avg_points', 'target', 'snap'])])}")
    print(f"   Week/timing: {len([c for c in feature_cols if 'week' in c or 'season' in c])}")
    print(f"   Environment: {len([c for c in feature_cols if any(x in c for x in ['temp', 'wind', 'dome', 'weather'])])}")
    print(f"   Team/defense: {len([c for c in feature_cols if any(x in c for x in ['team', 'def', 'opp'])])}")
    print(f"   Usage: {len([c for c in feature_cols if any(x in c for x in ['usage', 'high_'])])}")
    
    return features_df

def create_ensemble_model(position):
    """Create position-specific ensemble model"""
    if not XGBOOST_AVAILABLE:
        return RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        )
    
    # RF + XGBoost ensemble with position-specific tuning
    rf_model = RandomForestRegressor(
        n_estimators=150, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
    )
    
    # Position-specific XGBoost parameters
    xgb_params = {
        'QB': {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1},
        'RB': {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.08},
        'WR': {'n_estimators': 180, 'max_depth': 6, 'learning_rate': 0.1},
        'TE': {'n_estimators': 160, 'max_depth': 5, 'learning_rate': 0.12}
    }
    
    params = xgb_params.get(position, xgb_params['WR'])
    xgb_model = XGBRegressor(**params, random_state=42, eval_metric='rmse', verbosity=0)
    
    return VotingRegressor([('rf', rf_model), ('xgb', xgb_model)])

def train_comprehensive_models(features_df):
    """Train models with comprehensive features"""
    models = {}
    scalers = {}
    feature_names = {}
    
    feature_cols = [col for col in features_df.columns 
                   if col not in ['fantasy_points', 'player_name', 'position', 'team', 'season']]
    
    print(f"\n🤖 Training models with {len(feature_cols)} features...")
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        print(f"\n🏈 Training {position} model...")
        
        pos_data = features_df[features_df['position'] == position].copy()
        if len(pos_data) < 100:
            print(f"⚠️ Not enough {position} data ({len(pos_data)} rows)")
            continue
        
        X = pos_data[feature_cols]
        y = pos_data['fantasy_points']
        
        print(f"📊 Training on {len(X)} {position} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = create_ensemble_model(position)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        test_pred = model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"📈 {position} Performance: RMSE={test_rmse:.2f}, R²={test_r2:.3f}")
        
        # Feature importance
        if XGBOOST_AVAILABLE:
            rf_importance = model.estimators_[0].feature_importances_
        else:
            rf_importance = model.feature_importances_
        
        top_features = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_importance
        }).nlargest(5, 'importance')
        
        print(f"🔍 Top 5 {position} features:")
        for _, row in top_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        models[position] = model
        scalers[position] = scaler
        feature_names[position] = feature_cols
    
    return models, scalers, feature_names, feature_cols

def save_comprehensive_models(models, scalers, feature_names, feature_cols):
    """Save all models and create supporting files"""
    print(f"\n💾 Saving comprehensive models...")
    
    # Save position models
    joblib.dump(models, 'position_models.pkl')
    joblib.dump(scalers, 'position_scalers.pkl')
    joblib.dump(feature_names, 'position_feature_names.pkl')
    
    # Main model
    main_pos = 'RB' if 'RB' in models else list(models.keys())[0]
    joblib.dump(models[main_pos], 'fantasy_football_model.pkl')
    joblib.dump(scalers[main_pos], 'scaler.pkl')
    joblib.dump(feature_cols, 'feature_names.pkl')
    
    print(f"✅ Saved models for positions: {list(models.keys())}")
    print(f"✅ Main model: {main_pos}")
    
    return main_pos

def main():
    """Main comprehensive training pipeline"""
    print("🚀 Starting Comprehensive Fantasy Football Model Training")
    print("📊 Using ALL available datasets")
    print("=" * 70)
    
    # Find and load all datasets
    datasets = find_and_load_datasets()
    if datasets is None:
        return
    
    # Merge datasets
    merged_df = merge_datasets(datasets)
    
    # Create comprehensive features
    features_df = create_comprehensive_features(merged_df)
    
    # Train models
    models, scalers, feature_names, feature_cols = train_comprehensive_models(features_df)
    
    if not models:
        print("❌ No models trained successfully!")
        return
    
    # Save everything
    main_pos = save_comprehensive_models(models, scalers, feature_names, feature_cols)
    
    print(f"\n🎉 Comprehensive Training Complete!")
    print(f"✅ Integrated {len(datasets)} datasets")
    print(f"✅ Trained models: {list(models.keys())}")
    print(f"✅ Enhanced with defensive data")
    print(f"✅ Week-specific predictions enabled")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Restart your app: python app.py")
    print(f"2. Test different weeks - should see varying predictions!")
    print(f"3. Rankings will be much more accurate with comprehensive data!")

if __name__ == "__main__":
    main()