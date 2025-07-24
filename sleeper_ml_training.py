"""
Sleeper Fantasy Football ML Model Training Pipeline
This script creates and trains machine learning models for fantasy football predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import requests
import json
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class SleeperMLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.label_encoders = {}
        
    def fetch_sleeper_data(self, week_start=1, week_end=17, season=2024):
        """
        Fetch player data from Sleeper API
        """
        print("🔄 Fetching data from Sleeper API...")
        
        # Get all players
        players_url = "https://api.sleeper.app/v1/players/nfl"
        players_response = requests.get(players_url)
        players_data = players_response.json()
        
        # Filter for skill position players
        skill_positions = ['QB', 'RB', 'WR', 'TE', 'K']
        active_players = []
        
        for player_id, player_info in players_data.items():
            if (player_info.get('position') in skill_positions and 
                player_info.get('active') == True and
                player_info.get('team') is not None):
                active_players.append({
                    'player_id': player_id,
                    'name': player_info.get('full_name', ''),
                    'position': player_info.get('position'),
                    'team': player_info.get('team')
                })
        
        print(f"Found {len(active_players)} active NFL players")
        return active_players
        
    def create_training_dataset(self, sleeper_db_path="sleeper_enhanced.db"):
        """
        Create training dataset from Sleeper database
        """
        print("🏗️ Creating training dataset from Sleeper database...")
        
        import sqlite3
        from ml_features import prepare_sleeper_native_features  # Import from ml_features module
        
        conn = sqlite3.connect(sleeper_db_path)
        
        # Get all player-week combinations for training
        query = """
            SELECT DISTINCT p.full_name, p.position, p.team, w.week, 
                   w.pts_ppr, w.pts_half_ppr, w.pts_std,
                   w.pass_yd, w.pass_td, w.rush_yd, w.rush_td, 
                   w.rec, w.rec_yd, w.rec_td, w.rec_tgt, w.off_snp
            FROM players p
            JOIN weekly_stats w ON p.player_id = w.player_id
            WHERE p.position IN ('QB', 'RB', 'WR', 'TE', 'K')
            AND w.season = 2024 
            AND w.week >= 4  -- Need at least 3 weeks of history
            AND w.week <= 17
            AND w.gp > 0
            AND w.pts_ppr IS NOT NULL
            ORDER BY p.full_name, w.week
        """
        
        results = conn.execute(query).fetchall()
        
        training_data = []
        targets = []
        feature_names = []
        
        print(f"Processing {len(results)} player-week combinations...")
        
        for i, result in enumerate(results):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(results)} records...")
            
            player_name, position, team, week, pts_ppr, pts_half_ppr, pts_std = result[:7]
            
            # Skip if no fantasy points (injured/didn't play)
            if not pts_ppr or pts_ppr <= 0:
                continue
            
            # Create mock opponent (you'd get this from actual schedule data)
            # For now, rotate through common opponents
            all_teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 
                        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 
                        'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 
                        'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
            
            opponent = all_teams[(hash(f"{player_name}{week}") % len(all_teams))]
            if opponent == team:
                opponent = all_teams[(hash(f"{player_name}{week}") % len(all_teams) + 1) % len(all_teams)]
            
            # Create player data for feature engineering
            player_data = {
                'name': player_name,
                'position': position,
                'team': team,
                'opponent': opponent,
                'week': str(week),
                'homeAway': 'HOME' if week % 2 == 0 else 'AWAY'  # Mock home/away
            }
            
            try:
                # Generate features using your Sleeper-native function
                features = prepare_sleeper_native_features(player_data, sleeper_db_path)
                
                if features is not None and len(features.flatten()) > 0:
                    training_data.append(features.flatten())
                    targets.append(pts_ppr)
                    
                    # Store feature names on first iteration
                    if not feature_names:
                        feature_names = self._get_sleeper_feature_names()
                        
            except Exception as e:
                print(f"Error processing {player_name} week {week}: {e}")
                continue
        
        conn.close()
        
        if not training_data:
            raise ValueError("No training data generated! Check your database and feature engineering.")
        
        # Convert to numpy arrays
        X = np.array(training_data)
        y = np.array(targets)
        
        print(f"✅ Created training dataset:")
        print(f"   Shape: {X.shape}")
        print(f"   Target range: {y.min():.1f} - {y.max():.1f}")
        print(f"   Features: {len(feature_names)}")
        
        return X, y, feature_names
    
    def _get_sleeper_feature_names(self):
        """
        Define feature names that match your Sleeper-native feature engineering
        This must match the order in prepare_sleeper_native_features()
        """
        return [
            # Individual weekly stats (last 3 weeks)
            'week1_pts_ppr', 'week1_pts_half_ppr', 'week1_pts_std',
            'week1_pass_yd', 'week1_pass_td', 'week1_pass_att', 'week1_pass_int',
            'week1_rush_yd', 'week1_rush_td', 'week1_rush_att',
            'week1_rec', 'week1_rec_yd', 'week1_rec_td', 'week1_rec_tgt', 'week1_off_snp',
            
            'week2_pts_ppr', 'week2_pts_half_ppr', 'week2_pts_std', 
            'week2_pass_yd', 'week2_pass_td', 'week2_pass_att', 'week2_pass_int',
            'week2_rush_yd', 'week2_rush_td', 'week2_rush_att',
            'week2_rec', 'week2_rec_yd', 'week2_rec_td', 'week2_rec_tgt', 'week2_off_snp',
            
            'week3_pts_ppr', 'week3_pts_half_ppr', 'week3_pts_std',
            'week3_pass_yd', 'week3_pass_td', 'week3_pass_att', 'week3_pass_int', 
            'week3_rush_yd', 'week3_rush_td', 'week3_rush_att',
            'week3_rec', 'week3_rec_yd', 'week3_rec_td', 'week3_rec_tgt', 'week3_off_snp',
            
            # Trend features
            'pts_trend_up', 'pts_volatility', 'target_trend_up', 'snap_consistency',
            'games_with_td', 'games_with_100yd', 'best_week_pts', 'worst_week_pts',
            'pts_ceiling', 'pts_floor',
            
            # Opponent weekly defensive stats
            'opp_week1_sacks', 'opp_week1_qb_hits', 'opp_week1_ints', 'opp_week1_pass_def',
            'opp_week1_pts_allowed', 'opp_week1_yds_allowed',
            'opp_week2_sacks', 'opp_week2_qb_hits', 'opp_week2_ints', 'opp_week2_pass_def', 
            'opp_week2_pts_allowed', 'opp_week2_yds_allowed',
            'opp_def_trending_up', 'opp_def_volatility',
            
            # Fantasy points allowed by position (weekly)
            'opp_week1_fan_pts_allow_qb', 'opp_week1_fan_pts_allow_rb', 'opp_week1_fan_pts_allow_wr',
            'opp_week1_fan_pts_allow_te', 'opp_week1_fan_pts_allow_k',
            'opp_week2_fan_pts_allow_qb', 'opp_week2_fan_pts_allow_rb', 'opp_week2_fan_pts_allow_wr',
            'opp_week2_fan_pts_allow_te', 'opp_week2_fan_pts_allow_k',
            'matchup_pts_allowed_week1',
            
            # Kicker weekly stats
            'k_week1_fgm', 'k_week1_fga', 'k_week1_fgm_20_29', 'k_week1_fgm_30_39',
            'k_week1_fgm_40_49', 'k_week1_fgm_50_59', 'k_week1_xpm', 'k_week1_xpa', 'k_week1_kick_pts',
            'k_week2_fgm', 'k_week2_fga', 'k_week2_kick_pts',
            'k_pts_consistency', 'k_best_week', 'k_worst_week',
            
            # Position encoding
            'is_qb', 'is_rb', 'is_wr', 'is_te', 'is_k',
            
            # Week features
            'week_number', 'week_sin', 'week_cos', 'is_early_season', 'is_mid_season', 'is_late_season',
            
            # Game context
            'is_home', 'is_dome_game', 'team_off_strength'
        ]
    
    def train_models(self, X, y, feature_names):
        """
        Train multiple ML models and select the best one
        """
        print("🤖 Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to test
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        # Train and evaluate models
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Train model
            if name in ['ridge', 'elastic_net']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            
            trained_models[name] = model
            
            print(f"    MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}, R²: {r2:.3f}")
        
        # Select best model (lowest MAE)
        best_model_name = min(model_scores.keys(), key=lambda k: model_scores[k]['mae'])
        best_model = trained_models[best_model_name]
        
        print(f"🏆 Best model: {best_model_name}")
        print(f"   Final MAE: {model_scores[best_model_name]['mae']:.2f}")
        print(f"   Final RMSE: {model_scores[best_model_name]['rmse']:.2f}")
        print(f"   Final R²: {model_scores[best_model_name]['r2']:.3f}")
        
        # Store results
        self.models['best'] = best_model
        self.scalers['best'] = scaler if best_model_name in ['ridge', 'elastic_net'] else None
        self.feature_names['best'] = feature_names
        
        return best_model, scaler, model_scores
    
    def save_models(self, model_dir="models"):
        """
        Save trained models and preprocessing objects
        """
        print("💾 Saving models...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best model
        if 'best' in self.models:
            joblib.dump(self.models['best'], f"{model_dir}/fantasy_football_model.pkl")
            print(f"   Saved model: {model_dir}/fantasy_football_model.pkl")
        
        # Save scaler (if used)
        if 'best' in self.scalers and self.scalers['best'] is not None:
            joblib.dump(self.scalers['best'], f"{model_dir}/scaler.pkl")
            print(f"   Saved scaler: {model_dir}/scaler.pkl")
        
        # Save feature names
        if 'best' in self.feature_names:
            joblib.dump(self.feature_names['best'], f"{model_dir}/feature_names.pkl")
            print(f"   Saved feature names: {model_dir}/feature_names.pkl")
        
        print("✅ Models saved successfully!")
    
    def analyze_feature_importance(self, model, feature_names, top_n=20):
        """
        Analyze and display feature importance
        """
        print(f"📊 Top {top_n} Most Important Features:")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:top_n]):
                print(f"   {i+1:2d}. {feature:30s} {importance:.4f}")
        else:
            print("   Model doesn't support feature importance analysis")
    
    def validate_model(self, model, scaler, X, y, feature_names):
        """
        Comprehensive model validation
        """
        print("🔍 Model Validation:")
        
        # Cross-validation
        if scaler is not None:
            X_scaled = scaler.transform(X)
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        else:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        print(f"   Cross-validation MAE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
        
        # Prediction range analysis
        if scaler is not None:
            y_pred = model.predict(scaler.transform(X))
        else:
            y_pred = model.predict(X)
        
        print(f"   Actual range: {y.min():.1f} - {y.max():.1f}")
        print(f"   Predicted range: {y_pred.min():.1f} - {y_pred.max():.1f}")
        
        # Position-specific validation (if we can infer positions)
        print("   Position-specific accuracy analysis would require position labels")

def main():
    """
    Main training pipeline
    """
    print("🏈 Starting Sleeper Fantasy Football ML Training Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = SleeperMLPipeline()
    
    try:
        # Create training dataset
        X, y, feature_names = pipeline.create_training_dataset("sleeper_enhanced.db")
        
        # Train models
        best_model, scaler, scores = pipeline.train_models(X, y, feature_names)
        
        # Analyze feature importance
        pipeline.analyze_feature_importance(best_model, feature_names)
        
        # Validate model
        pipeline.validate_model(best_model, scaler, X, y, feature_names)
        
        # Save models
        pipeline.save_models()
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"   Model ready for predictions in FantasyAlpha app")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()