from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime
import os
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connections

# Load your trained ML model
try:
    model = joblib.load('fantasy_football_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✅ ML Model loaded successfully!")
    print(f"✅ Model expects {len(feature_names)} features")
except Exception as e:
    print(f"⚠️ Model loading error: {str(e)}")
    print("⚠️ Model files not found. Please ensure you have:")
    print("   - fantasy_football_model.pkl")
    print("   - scaler.pkl")
    print("   - feature_names.pkl")
    model = None
    scaler = None
    feature_names = None

# Team mappings and data
NFL_TEAMS = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
}

def get_weather_data(city="New York"):
    """
    Get weather data for game location
    Replace with actual weather API (OpenWeatherMap, etc.)
    """
    try:
        # For now, return simulated weather data
        weather_data = {
            "temperature": np.random.randint(35, 75),
            "wind_speed": np.random.randint(3, 20),
            "conditions": np.random.choice(["Clear", "Partly Cloudy", "Overcast", "Light Rain"]),
            "fantasy_impact": "Minimal"
        }
        
        # Adjust fantasy impact based on conditions
        if weather_data["wind_speed"] > 15 or weather_data["conditions"] == "Light Rain":
            weather_data["fantasy_impact"] = "Moderate Negative"
        elif weather_data["wind_speed"] > 10:
            weather_data["fantasy_impact"] = "Slight Negative"
            
        return weather_data
    except:
        return {
            "temperature": 65,
            "wind_speed": 8,
            "conditions": "Clear",
            "fantasy_impact": "Minimal"
        }

import numpy as np
import math

def prepare_features(player_data):
    """
    Convert player data into features for ML model
    This matches the exact 29 features your model was trained with
    """
    features = {}
    
    # Get basic values
    avg_points = float(player_data.get('avgPoints', 0))
    week = int(player_data.get('week', 1))
    vegas_total = float(player_data.get('vegasTotal', 47))
    target_share = float(player_data.get('targetShare', 0.15))
    snap_count_pct = float(player_data.get('snapCountPct', 0.75))
    temp = float(player_data.get('weather_temp', 65))
    wind_speed = float(player_data.get('weather_wind', 8))
    position = player_data.get('position', 'WR')
    team = player_data.get('team', '')
    opponent = player_data.get('opponent', '')
    
    # 1. avg_points_last_3 (your model uses this instead of just avgPoints)
    features['avg_points_last_3'] = avg_points
    
    # 2. target_share
    features['target_share'] = target_share
    
    # 3. snap_count_pct  
    features['snap_count_pct'] = snap_count_pct
    
    # 4. vegas_total
    features['vegas_total'] = vegas_total
    
    # 5. temp
    features['temp'] = temp
    
    # 6. wind_speed
    features['wind_speed'] = wind_speed
    
    # 7. is_dome (dome stadiums)
    dome_teams = ['ATL', 'NO', 'DET', 'MIN', 'ARI', 'LV', 'LAR', 'IND']
    features['is_dome'] = 1 if team in dome_teams or opponent in dome_teams else 0
    
    # 8. week
    features['week'] = week
    
    # 9. week_sin (cyclical encoding for week)
    features['week_sin'] = math.sin(2 * math.pi * week / 18)
    
    # 10. week_cos (cyclical encoding for week) 
    features['week_cos'] = math.cos(2 * math.pi * week / 18)
    
    # 11. early_season (weeks 1-6)
    features['early_season'] = 1 if week <= 6 else 0
    
    # 12. mid_season (weeks 7-13)
    features['mid_season'] = 1 if 7 <= week <= 13 else 0
    
    # 13. late_season (weeks 14-18)
    features['late_season'] = 1 if week >= 14 else 0
    
    # 14. playoff_push (weeks 15-18)
    features['playoff_push'] = 1 if week >= 15 else 0
    
    # 15. is_home
    features['is_home'] = 1 if player_data.get('homeAway') == 'HOME' else 0
    
    # 16-19. Position encoding
    features['pos_QB'] = 1 if position == 'QB' else 0
    features['pos_RB'] = 1 if position == 'RB' else 0
    features['pos_WR'] = 1 if position == 'WR' else 0
    features['pos_TE'] = 1 if position == 'TE' else 0
    
    # 20. team_strength
    team_ratings = {
        'BUF': 0.90, 'KC': 0.88, 'DAL': 0.85, 'SF': 0.83, 'MIA': 0.82,
        'GB': 0.78, 'PHI': 0.76, 'BAL': 0.75, 'CIN': 0.73, 'LAC': 0.72,
        'HOU': 0.65, 'MIN': 0.63, 'TB': 0.62, 'LAR': 0.60, 'SEA': 0.58,
        'DET': 0.57, 'ATL': 0.55, 'NO': 0.53, 'JAX': 0.52, 'IND': 0.50,
        'LV': 0.48, 'NYJ': 0.45, 'TEN': 0.43, 'CLE': 0.42, 'PIT': 0.40,
        'WAS': 0.38, 'CAR': 0.37, 'CHI': 0.35, 'NE': 0.33, 'NYG': 0.30,
        'DEN': 0.28, 'ARI': 0.25
    }
    features['team_strength'] = team_ratings.get(team, 0.5)
    
    # 21. opp_def_rating
    def_ratings = {
        'MIA': 0.25, 'WAS': 0.28, 'DET': 0.30, 'LAC': 0.32, 'ATL': 0.35,
        'GB': 0.37, 'MIN': 0.40, 'TB': 0.42, 'SEA': 0.45, 'NO': 0.47,
        'KC': 0.50, 'LAR': 0.52, 'TEN': 0.53, 'IND': 0.55, 'JAX': 0.57,
        'HOU': 0.58, 'CIN': 0.60, 'NYJ': 0.62, 'CAR': 0.63, 'LV': 0.65,
        'PHI': 0.67, 'DAL': 0.70, 'BAL': 0.72, 'SF': 0.75, 'PIT': 0.77,
        'CLE': 0.78, 'BUF': 0.80, 'DEN': 0.82, 'CHI': 0.85, 'NE': 0.88,
        'NYG': 0.90, 'ARI': 0.92
    }
    features['opp_def_rating'] = def_ratings.get(opponent, 0.5)
    
    # 22. def_coverage (opponent's defensive coverage style)
    # Zone-heavy defenses vs Man coverage - affects WR/TE differently
    zone_heavy_defenses = ['SF', 'BAL', 'PIT', 'GB', 'TB']
    features['def_coverage'] = 0.7 if opponent in zone_heavy_defenses else 0.3
    
    # 23. hist_avg_performance (historical average for position)
    pos_averages = {'QB': 18.5, 'RB': 12.8, 'WR': 10.2, 'TE': 8.1}
    features['hist_avg_performance'] = pos_averages.get(position, 10.0)
    
    # 24. high_target_share (binary: high target share player)
    features['high_target_share'] = 1 if target_share > 0.20 else 0
    
    # 25. high_snap_pct (binary: high snap count player)
    features['high_snap_pct'] = 1 if snap_count_pct > 0.80 else 0
    
    # 26. usage_score (composite usage metric)
    features['usage_score'] = (target_share * 0.6) + (snap_count_pct * 0.4)
    
    # 27. bad_weather (binary: poor weather conditions)
    features['bad_weather'] = 1 if (temp < 40 or temp > 85 or wind_speed > 15) else 0
    
    # 28. game_environment (game flow expectation)
    # Higher vegas total = more offensive environment
    features['game_environment'] = min(1.0, vegas_total / 60.0)
    
    # 29. matchup_difficulty (composite matchup score)
    # Combines opponent defense rating and specific positional matchup
    base_difficulty = features['opp_def_rating']
    
    # Position-specific adjustments
    if position == 'QB':
        # QBs face less variance in matchup difficulty
        matchup_adj = 0.1
    elif position == 'RB':
        # RBs more affected by run defense
        run_def_multiplier = 1.2 if opponent in ['SF', 'BAL', 'BUF', 'PIT'] else 0.8
        matchup_adj = 0.2 * run_def_multiplier
    elif position in ['WR', 'TE']:
        # WRs/TEs more affected by pass defense and coverage
        pass_def_multiplier = 1.3 if opponent in ['SF', 'BUF', 'DAL', 'NE'] else 0.7
        matchup_adj = 0.25 * pass_def_multiplier
    else:
        matchup_adj = 0.15
    
    features['matchup_difficulty'] = min(1.0, base_difficulty + matchup_adj)
    
    # Create the feature array in the exact order expected by the model
    expected_features = [
        'avg_points_last_3', 'target_share', 'snap_count_pct', 'vegas_total', 'temp', 
        'wind_speed', 'is_dome', 'week', 'week_sin', 'week_cos', 'early_season', 
        'mid_season', 'late_season', 'playoff_push', 'is_home', 'pos_QB', 'pos_RB', 
        'pos_WR', 'pos_TE', 'team_strength', 'opp_def_rating', 'def_coverage', 
        'hist_avg_performance', 'high_target_share', 'high_snap_pct', 'usage_score', 
        'bad_weather', 'game_environment', 'matchup_difficulty'
    ]
    
    # Build feature array in correct order
    feature_array = []
    for feature_name in expected_features:
        feature_array.append(features[feature_name])
    
    print(f"Generated {len(feature_array)} features for ML model")
    return np.array(feature_array).reshape(1, -1)

# Test the corrected function
def test_corrected_features():
    """Test the corrected feature generation"""
    import joblib
    
    test_player = {
        'name': 'Josh Allen',
        'position': 'QB',
        'team': 'BUF',
        'opponent': 'MIA',
        'homeAway': 'HOME',
        'week': 8,
        'avgPoints': 22.5,
        'vegasTotal': 48.5,
        'targetShare': 0.25,
        'snapCountPct': 0.95
    }
    
    try:
        # Test feature generation
        features = prepare_features(test_player)
        print(f"✅ Generated {features.shape[1]} features")
        
        # Load model and test prediction
        model = joblib.load('fantasy_football_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        if scaler:
            scaled_features = scaler.transform(features)
            print("✅ Scaling successful!")
        else:
            scaled_features = features
        
        prediction = model.predict(scaled_features)[0]
        print(f"✅ Prediction successful: {prediction:.1f} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_corrected_features()

def generate_analysis(player_data, prediction):
    """
    Generate AI analysis text based on prediction factors
    """
    analysis = []
    position = player_data.get('position')
    avg_points = float(player_data.get('avgPoints', 0))
    vegas_total = float(player_data.get('vegasTotal', 47))
    
    # Position-specific analysis
    pos_avg = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8}
    if avg_points > pos_avg.get(position, 10):
        analysis.append(f"Strong recent form with {avg_points:.1f} avg points")
    else:
        analysis.append(f"Below average recent performance at {avg_points:.1f} points")
    
    # Game script analysis
    if vegas_total > 50:
        analysis.append("High-scoring game environment favors offensive production")
    elif vegas_total < 45:
        analysis.append("Low-scoring game may limit upside potential")
    else:
        analysis.append("Moderate scoring expected based on Vegas projections")
    
    # Home/Away factor
    if player_data.get('homeAway') == 'HOME':
        analysis.append("Home field advantage provides 1-2 point boost")
    else:
        analysis.append("Road game presents additional challenges")
    
    # Prediction confidence
    if prediction > 20:
        analysis.append("High-confidence projection based on multiple positive factors")
    elif prediction > 15:
        analysis.append("Solid projection with good floor and ceiling")
    elif prediction > 10:
        analysis.append("Moderate projection with some risk factors")
    else:
        analysis.append("Low projection due to challenging matchup factors")
    
    return analysis

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_player():
    """
    Main prediction endpoint for individual players
    """
    try:
        data = request.get_json()
        print(f"Received prediction request: {data}")
        
        if not model:
            # Fallback simulation if model not loaded
            from random import uniform
            base_points = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8}
            position = data.get('position', 'WR')
            prediction = base_points[position] + uniform(-5, 8)
            print("Using fallback prediction (model not loaded)")
        else:
            # Use actual ML model
            print("Using trained ML model for prediction")
            features = prepare_features(data)
            print(f"Feature shape: {features.shape}")
            
            # Scale features if scaler was used during training
            if scaler:
                features = scaler.transform(features)
                print("Features scaled successfully")
            
            prediction = model.predict(features)[0]
            print(f"Raw ML prediction: {prediction}")
        
        # Ensure prediction is reasonable
        prediction = max(0, min(50, prediction))
        print(f"Final prediction: {prediction}")
        
        # Generate analysis
        analysis = generate_analysis(data, prediction)
        
        # Get weather data
        weather = get_weather_data()
        
        return jsonify({
            'prediction': round(prediction, 1),
            'analysis': analysis,
            'weather': weather,
            'confidence': min(95, max(60, 85 - abs(prediction - 15) * 2))
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/rankings', methods=['POST'])
def generate_rankings():
    """
    Generate position rankings for a given week
    """
    try:
        data = request.get_json()
        position = data.get('position', 'QB')
        week = data.get('week', 8)
        
        # Sample players by position (you'd load this from your database)
        sample_players = {
            'QB': [
                {'name': 'Josh Allen', 'team': 'BUF', 'opponent': 'MIA'},
                {'name': 'Lamar Jackson', 'team': 'BAL', 'opponent': 'CLE'},
                {'name': 'Patrick Mahomes', 'team': 'KC', 'opponent': 'LV'},
                {'name': 'Dak Prescott', 'team': 'DAL', 'opponent': 'SF'},
                {'name': 'Tua Tagovailoa', 'team': 'MIA', 'opponent': 'BUF'}
            ],
            'RB': [
                {'name': 'Christian McCaffrey', 'team': 'SF', 'opponent': 'DAL'},
                {'name': 'Derrick Henry', 'team': 'BAL', 'opponent': 'CLE'},
                {'name': 'Saquon Barkley', 'team': 'PHI', 'opponent': 'CIN'},
                {'name': 'Josh Jacobs', 'team': 'GB', 'opponent': 'ARI'},
                {'name': 'Alvin Kamara', 'team': 'NO', 'opponent': 'LAC'}
            ],
            'WR': [
                {'name': 'Tyreek Hill', 'team': 'MIA', 'opponent': 'BUF'},
                {'name': 'Davante Adams', 'team': 'LV', 'opponent': 'KC'},
                {'name': 'Stefon Diggs', 'team': 'HOU', 'opponent': 'IND'},
                {'name': 'Cooper Kupp', 'team': 'LAR', 'opponent': 'SEA'},
                {'name': 'A.J. Brown', 'team': 'PHI', 'opponent': 'CIN'}
            ],
            'TE': [
                {'name': 'Travis Kelce', 'team': 'KC', 'opponent': 'LV'},
                {'name': 'Mark Andrews', 'team': 'BAL', 'opponent': 'CLE'},
                {'name': 'George Kittle', 'team': 'SF', 'opponent': 'DAL'},
                {'name': 'T.J. Hockenson', 'team': 'MIN', 'opponent': 'LAR'},
                {'name': 'Evan Engram', 'team': 'JAX', 'opponent': 'GB'}
            ]
        }
        
        players = sample_players.get(position, [])
        rankings = []
        
        for player in players:
            # Create mock player data for prediction
            player_data = {
                'position': position,
                'team': player['team'],
                'opponent': player['opponent'],
                'homeAway': 'HOME',
                'week': week,
                'avgPoints': np.random.uniform(8, 25),
                'vegasTotal': np.random.uniform(42, 55)
            }
            
            # Get prediction
            if model:
                try:
                    features = prepare_features(player_data)
                    if scaler:
                        features = scaler.transform(features)
                    prediction = model.predict(features)[0]
                except Exception as e:
                    print(f"Error predicting for {player['name']}: {str(e)}")
                    base_points = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8}
                    prediction = base_points[position] + np.random.uniform(-3, 6)
            else:
                base_points = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8}
                prediction = base_points[position] + np.random.uniform(-3, 6)
            
            rankings.append({
                'name': player['name'],
                'team': player['team'],
                'opponent': player['opponent'],
                'points': round(prediction, 1)
            })
        
        # Sort by predicted points
        rankings.sort(key=lambda x: x['points'], reverse=True)
        
        return jsonify({'rankings': rankings})
        
    except Exception as e:
        print(f"Error generating rankings: {str(e)}")
        return jsonify({'error': 'Rankings generation failed'}), 500

@app.route('/api/compare', methods=['POST'])
def compare_players():
    """
    Compare two players head-to-head
    """
    try:
        data = request.get_json()
        player1 = data.get('player1')
        player2 = data.get('player2')
        
        results = {}
        
        for i, player in enumerate([player1, player2], 1):
            # Create mock data for prediction
            player_data = {
                'position': player['position'],
                'team': player['team'],
                'opponent': 'MIA',  # You'd get actual opponent data
                'homeAway': 'HOME',
                'week': 8,
                'avgPoints': np.random.uniform(10, 20),
                'vegasTotal': np.random.uniform(45, 52)
            }
            
            # Get prediction
            if model:
                try:
                    features = prepare_features(player_data)
                    if scaler:
                        features = scaler.transform(features)
                    prediction = model.predict(features)[0]
                except Exception as e:
                    print(f"Error predicting for {player['name']}: {str(e)}")
                    base_points = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8}
                    prediction = base_points[player['position']] + np.random.uniform(-2, 5)
            else:
                base_points = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8}
                prediction = base_points[player['position']] + np.random.uniform(-2, 5)
            
            results[f'player{i}'] = {
                'name': player['name'],
                'prediction': round(prediction, 1),
                'analysis': generate_analysis(player_data, prediction)
            }
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error comparing players: {str(e)}")
        return jsonify({'error': 'Player comparison failed'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Fantasy Football AI API...")
    print("📊 Endpoints available:")
    print("   POST /api/predict - Player predictions")
    print("   POST /api/rankings - Position rankings")
    print("   POST /api/compare - Player comparisons")
    print("   GET /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5001)