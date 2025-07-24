# test_features.py - Quick test after updating app.py

import joblib
import numpy as np

# Test the updated prepare_features function
def test_updated_features():
    try:
        # Load models
        model = joblib.load('fantasy_football_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        print(f"✅ Model expects {len(feature_names)} features")
        
        # Import the updated function from your app
        from app import prepare_features
        
        # Test with sample data
        test_player = {
            'name': 'Josh Allen',
            'position': 'QB',
            'team': 'BUF',
            'opponent': 'MIA',
            'homeAway': 'HOME',
            'week': 8,
            'avgPoints': 22.5,
            'vegasTotal': 48.5
        }
        
        # Generate features
        features = prepare_features(test_player)
        print(f"✅ Generated {features.shape[1]} features")
        
        # Test scaling
        if scaler:
            scaled_features = scaler.transform(features)
            print(f"✅ Features scaled successfully")
        
        # Test prediction
        prediction = model.predict(scaled_features if scaler else features)[0]
        print(f"✅ Prediction successful: {prediction:.1f} points")
        
        print("\n🎉 All tests passed! Your app.py is ready to use.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_features()