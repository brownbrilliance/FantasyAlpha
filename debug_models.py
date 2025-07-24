# debug_models.py
import joblib
import os

print("🔍 Checking model files...")

# Check what model files exist
model_files = [
    'fantasy_football_model.pkl',
    'scaler.pkl', 
    'feature_names.pkl',
    'position_models.pkl',
    'position_scalers.pkl'
]

for file in model_files:
    if os.path.exists(file):
        try:
            data = joblib.load(file)
            print(f"✅ {file}: Loaded successfully")
            if file == 'position_models.pkl':
                print(f"   Positions: {list(data.keys())}")
            elif file == 'feature_names.pkl':
                print(f"   Features: {len(data)} features")
        except Exception as e:
            print(f"❌ {file}: Error loading - {e}")
    else:
        print(f"❌ {file}: Not found")

# Test model prediction
try:
    model = joblib.load('fantasy_football_model.pkl')
    import numpy as np
    
    # Test with dummy features
    test_features = np.random.random((1, 20))  # Adjust size based on your features
    prediction = model.predict(test_features)[0]
    print(f"✅ Model prediction test: {prediction:.2f}")
except Exception as e:
    print(f"❌ Model prediction test failed: {e}")