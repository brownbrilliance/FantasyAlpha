#!/usr/bin/env python3
"""
Setup Script for Ultimate Fantasy Football App
Handles the migration from paste.txt to proper module structure
"""

import os
import shutil
import sys
from pathlib import Path

def setup_ultimate_fantasy():
    """Setup the ultimate fantasy football app"""
    
    print("🏈 Setting Up Ultimate Fantasy Football App")
    print("=" * 50)
    
    # 1. Create required directories
    print("📁 Creating required directories...")
    directories = ['logs', 'models', 'templates', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    # 2. Handle paste.txt conversion
    print("\n🔄 Handling module imports...")
    
    if os.path.exists('paste.txt'):
        print("   📄 Found paste.txt")
        print("   ℹ️  The ml_features.py module has been created to replace paste.txt functionality")
        print("   ℹ️  You can keep paste.txt as backup or remove it")
        
        # Ask user if they want to rename paste.txt as backup
        try:
            response = input("   📝 Rename paste.txt to paste_backup.txt? (y/n): ").lower()
            if response in ['y', 'yes']:
                shutil.move('paste.txt', 'paste_backup.txt')
                print("   ✅ Renamed paste.txt → paste_backup.txt")
        except KeyboardInterrupt:
            print("\n   ⏹️  Skipping paste.txt rename")
    else:
        print("   ⚠️  paste.txt not found - ml_features.py will provide all functionality")
    
    # 3. Check for database
    print("\n🗄️  Checking database...")
    if os.path.exists('sleeper_enhanced.db'):
        print("   ✅ Found sleeper_enhanced.db")
        
        # Quick database check
        try:
            import sqlite3
            conn = sqlite3.connect('sleeper_enhanced.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM players WHERE position IN ('QB', 'RB', 'WR', 'TE', 'K')")
            player_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM weekly_stats WHERE season = 2024")
            stats_count = cursor.fetchone()[0]
            
            print(f"   📊 {player_count:,} players, {stats_count:,} weekly stats")
            conn.close()
            
        except Exception as e:
            print(f"   ⚠️  Database check failed: {e}")
    else:
        print("   ❌ sleeper_enhanced.db not found")
        print("   💡 You'll need to create your Sleeper database first")
    
    # 4. Check for existing models
    print("\n🤖 Checking ML models...")
    model_files = [
        'models/fantasy_football_model.pkl',
        'models/scaler.pkl', 
        'models/feature_names.pkl'
    ]
    
    model_exists = False
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"   ✅ Found: {model_file}")
            model_exists = True
        else:
            print(f"   ❌ Missing: {model_file}")
    
    if not model_exists:
        print("   💡 No models found - you'll need to train them first")
        print("   🏃 Run: python sleeper_ml_training.py")
    
    # 5. Check Python dependencies
    print("\n📦 Checking Python dependencies...")
    required_packages = [
        'flask', 'flask-cors', 'numpy', 'pandas', 'scikit-learn', 
        'joblib', 'sqlite3'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   📥 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # 6. Verify file structure
    print("\n📋 Verifying file structure...")
    required_files = [
        'config.py',
        'ml_features.py',
        'sleeper_ml_training.py',
        'app.py'  # Will be the ultimate version
    ]
    
    all_files_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            all_files_present = False
    
    # 7. Show next steps
    print("\n🚀 Next Steps:")
    print("   1. Ensure your Sleeper database exists (sleeper_enhanced.db)")
    print("   2. Train ML models: python sleeper_ml_training.py")
    print("   3. Start the app: python app.py")
    print("   4. Visit: http://localhost:5000")
    
    print("\n🎯 Training Pipeline:")
    if os.path.exists('sleeper_enhanced.db') and all_files_present:
        print("   ✅ Ready to train ML models!")
        print("   🏃 Run: python sleeper_ml_training.py")
    else:
        print("   ⚠️  Need database and all files before training")
    
    print("\n🌐 Web App:")
    if model_exists and all_files_present:
        print("   ✅ Ready to start web app!")
        print("   🏃 Run: python app.py")
    else:
        print("   ⚠️  Need trained models before starting app")
    
    print("\n" + "=" * 50)
    print("🏁 Setup Complete!")
    
    # 8. Optional: Quick test
    try:
        test_response = input("\n🧪 Test ml_features import? (y/n): ").lower()
        if test_response in ['y', 'yes']:
            print("   🔄 Testing ml_features import...")
            try:
                from ml_features import prepare_sleeper_native_features
                print("   ✅ ml_features import successful!")
                
                # Test with dummy data
                dummy_data = {
                    'name': 'Test Player',
                    'position': 'QB',
                    'team': 'BUF',
                    'opponent': 'MIA',
                    'week': '9',
                    'homeAway': 'HOME'
                }
                
                features = prepare_sleeper_native_features(dummy_data)
                print(f"   ✅ Feature generation test: {features.shape} features")
                
            except Exception as e:
                print(f"   ❌ Import test failed: {e}")
                
    except KeyboardInterrupt:
        print("\n   ⏹️  Skipping test")
    
    print("\n🎉 Ultimate Fantasy Football App is ready!")

if __name__ == "__main__":
    setup_ultimate_fantasy()