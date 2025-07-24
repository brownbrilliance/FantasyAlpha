# create_player_data.py (updated)
import pandas as pd
import numpy as np

def create_player_files():
    """Create the missing player data files"""
    
    try:
        # Try different possible filenames
        possible_files = [
            'fantasy_training_data.csv',
            'fantasy_football_dataset_complete.csv', 
            'nfl_weekly_stats.csv'
        ]
        
        df = None
        used_file = None
        
        for filename in possible_files:
            try:
                df = pd.read_csv(filename)
                used_file = filename
                print(f"✅ Found and loaded {filename}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            print("❌ No training data file found. Available files:")
            import os
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            for f in csv_files:
                print(f"  - {f}")
            return
        
        print(f"✅ Loaded {len(df)} rows from {used_file}")
        
        # Check the column structure
        print(f"📊 Columns: {list(df.columns)}")
        
        # Adapt to your actual column names
        player_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
        team_col = 'recent_team' if 'recent_team' in df.columns else 'team'
        points_col = 'fantasy_points' if 'fantasy_points' in df.columns else 'fantasy_points_ppr'
        
        # Create player averages
        if 'season' in df.columns:
            latest_season = df['season'].max()
            recent_data = df[df['season'] >= latest_season - 1]
        else:
            recent_data = df
        
        player_avgs = recent_data.groupby([player_col, 'position']).agg({
            points_col: 'mean',
            team_col: 'last'
        }).reset_index()
        
        player_avgs.columns = ['player_name', 'position', 'avg_points', 'team']
        player_avgs['avg_points'] = player_avgs['avg_points'].round(1)
        
        # Save player averages
        player_avgs.to_csv('player_averages.csv', index=False)
        print(f"✅ Created player_averages.csv with {len(player_avgs)} players")
        
        # Show sample data
        print("\n📋 Sample players:")
        print(player_avgs.head(10).to_string(index=False))
        
        # Create team ratings
        team_ratings = df.groupby(team_col)[points_col].mean()
        team_ratings = (team_ratings - team_ratings.min()) / (team_ratings.max() - team_ratings.min())
        
        team_df = pd.DataFrame({'team': team_ratings.index, 'rating': team_ratings.values})
        team_df.to_csv('team_ratings.csv', index=False)
        print(f"\n✅ Created team_ratings.csv with {len(team_df)} teams")
        
        # Create defense ratings
        if 'opponent' in df.columns:
            def_ratings = df.groupby('opponent')[points_col].mean()
            def_ratings = 1 - ((def_ratings - def_ratings.min()) / (def_ratings.max() - def_ratings.min()))
            
            def_df = pd.DataFrame({'team': def_ratings.index, 'rating': def_ratings.values})
            def_df.to_csv('defense_ratings.csv', index=False)
            print(f"✅ Created defense_ratings.csv with {len(def_df)} defenses")
        else:
            print("⚠️ No 'opponent' column found, skipping defense ratings")
        
        print("\n🎉 Player data files created successfully!")
        print("\nNow restart your app with: python app.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_player_files()
