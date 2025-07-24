# create_full_schedule.py
import json

def create_complete_2025_schedule():
    """Complete 2025 NFL schedule for all 18 weeks"""
    return {
        "1": {
            'DAL': 'PHI', 'PHI': 'DAL',  # Thursday opener
            'KC': 'LAC', 'LAC': 'KC',    # Friday in Brazil
            'LV': 'NE', 'NE': 'LV',
            'PIT': 'NYJ', 'NYJ': 'PIT',
            'MIA': 'IND', 'IND': 'MIA',
            'ARI': 'NO', 'NO': 'ARI',
            'NYG': 'WAS', 'WAS': 'NYG',
            'CAR': 'JAX', 'JAX': 'CAR',
            'CIN': 'CLE', 'CLE': 'CIN',
            'TB': 'ATL', 'ATL': 'TB',
            'TEN': 'DEN', 'DEN': 'TEN',
            'SF': 'SEA', 'SEA': 'SF',
            'DET': 'GB', 'GB': 'DET',
            'HOU': 'LAR', 'LAR': 'HOU',
            'BAL': 'BUF', 'BUF': 'BAL',
            'MIN': 'CHI', 'CHI': 'MIN'
        },
        "2": {
            'WAS': 'GB', 'GB': 'WAS',
            'JAX': 'CIN', 'CIN': 'JAX',
            'BUF': 'NYJ', 'NYJ': 'BUF',
            'NE': 'MIA', 'MIA': 'NE',
            'LAR': 'TEN', 'TEN': 'LAR',
            'CLE': 'BAL', 'BAL': 'CLE',
            'SF': 'NO', 'NO': 'SF',
            'NYG': 'DAL', 'DAL': 'NYG',
            'SEA': 'PIT', 'PIT': 'SEA',
            'CHI': 'DET', 'DET': 'CHI',
            'DEN': 'IND', 'IND': 'DEN',
            'CAR': 'ARI', 'ARI': 'CAR',
            'PHI': 'KC', 'KC': 'PHI',
            'ATL': 'MIN', 'MIN': 'ATL',
            'TB': 'HOU', 'HOU': 'TB',
            'LAC': 'LV', 'LV': 'LAC'
        },
        "3": {
            'MIA': 'BUF', 'BUF': 'MIA',
            'PIT': 'NE', 'NE': 'PIT',
            'HOU': 'JAX', 'JAX': 'HOU',
            'IND': 'TEN', 'TEN': 'IND',
            'CIN': 'MIN', 'MIN': 'CIN',
            'NYJ': 'TB', 'TB': 'NYJ',
            'GB': 'CLE', 'CLE': 'GB',
            'LV': 'WAS', 'WAS': 'LV',
            'ATL': 'CAR', 'CAR': 'ATL',
            'LAR': 'PHI', 'PHI': 'LAR',
            'NO': 'SEA', 'SEA': 'NO',
            'DEN': 'LAC', 'LAC': 'DEN',
            'DAL': 'CHI', 'CHI': 'DAL',
            'ARI': 'SF', 'SF': 'ARI',
            'KC': 'NYG', 'NYG': 'KC',
            'DET': 'BAL', 'BAL': 'DET'
        },
        "4": {
            'SEA': 'ARI', 'ARI': 'SEA',
            'MIN': 'PIT', 'PIT': 'MIN',  # Dublin
            'NO': 'BUF', 'BUF': 'NO',
            'WAS': 'ATL', 'ATL': 'WAS',
            'LAC': 'NYG', 'NYG': 'LAC',
            'TEN': 'HOU', 'HOU': 'TEN',
            'CLE': 'DET', 'DET': 'CLE',
            'CAR': 'NE', 'NE': 'CAR',
            'PHI': 'TB', 'TB': 'PHI',
            'JAX': 'SF', 'SF': 'JAX',
            'IND': 'LAR', 'LAR': 'IND',
            'BAL': 'KC', 'KC': 'BAL',
            'CHI': 'LV', 'LV': 'CHI',
            'GB': 'DAL', 'DAL': 'GB',
            'NYJ': 'MIA', 'MIA': 'NYJ',
            'CIN': 'DEN', 'DEN': 'CIN'
        },
        "5": {
            'SF': 'LAR', 'LAR': 'SF',
            'MIN': 'CLE', 'CLE': 'MIN',  # London
            'NYG': 'NO', 'NO': 'NYG',
            'DEN': 'PHI', 'PHI': 'DEN',
            'HOU': 'BAL', 'BAL': 'HOU',
            'DAL': 'NYJ', 'NYJ': 'DAL',
            'LV': 'IND', 'IND': 'LV',
            'MIA': 'CAR', 'CAR': 'MIA',
            'TEN': 'ARI', 'ARI': 'TEN',
            'TB': 'SEA', 'SEA': 'TB',
            'WAS': 'LAC', 'LAC': 'WAS',
            'DET': 'CIN', 'CIN': 'DET',
            'NE': 'BUF', 'BUF': 'NE',
            'KC': 'JAX', 'JAX': 'KC',
            # Bye teams
            'ATL': 'BYE', 'CHI': 'BYE', 'GB': 'BYE', 'PIT': 'BYE'
        },
        "6": {
            'PHI': 'NYG', 'NYG': 'PHI',
            'DEN': 'NYJ', 'NYJ': 'DEN',  # London
            'CLE': 'PIT', 'PIT': 'CLE',
            'LAC': 'MIA', 'MIA': 'LAC',
            'SF': 'TB', 'TB': 'SF',
            'SEA': 'JAX', 'JAX': 'SEA',
            'DAL': 'CAR', 'CAR': 'DAL',
            'LAR': 'BAL', 'BAL': 'LAR',
            'ARI': 'IND', 'IND': 'ARI',
            'TEN': 'LV', 'LV': 'TEN',
            'CIN': 'GB', 'GB': 'CIN',
            'NE': 'NO', 'NO': 'NE',
            'DET': 'KC', 'KC': 'DET',
            'CHI': 'WAS', 'WAS': 'CHI',
            'BUF': 'ATL', 'ATL': 'BUF',
            # Bye teams
            'HOU': 'BYE', 'MIN': 'BYE'
        },
        "7": {
            'PIT': 'CIN', 'CIN': 'PIT',
            'LAR': 'JAX', 'JAX': 'LAR',  # London
            'NE': 'TEN', 'TEN': 'NE',
            'MIA': 'CLE', 'CLE': 'MIA',
            'LV': 'KC', 'KC': 'LV',
            'CAR': 'NYJ', 'NYJ': 'CAR',
            'NO': 'CHI', 'CHI': 'NO',
            'PHI': 'MIN', 'MIN': 'PHI',
            'NYG': 'DEN', 'DEN': 'NYG',
            'IND': 'LAC', 'LAC': 'IND',
            'WAS': 'DAL', 'DAL': 'WAS',
            'GB': 'ARI', 'ARI': 'GB',
            'ATL': 'SF', 'SF': 'ATL',
            'TB': 'DET', 'DET': 'TB',
            'HOU': 'SEA', 'SEA': 'HOU',
            # Bye teams
            'BAL': 'BYE', 'BUF': 'BYE'
        },
        "8": {
            'MIN': 'LAC', 'LAC': 'MIN',  # Thursday Night
            'NYJ': 'CIN', 'CIN': 'NYJ',
            'CHI': 'BAL', 'BAL': 'CHI',
            'MIA': 'ATL', 'ATL': 'MIA',
            'CLE': 'NE', 'NE': 'CLE',
            'NYG': 'PHI', 'PHI': 'NYG',
            'BUF': 'CAR', 'CAR': 'BUF',
            'SF': 'HOU', 'HOU': 'SF',
            'TB': 'NO', 'NO': 'TB',
            'DAL': 'DEN', 'DEN': 'DAL',
            'TEN': 'IND', 'IND': 'TEN',
            'GB': 'PIT', 'PIT': 'GB',    # Sunday Night
            'WAS': 'KC', 'KC': 'WAS',    # Monday Night
            # Bye teams
            'ARI': 'BYE', 'DET': 'BYE', 'JAX': 'BYE', 'LAR': 'BYE', 'LV': 'BYE', 'SEA': 'BYE'
        },
        "9": {
            'BAL': 'MIA', 'MIA': 'BAL',
            'IND': 'PIT', 'PIT': 'IND',
            'ATL': 'NE', 'NE': 'ATL',
            'CHI': 'CIN', 'CIN': 'CHI',
            'LAC': 'TEN', 'TEN': 'LAC',
            'SF': 'NYG', 'NYG': 'SF',
            'CAR': 'GB', 'GB': 'CAR',
            'DEN': 'HOU', 'HOU': 'DEN',
            'MIN': 'DET', 'DET': 'MIN',
            'JAX': 'LV', 'LV': 'JAX',
            'NO': 'LAR', 'LAR': 'NO',
            'KC': 'BUF', 'BUF': 'KC',    # Big game!
            'SEA': 'WAS', 'WAS': 'SEA',
            'ARI': 'DAL', 'DAL': 'ARI',
            # Bye teams
            'CLE': 'BYE', 'NYJ': 'BYE', 'PHI': 'BYE', 'TB': 'BYE'
        },
        "10": {
            'LV': 'DEN', 'DEN': 'LV',
            'ATL': 'IND', 'IND': 'ATL',  # Berlin
            'JAX': 'HOU', 'HOU': 'JAX',
            'BUF': 'MIA', 'MIA': 'BUF',
            'NE': 'TB', 'TB': 'NE',
            'CLE': 'NYJ', 'NYJ': 'CLE',
            'NYG': 'CHI', 'CHI': 'NYG',
            'NO': 'CAR', 'CAR': 'NO',
            'BAL': 'MIN', 'MIN': 'BAL',
            'ARI': 'SEA', 'SEA': 'ARI',
            'LAR': 'SF', 'SF': 'LAR',
            'DET': 'WAS', 'WAS': 'DET',
            'PIT': 'LAC', 'LAC': 'PIT',
            'PHI': 'GB', 'GB': 'PHI',
            # Bye teams
            'CIN': 'BYE', 'DAL': 'BYE', 'KC': 'BYE', 'TEN': 'BYE'
        },
        "11": {
            'NYJ': 'NE', 'NE': 'NYJ',
            'WAS': 'MIA', 'MIA': 'WAS',  # Madrid
            'TB': 'BUF', 'BUF': 'TB',
            'LAC': 'JAX', 'JAX': 'LAC',
            'CIN': 'PIT', 'PIT': 'CIN',
            'CAR': 'ATL', 'ATL': 'CAR',
            'GB': 'NYG', 'NYG': 'GB',
            'CHI': 'MIN', 'MIN': 'CHI',
            'HOU': 'TEN', 'TEN': 'HOU',
            'SF': 'ARI', 'ARI': 'SF',
            'SEA': 'LAR', 'LAR': 'SEA',
            'KC': 'DEN', 'DEN': 'KC',
            'BAL': 'CLE', 'CLE': 'BAL',
            'DET': 'PHI', 'PHI': 'DET',
            'DAL': 'LV', 'LV': 'DAL',
            # Bye teams
            'IND': 'BYE', 'NO': 'BYE'
        },
        "12": {
            'BUF': 'HOU', 'HOU': 'BUF',
            'NE': 'CIN', 'CIN': 'NE',
            'PIT': 'CHI', 'CHI': 'PIT',
            'IND': 'KC', 'KC': 'IND',
            'NYJ': 'BAL', 'BAL': 'NYJ',
            'NYG': 'DET', 'DET': 'NYG',
            'SEA': 'TEN', 'TEN': 'SEA',
            'MIN': 'GB', 'GB': 'MIN',
            'CLE': 'LV', 'LV': 'CLE',
            'JAX': 'ARI', 'ARI': 'JAX',
            'ATL': 'NO', 'NO': 'ATL',
            'PHI': 'DAL', 'DAL': 'PHI',
            'TB': 'LAR', 'LAR': 'TB',
            'CAR': 'SF', 'SF': 'CAR',
            # Bye teams
            'DEN': 'BYE', 'LAC': 'BYE', 'MIA': 'BYE', 'WAS': 'BYE'
        },
        "13": {
            # Thanksgiving Thursday
            'GB': 'DET', 'DET': 'GB',
            'KC': 'DAL', 'DAL': 'KC',
            'CIN': 'BAL', 'BAL': 'CIN',
            # Friday
            'CHI': 'PHI', 'PHI': 'CHI',
            # Sunday
            'SF': 'CLE', 'CLE': 'SF',
            'JAX': 'TEN', 'TEN': 'JAX',
            'HOU': 'IND', 'IND': 'HOU',
            'ARI': 'TB', 'TB': 'ARI',
            'NO': 'MIA', 'MIA': 'NO',
            'ATL': 'NYJ', 'NYJ': 'ATL',
            'LAR': 'CAR', 'CAR': 'LAR',
            'MIN': 'SEA', 'SEA': 'MIN',
            'BUF': 'PIT', 'PIT': 'BUF',
            'LV': 'LAC', 'LAC': 'LV',
            'DEN': 'WAS', 'WAS': 'DEN',
            'NYG': 'NE', 'NE': 'NYG'
        },
        "14": {
            'DAL': 'DET', 'DET': 'DAL',
            'IND': 'JAX', 'JAX': 'IND',
            'NO': 'TB', 'TB': 'NO',
            'MIA': 'NYJ', 'NYJ': 'MIA',
            'PIT': 'BAL', 'BAL': 'PIT',
            'SEA': 'ATL', 'ATL': 'SEA',
            'TEN': 'CLE', 'CLE': 'TEN',
            'WAS': 'MIN', 'MIN': 'WAS',
            'CHI': 'GB', 'GB': 'CHI',
            'DEN': 'LV', 'LV': 'DEN',
            'LAR': 'ARI', 'ARI': 'LAR',
            'CIN': 'BUF', 'BUF': 'CIN',
            'HOU': 'KC', 'KC': 'HOU',
            'PHI': 'LAC', 'LAC': 'PHI',
            # Bye teams
            'CAR': 'BYE', 'NE': 'BYE', 'NYG': 'BYE', 'SF': 'BYE'
        },
        "15": {
            'ATL': 'TB', 'TB': 'ATL',
            'LAC': 'KC', 'KC': 'LAC',
            'BUF': 'NE', 'NE': 'BUF',
            'NYJ': 'JAX', 'JAX': 'NYJ',
            'BAL': 'CIN', 'CIN': 'BAL',
            'LV': 'PHI', 'PHI': 'LV',
            'ARI': 'HOU', 'HOU': 'ARI',
            'WAS': 'NYG', 'NYG': 'WAS',
            'CLE': 'CHI', 'CHI': 'CLE',
            'DET': 'LAR', 'LAR': 'DET',
            'TEN': 'SF', 'SF': 'TEN',
            'CAR': 'NO', 'NO': 'CAR',
            'GB': 'DEN', 'DEN': 'GB',
            'IND': 'SEA', 'SEA': 'IND',
            'MIN': 'DAL', 'DAL': 'MIN',
            'MIA': 'PIT', 'PIT': 'MIA'
        },
        "16": {
            'LAR': 'SEA', 'SEA': 'LAR',
            'GB': 'CHI', 'CHI': 'GB',
            'PHI': 'WAS', 'WAS': 'PHI',
            'KC': 'TEN', 'TEN': 'KC',
            'NYJ': 'NO', 'NO': 'NYJ',
            'NE': 'BAL', 'BAL': 'NE',
            'BUF': 'CLE', 'CLE': 'BUF',
            'TB': 'CAR', 'CAR': 'TB',
            'MIN': 'NYG', 'NYG': 'MIN',
            'LAC': 'DAL', 'DAL': 'LAC',
            'ATL': 'ARI', 'ARI': 'ATL',
            'JAX': 'DEN', 'DEN': 'JAX',
            'PIT': 'DET', 'DET': 'PIT',
            'LV': 'HOU', 'HOU': 'LV',
            'CIN': 'MIA', 'MIA': 'CIN',
            'SF': 'IND', 'IND': 'SF'
        },
        "17": {
            # Christmas Day
            'DAL': 'WAS', 'WAS': 'DAL',
            'DET': 'MIN', 'MIN': 'DET',
            'DEN': 'KC', 'KC': 'DEN',
            # Regular games
            'NYG': 'LV', 'LV': 'NYG',
            'HOU': 'LAC', 'LAC': 'HOU',
            'ARI': 'CIN', 'CIN': 'ARI',
            'BAL': 'GB', 'GB': 'BAL',
            'SEA': 'CAR', 'CAR': 'SEA',
            'NO': 'TEN', 'TEN': 'NO',
            'PIT': 'CLE', 'CLE': 'PIT',
            'NE': 'NYJ', 'NYJ': 'NE',
            'JAX': 'IND', 'IND': 'JAX',
            'TB': 'MIA', 'MIA': 'TB',
            'PHI': 'BUF', 'BUF': 'PHI',
            'CHI': 'SF', 'SF': 'CHI',
            'LAR': 'ATL', 'ATL': 'LAR'
        },
        "18": {
            # Week 18 - all division games
            'NYJ': 'BUF', 'BUF': 'NYJ',
            'KC': 'LAC', 'LAC': 'KC',
            'BAL': 'PIT', 'PIT': 'BAL',
            'CLE': 'CIN', 'CIN': 'CLE',
            'MIA': 'NE', 'NE': 'MIA',
            'TEN': 'JAX', 'JAX': 'TEN',
            'LAC': 'DEN', 'DEN': 'LAC',
            'IND': 'HOU', 'HOU': 'IND',
            'DET': 'CHI', 'CHI': 'DET',
            'GB': 'MIN', 'MIN': 'GB',
            'NO': 'ATL', 'ATL': 'NO',
            'SEA': 'SF', 'SF': 'SEA',
            'WAS': 'PHI', 'PHI': 'WAS',
            'DAL': 'NYG', 'NYG': 'DAL',
            'CAR': 'TB', 'TB': 'CAR',
            'ARI': 'LAR', 'LAR': 'ARI'
        }
    }

def save_complete_schedule():
    """Save the complete 18-week schedule"""
    schedule = create_complete_2025_schedule()
    
    # Save as the file your app expects
    with open('nfl_2025_simple_matchups.json', 'w') as f:
        json.dump(schedule, f, indent=2)
    
    print(f"✅ Created complete 18-week NFL schedule!")
    print(f"📅 Weeks included: {list(schedule.keys())}")
    print(f"📊 Total weeks: {len(schedule)}")
    
    # Show sample data
    print(f"\n📋 Week 1 sample: DAL @ PHI")
    print(f"📋 Week 8 sample: WAS @ KC (Monday Night)")
    print(f"📋 Week 18 sample: Division games")

if __name__ == "__main__":
    save_complete_schedule()