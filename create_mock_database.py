#!/usr/bin/env python3
"""
Create Mock Sleeper Database for Testing
This creates the basic table structure and sample data
"""

import sqlite3
import random
from datetime import datetime

def create_mock_sleeper_database():
    """Create a mock Sleeper database with sample data for testing"""
    
    print("🗄️ Creating Mock Sleeper Database...")
    print("=" * 50)
    
    # Connect to database
    conn = sqlite3.connect('sleeper_enhanced.db')
    cursor = conn.cursor()
    
    # Create players table
    print("📋 Creating players table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id TEXT PRIMARY KEY,
            full_name TEXT,
            position TEXT,
            team TEXT,
            active INTEGER DEFAULT 1
        )
    ''')
    
    # Create weekly_stats table
    print("📊 Creating weekly_stats table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weekly_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            season INTEGER,
            week INTEGER,
            gp INTEGER DEFAULT 1,
            pts_ppr REAL,
            pts_half_ppr REAL,
            pts_std REAL,
            pass_yd INTEGER DEFAULT 0,
            pass_td INTEGER DEFAULT 0,
            pass_att INTEGER DEFAULT 0,
            pass_int INTEGER DEFAULT 0,
            rush_yd INTEGER DEFAULT 0,
            rush_td INTEGER DEFAULT 0,
            rush_att INTEGER DEFAULT 0,
            rec INTEGER DEFAULT 0,
            rec_yd INTEGER DEFAULT 0,
            rec_td INTEGER DEFAULT 0,
            rec_tgt INTEGER DEFAULT 0,
            off_snp INTEGER DEFAULT 0,
            idp_sack REAL DEFAULT 0,
            idp_qb_hit REAL DEFAULT 0,
            idp_int REAL DEFAULT 0,
            idp_pass_def REAL DEFAULT 0,
            team_sack REAL DEFAULT 0,
            team_int REAL DEFAULT 0,
            pts_allow REAL DEFAULT 22,
            yds_allow REAL DEFAULT 350,
            fan_pts_allow_qb REAL DEFAULT 18,
            fan_pts_allow_rb REAL DEFAULT 12,
            fan_pts_allow_wr REAL DEFAULT 25,
            fan_pts_allow_te REAL DEFAULT 10,
            fan_pts_allow_k REAL DEFAULT 8,
            fgm INTEGER DEFAULT 0,
            fga INTEGER DEFAULT 0,
            fgm_20_29 INTEGER DEFAULT 0,
            fgm_30_39 INTEGER DEFAULT 0,
            fgm_40_49 INTEGER DEFAULT 0,
            fgm_50_59 INTEGER DEFAULT 0,
            xpm INTEGER DEFAULT 0,
            xpa INTEGER DEFAULT 0,
            kick_pts REAL DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players (player_id)
        )
    ''')
    
    # Sample players data
    print("👥 Adding sample players...")
    sample_players = [
        # QBs
        ('QB_001', 'Josh Allen', 'QB', 'BUF'),
        ('QB_002', 'Lamar Jackson', 'QB', 'BAL'),
        ('QB_003', 'Patrick Mahomes', 'QB', 'KC'),
        ('QB_004', 'Dak Prescott', 'QB', 'DAL'),
        ('QB_005', 'Tua Tagovailoa', 'QB', 'MIA'),
        
        # RBs
        ('RB_001', 'Christian McCaffrey', 'RB', 'SF'),
        ('RB_002', 'Saquon Barkley', 'RB', 'PHI'),
        ('RB_003', 'Derrick Henry', 'RB', 'BAL'),
        ('RB_004', 'Josh Jacobs', 'RB', 'GB'),
        ('RB_005', 'Alvin Kamara', 'RB', 'NO'),
        
        # WRs
        ('WR_001', 'Tyreek Hill', 'WR', 'MIA'),
        ('WR_002', 'Stefon Diggs', 'WR', 'HOU'),
        ('WR_003', 'A.J. Brown', 'WR', 'PHI'),
        ('WR_004', 'Cooper Kupp', 'WR', 'LAR'),
        ('WR_005', 'CeeDee Lamb', 'WR', 'DAL'),
        
        # TEs
        ('TE_001', 'Travis Kelce', 'TE', 'KC'),
        ('TE_002', 'George Kittle', 'TE', 'SF'),
        ('TE_003', 'Mark Andrews', 'TE', 'BAL'),
        ('TE_004', 'Sam LaPorta', 'TE', 'DET'),
        
        # Kickers
        ('K_001', 'Harrison Butker', 'K', 'KC'),
        ('K_002', 'Brandon Aubrey', 'K', 'DAL'),
        ('K_003', 'Tyler Bass', 'K', 'BUF'),
        
        # Defenses (for opponent stats)
        ('DEF_BUF', 'Buffalo Bills', 'DEF', 'BUF'),
        ('DEF_MIA', 'Miami Dolphins', 'DEF', 'MIA'),
        ('DEF_KC', 'Kansas City Chiefs', 'DEF', 'KC'),
        ('DEF_SF', 'San Francisco 49ers', 'DEF', 'SF'),
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO players (player_id, full_name, position, team)
        VALUES (?, ?, ?, ?)
    ''', sample_players)
    
    print(f"   ✅ Added {len(sample_players)} sample players")
    
    # Generate sample weekly stats
    print("📈 Generating sample weekly stats...")
    
    # Position-specific stat generators
    def generate_qb_stats(week):
        base_pts = random.uniform(15, 25)
        return {
            'pts_ppr': base_pts,
            'pts_half_ppr': base_pts * 0.95,
            'pts_std': base_pts * 0.85,
            'pass_yd': random.randint(200, 350),
            'pass_td': random.randint(1, 4),
            'pass_att': random.randint(25, 45),
            'pass_int': random.randint(0, 2),
            'rush_yd': random.randint(0, 50),
            'rush_td': random.randint(0, 1),
            'rush_att': random.randint(2, 8),
            'off_snp': random.randint(55, 75)
        }
    
    def generate_rb_stats(week):
        base_pts = random.uniform(8, 20)
        return {
            'pts_ppr': base_pts,
            'pts_half_ppr': base_pts * 0.9,
            'pts_std': base_pts * 0.8,
            'rush_yd': random.randint(40, 150),
            'rush_td': random.randint(0, 2),
            'rush_att': random.randint(10, 25),
            'rec': random.randint(2, 8),
            'rec_yd': random.randint(10, 80),
            'rec_td': random.randint(0, 1),
            'rec_tgt': random.randint(3, 10),
            'off_snp': random.randint(35, 65)
        }
    
    def generate_wr_stats(week):
        base_pts = random.uniform(5, 18)
        return {
            'pts_ppr': base_pts,
            'pts_half_ppr': base_pts * 0.85,
            'pts_std': base_pts * 0.7,
            'rec': random.randint(3, 10),
            'rec_yd': random.randint(30, 120),
            'rec_td': random.randint(0, 2),
            'rec_tgt': random.randint(5, 15),
            'off_snp': random.randint(40, 70)
        }
    
    def generate_te_stats(week):
        base_pts = random.uniform(3, 15)
        return {
            'pts_ppr': base_pts,
            'pts_half_ppr': base_pts * 0.85,
            'pts_std': base_pts * 0.7,
            'rec': random.randint(2, 8),
            'rec_yd': random.randint(20, 90),
            'rec_td': random.randint(0, 1),
            'rec_tgt': random.randint(3, 10),
            'off_snp': random.randint(30, 60)
        }
    
    def generate_k_stats(week):
        base_pts = random.uniform(5, 12)
        return {
            'pts_ppr': base_pts,
            'pts_half_ppr': base_pts,
            'pts_std': base_pts,
            'fgm': random.randint(1, 4),
            'fga': random.randint(2, 5),
            'fgm_20_29': random.randint(0, 1),
            'fgm_30_39': random.randint(0, 2),
            'fgm_40_49': random.randint(0, 1),
            'fgm_50_59': random.randint(0, 1),
            'xpm': random.randint(1, 5),
            'xpa': random.randint(1, 5),
            'kick_pts': base_pts
        }
    
    def generate_def_stats(week):
        return {
            'pts_ppr': random.uniform(6, 15),
            'pts_half_ppr': random.uniform(6, 15),
            'pts_std': random.uniform(6, 15),
            'idp_sack': random.uniform(0, 3),
            'idp_qb_hit': random.uniform(2, 8),
            'idp_int': random.uniform(0, 2),
            'idp_pass_def': random.uniform(5, 15),
            'team_sack': random.uniform(1, 4),
            'team_int': random.uniform(0, 2),
            'pts_allow': random.uniform(14, 28),
            'yds_allow': random.uniform(280, 420),
            'fan_pts_allow_qb': random.uniform(15, 22),
            'fan_pts_allow_rb': random.uniform(10, 16),
            'fan_pts_allow_wr': random.uniform(20, 30),
            'fan_pts_allow_te': random.uniform(8, 12),
            'fan_pts_allow_k': random.uniform(6, 10)
        }
    
    # Generate stats for weeks 1-9 (need history for week 9+ predictions)
    total_records = 0
    for week in range(1, 10):
        print(f"   📅 Generating Week {week} stats...")
        
        for player_id, name, position, team in sample_players:
            # Generate position-appropriate stats
            if position == 'QB':
                stats = generate_qb_stats(week)
            elif position == 'RB':
                stats = generate_rb_stats(week)
            elif position == 'WR':
                stats = generate_wr_stats(week)
            elif position == 'TE':
                stats = generate_te_stats(week)
            elif position == 'K':
                stats = generate_k_stats(week)
            elif position == 'DEF':
                stats = generate_def_stats(week)
            else:
                continue
            
            # Insert weekly stats
            cursor.execute('''
                INSERT INTO weekly_stats (
                    player_id, season, week, gp, pts_ppr, pts_half_ppr, pts_std,
                    pass_yd, pass_td, pass_att, pass_int, rush_yd, rush_td, rush_att,
                    rec, rec_yd, rec_td, rec_tgt, off_snp, idp_sack, idp_qb_hit,
                    idp_int, idp_pass_def, team_sack, team_int, pts_allow, yds_allow,
                    fan_pts_allow_qb, fan_pts_allow_rb, fan_pts_allow_wr,
                    fan_pts_allow_te, fan_pts_allow_k, fgm, fga, fgm_20_29,
                    fgm_30_39, fgm_40_49, fgm_50_59, xpm, xpa, kick_pts
                ) VALUES (?, 2024, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_id, week,
                stats.get('pts_ppr', 0), stats.get('pts_half_ppr', 0), stats.get('pts_std', 0),
                stats.get('pass_yd', 0), stats.get('pass_td', 0), stats.get('pass_att', 0), stats.get('pass_int', 0),
                stats.get('rush_yd', 0), stats.get('rush_td', 0), stats.get('rush_att', 0),
                stats.get('rec', 0), stats.get('rec_yd', 0), stats.get('rec_td', 0), stats.get('rec_tgt', 0),
                stats.get('off_snp', 0), stats.get('idp_sack', 0), stats.get('idp_qb_hit', 0),
                stats.get('idp_int', 0), stats.get('idp_pass_def', 0), stats.get('team_sack', 0),
                stats.get('team_int', 0), stats.get('pts_allow', 22), stats.get('yds_allow', 350),
                stats.get('fan_pts_allow_qb', 18), stats.get('fan_pts_allow_rb', 12),
                stats.get('fan_pts_allow_wr', 25), stats.get('fan_pts_allow_te', 10),
                stats.get('fan_pts_allow_k', 8), stats.get('fgm', 0), stats.get('fga', 0),
                stats.get('fgm_20_29', 0), stats.get('fgm_30_39', 0), stats.get('fgm_40_49', 0),
                stats.get('fgm_50_59', 0), stats.get('xpm', 0), stats.get('xpa', 0), stats.get('kick_pts', 0)
            ))
            
            total_records += 1
    
    # Commit changes
    conn.commit()
    
    # Verify database
    print("\n🔍 Verifying database...")
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM weekly_stats")
    stats_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(week) FROM weekly_stats WHERE season = 2024")
    latest_week = cursor.fetchone()[0]
    
    print(f"   ✅ {player_count} players")
    print(f"   ✅ {stats_count} weekly stat records")
    print(f"   ✅ Latest week: {latest_week}")
    
    conn.close()
    
    print("\n🎉 Mock database created successfully!")
    print("=" * 50)
    print("🚀 You can now:")
    print("   1. Train ML models: python sleeper_ml_training.py")
    print("   2. Start the app: python app.py")
    print("   3. Test predictions with the sample players")
    
    return True

if __name__ == "__main__":
    create_mock_sleeper_database()