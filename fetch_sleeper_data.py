#!/usr/bin/env python3
"""
Fixed Expert Sleeper Data Fetcher - Column Count Issue Resolved
Precise column mapping with exact count verification
"""

import requests
import sqlite3
import json
import time
from datetime import datetime
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedSleeperFetcher:
    """Column-count-verified Sleeper API data fetcher"""
    
    def __init__(self):
        self.base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FantasyAlpha/2.0 (Fantasy Football ML Training)',
            'Accept': 'application/json'
        })
    
    def fetch_players(self):
        """Fetch all NFL players"""
        logger.info("📥 Fetching players from Sleeper API...")
        
        try:
            url = f"{self.base_url}/players/nfl"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            all_players = response.json()
            
            # Filter for active skill position players
            skill_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
            active_players = []
            
            for player_id, player_info in all_players.items():
                if (player_info.get('position') in skill_positions and 
                    player_info.get('active') == True and
                    player_info.get('team') is not None):
                    
                    active_players.append({
                        'player_id': player_id,
                        'full_name': player_info.get('full_name', ''),
                        'position': player_info.get('position'),
                        'team': player_info.get('team')
                    })
            
            logger.info(f"✅ Found {len(active_players)} active players")
            return active_players
            
        except Exception as e:
            logger.error(f"❌ Error fetching players: {e}")
            return []
    
    def fetch_weekly_stats(self, season=2024, max_week=11):
        """Fetch weekly stats"""
        logger.info(f"📊 Fetching weekly stats for weeks 1-{max_week}...")
        
        all_stats = []
        for week in range(1, max_week + 1):
            logger.info(f"   📅 Week {week}...")
            
            try:
                url = f"{self.base_url}/stats/nfl/regular/{season}/{week}"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    week_stats = response.json()
                    
                    for player_id, stats in week_stats.items():
                        # Only include players with meaningful stats
                        if stats.get('pts_ppr', 0) and stats.get('pts_ppr', 0) > 0:
                            stats['player_id'] = player_id
                            stats['season'] = season
                            stats['week'] = week
                            stats['gp'] = 1
                            all_stats.append(stats)
                    
                    logger.info(f"      ✅ {len([s for s in week_stats.values() if s.get('pts_ppr', 0) > 0])} players")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"      ❌ Week {week}: {e}")
                continue
        
        logger.info(f"✅ Total stats: {len(all_stats)}")
        return all_stats
    
    def create_database(self, db_path="sleeper_enhanced.db"):
        """Create database with exactly matching column counts"""
        logger.info("🗄️ Creating database with verified column mapping...")
        
        # Backup existing
        try:
            import shutil
            shutil.copy(db_path, f"{db_path}.backup")
            logger.info("📁 Backed up existing database")
        except:
            pass
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS weekly_stats")
        cursor.execute("DROP TABLE IF EXISTS players")
        
        # Create players table
        logger.info("👥 Creating players table...")
        cursor.execute('''
            CREATE TABLE players (
                player_id TEXT PRIMARY KEY,
                full_name TEXT,
                position TEXT,
                team TEXT,
                active INTEGER DEFAULT 1
            )
        ''')
        
        # Create weekly_stats table - EXACTLY count columns
        logger.info("📊 Creating weekly_stats table...")
        cursor.execute('''
            CREATE TABLE weekly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                season INTEGER,
                week INTEGER,
                gp INTEGER DEFAULT 1,
                pts_ppr REAL DEFAULT 0,
                pts_half_ppr REAL DEFAULT 0,
                pts_std REAL DEFAULT 0,
                pass_yd INTEGER DEFAULT 0,
                pass_td INTEGER DEFAULT 0,
                pass_att INTEGER DEFAULT 0,
                pass_cmp INTEGER DEFAULT 0,
                pass_int INTEGER DEFAULT 0,
                pass_2pt INTEGER DEFAULT 0,
                rush_yd INTEGER DEFAULT 0,
                rush_td INTEGER DEFAULT 0,
                rush_att INTEGER DEFAULT 0,
                rush_2pt INTEGER DEFAULT 0,
                rec INTEGER DEFAULT 0,
                rec_yd INTEGER DEFAULT 0,
                rec_td INTEGER DEFAULT 0,
                rec_tgt INTEGER DEFAULT 0,
                rec_2pt INTEGER DEFAULT 0,
                off_snp INTEGER DEFAULT 0,
                def_snp INTEGER DEFAULT 0,
                st_snp INTEGER DEFAULT 0,
                fgm INTEGER DEFAULT 0,
                fga INTEGER DEFAULT 0,
                fgm_0_19 INTEGER DEFAULT 0,
                fgm_20_29 INTEGER DEFAULT 0,
                fgm_30_39 INTEGER DEFAULT 0,
                fgm_40_49 INTEGER DEFAULT 0,
                fgm_50_59 INTEGER DEFAULT 0,
                fgm_60p INTEGER DEFAULT 0,
                xpm INTEGER DEFAULT 0,
                xpa INTEGER DEFAULT 0,
                idp_solo INTEGER DEFAULT 0,
                idp_ast INTEGER DEFAULT 0,
                idp_sack REAL DEFAULT 0,
                idp_int INTEGER DEFAULT 0,
                idp_fum_rec INTEGER DEFAULT 0,
                idp_ff INTEGER DEFAULT 0,
                idp_td INTEGER DEFAULT 0,
                idp_safety INTEGER DEFAULT 0,
                idp_pass_def INTEGER DEFAULT 0,
                idp_blk_kick INTEGER DEFAULT 0,
                idp_tkl_loss REAL DEFAULT 0,
                idp_qb_hit INTEGER DEFAULT 0,
                pts_allow INTEGER DEFAULT 0,
                yds_allow INTEGER DEFAULT 0,
                pass_yds_allow INTEGER DEFAULT 0,
                rush_yds_allow INTEGER DEFAULT 0,
                fum INTEGER DEFAULT 0,
                fum_lost INTEGER DEFAULT 0,
                bonus_rec_yd INTEGER DEFAULT 0,
                bonus_rush_yd INTEGER DEFAULT 0,
                bonus_pass_yd INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        ''')
        
        # Count columns to verify (excluding id and created_at which are auto-generated)
        cursor.execute("PRAGMA table_info(weekly_stats)")
        columns = cursor.fetchall()
        insertable_columns = [col[1] for col in columns if col[1] not in ['id', 'created_at']]
        
        logger.info(f"🔍 Table has {len(insertable_columns)} insertable columns")
        logger.info(f"📋 Columns: {', '.join(insertable_columns)}")
        
        # Fetch real data
        players = self.fetch_players()
        if not players:
            logger.error("❌ No players fetched")
            return False
        
        # Insert players
        logger.info("👥 Inserting players...")
        cursor.executemany('''
            INSERT INTO players (player_id, full_name, position, team, active)
            VALUES (?, ?, ?, ?, ?)
        ''', [(p['player_id'], p['full_name'], p['position'], p['team'], 1) for p in players])
        
        # Fetch stats
        stats = self.fetch_weekly_stats()
        if not stats:
            logger.error("❌ No stats fetched")
            return False
        
        # Insert stats with EXACT column mapping
        logger.info("📊 Inserting weekly stats with exact column mapping...")
        
        stats_tuples = []
        for stat in stats:
            # Create tuple with EXACTLY the right number of values for insertable columns
            stats_tuples.append((
                stat.get('player_id', ''),           # player_id
                stat.get('season', 2024),            # season  
                stat.get('week', 1),                 # week
                stat.get('gp', 1),                   # gp
                float(stat.get('pts_ppr', 0) or 0),  # pts_ppr
                float(stat.get('pts_half_ppr', 0) or 0), # pts_half_ppr
                float(stat.get('pts_std', 0) or 0),  # pts_std
                int(stat.get('pass_yd', 0) or 0),    # pass_yd
                int(stat.get('pass_td', 0) or 0),    # pass_td
                int(stat.get('pass_att', 0) or 0),   # pass_att
                int(stat.get('pass_cmp', 0) or 0),   # pass_cmp
                int(stat.get('pass_int', 0) or 0),   # pass_int
                int(stat.get('pass_2pt', 0) or 0),   # pass_2pt
                int(stat.get('rush_yd', 0) or 0),    # rush_yd
                int(stat.get('rush_td', 0) or 0),    # rush_td
                int(stat.get('rush_att', 0) or 0),   # rush_att
                int(stat.get('rush_2pt', 0) or 0),   # rush_2pt
                int(stat.get('rec', 0) or 0),        # rec
                int(stat.get('rec_yd', 0) or 0),     # rec_yd
                int(stat.get('rec_td', 0) or 0),     # rec_td
                int(stat.get('rec_tgt', 0) or 0),    # rec_tgt
                int(stat.get('rec_2pt', 0) or 0),    # rec_2pt
                int(stat.get('off_snp', 0) or 0),    # off_snp
                int(stat.get('def_snp', 0) or 0),    # def_snp
                int(stat.get('st_snp', 0) or 0),     # st_snp
                int(stat.get('fgm', 0) or 0),        # fgm
                int(stat.get('fga', 0) or 0),        # fga
                int(stat.get('fgm_0_19', 0) or 0),   # fgm_0_19
                int(stat.get('fgm_20_29', 0) or 0),  # fgm_20_29
                int(stat.get('fgm_30_39', 0) or 0),  # fgm_30_39
                int(stat.get('fgm_40_49', 0) or 0),  # fgm_40_49
                int(stat.get('fgm_50_59', 0) or 0),  # fgm_50_59
                int(stat.get('fgm_60p', 0) or 0),    # fgm_60p
                int(stat.get('xpm', 0) or 0),        # xpm
                int(stat.get('xpa', 0) or 0),        # xpa
                int(stat.get('idp_solo', 0) or 0),   # idp_solo
                int(stat.get('idp_ast', 0) or 0),    # idp_ast
                float(stat.get('idp_sack', 0) or 0), # idp_sack
                int(stat.get('idp_int', 0) or 0),    # idp_int
                int(stat.get('idp_fum_rec', 0) or 0), # idp_fum_rec
                int(stat.get('idp_ff', 0) or 0),     # idp_ff
                int(stat.get('idp_td', 0) or 0),     # idp_td
                int(stat.get('idp_safety', 0) or 0), # idp_safety
                int(stat.get('idp_pass_def', 0) or 0), # idp_pass_def
                int(stat.get('idp_blk_kick', 0) or 0), # idp_blk_kick
                float(stat.get('idp_tkl_loss', 0) or 0), # idp_tkl_loss
                int(stat.get('idp_qb_hit', 0) or 0), # idp_qb_hit
                int(stat.get('pts_allow', 0) or 0),  # pts_allow
                int(stat.get('yds_allow', 0) or 0),  # yds_allow
                int(stat.get('pass_yds_allow', 0) or 0), # pass_yds_allow
                int(stat.get('rush_yds_allow', 0) or 0), # rush_yds_allow
                int(stat.get('fum', 0) or 0),        # fum
                int(stat.get('fum_lost', 0) or 0),   # fum_lost
                int(stat.get('bonus_rec_yd', 0) or 0), # bonus_rec_yd
                int(stat.get('bonus_rush_yd', 0) or 0), # bonus_rush_yd
                int(stat.get('bonus_pass_yd', 0) or 0)  # bonus_pass_yd
                # Total: 54 values for 54 insertable columns
            ))
        
        logger.info(f"🔍 Prepared {len(stats_tuples)} stat records")
        logger.info(f"🔍 Each record has {len(stats_tuples[0])} values")
        logger.info(f"🔍 Table expects {len(insertable_columns)} values")
        
        # Verify count match
        if len(stats_tuples[0]) != len(insertable_columns):
            logger.error(f"❌ Column count mismatch: {len(stats_tuples[0])} values vs {len(insertable_columns)} columns")
            return False
        
        # Execute insert with exact column list
        cursor.executemany(f'''
            INSERT OR REPLACE INTO weekly_stats (
                {', '.join(insertable_columns)}
            ) VALUES ({', '.join(['?'] * len(insertable_columns))})
        ''', stats_tuples)
        
        conn.commit()
        
        # Verify data
        logger.info("🔍 Verifying database...")
        cursor.execute("SELECT COUNT(*) FROM players WHERE position IN ('QB', 'RB', 'WR', 'TE', 'K')")
        player_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM weekly_stats WHERE season = 2024")
        stats_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(week) FROM weekly_stats WHERE season = 2024")
        latest_week = cursor.fetchone()[0]
        
        # Sample top performances
        cursor.execute("""
            SELECT p.full_name, p.position, p.team, w.week, w.pts_ppr
            FROM players p
            JOIN weekly_stats w ON p.player_id = w.player_id
            WHERE w.season = 2024 AND w.pts_ppr > 15
            ORDER BY w.pts_ppr DESC
            LIMIT 5
        """)
        top_performances = cursor.fetchall()
        
        conn.close()
        
        logger.info(f"✅ {player_count:,} players")
        logger.info(f"✅ {stats_count:,} weekly stat records") 
        logger.info(f"✅ Data through Week {latest_week}")
        
        if top_performances:
            logger.info("🏆 Top performances:")
            for name, pos, team, week, pts in top_performances:
                logger.info(f"   {name} ({pos}-{team}): {pts:.1f} pts (Week {week})")
        
        logger.info("\n🎉 Database created successfully!")
        logger.info("🚀 Ready for ML training with real NFL data")
        
        return True

def main():
    """Main execution"""
    print("🔧 Fixed Expert Sleeper Data Fetcher")
    print("=" * 50)
    print("✅ Column count issue resolved")
    print("📊 Verified column mapping")
    print("🗄️ Production-grade database creation")
    
    try:
        response = input("\n🤔 Create database with real Sleeper data? (y/n): ").lower()
        if response not in ['y', 'yes']:
            print("⏹️ Cancelled")
            return
    except KeyboardInterrupt:
        print("\n⏹️ Cancelled")
        return
    
    fetcher = FixedSleeperFetcher()
    
    start_time = time.time()
    success = fetcher.create_database()
    end_time = time.time()
    
    if success:
        duration = end_time - start_time
        print(f"\n✅ Success! Database created in {duration:.1f} seconds")
        print("\n🚀 Next Steps:")
        print("   1. Train ML models: python sleeper_ml_training.py")
        print("   2. Start app: python app.py")
        print("   3. Make predictions with real data!")
    else:
        print("\n❌ Database creation failed")
        print("📝 Check the error details above")

if __name__ == "__main__":
    main()