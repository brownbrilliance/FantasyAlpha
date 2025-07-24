"""
ml_features.py - Sleeper-Native Feature Engineering
==================================================
Features built ENTIRELY from Sleeper API data - no legacy features
Based on actual Sleeper investigation results
"""

import numpy as np
import math
import sqlite3
from datetime import datetime

def prepare_sleeper_native_features(player_data, sleeper_db_path="sleeper_enhanced.db"):
    """
    Build ML features entirely from Sleeper API data - INDIVIDUAL WEEKLY STATS
    No averages - we want raw weekly performance data for ML training
    """
    features = {}
    
    # Basic player info
    player_name = player_data.get('name', '')
    position = player_data.get('position', 'WR')
    team = player_data.get('team', '')
    opponent = player_data.get('opponent', '')
    week = int(player_data.get('week', 9))
    
    # Connect to Sleeper database
    try:
        conn = sqlite3.connect(sleeper_db_path)
    except Exception as e:
        print(f"Database connection failed: {e}")
        return _get_default_features(position, week)
    
    # =========================================================================
    # INDIVIDUAL WEEKLY STATS (Last 3 weeks) - RAW DATA FOR ML
    # =========================================================================
    
    # Get player's last 3 individual weekly performances
    weekly_stats = get_player_weekly_stats_individual(conn, player_name, position, team, week)
    
    if weekly_stats and len(weekly_stats) > 0:
        # INDIVIDUAL WEEK STATS (most recent 3 weeks)
        # Week 1 (most recent)
        w1 = weekly_stats[0] if len(weekly_stats) > 0 else {}
        features['week1_pts_ppr'] = w1.get('pts_ppr', 0)
        features['week1_pts_half_ppr'] = w1.get('pts_half_ppr', 0)
        features['week1_pts_std'] = w1.get('pts_std', 0)
        features['week1_pass_yd'] = w1.get('pass_yd', 0)
        features['week1_pass_td'] = w1.get('pass_td', 0)
        features['week1_pass_att'] = w1.get('pass_att', 0)
        features['week1_pass_int'] = w1.get('pass_int', 0)
        features['week1_rush_yd'] = w1.get('rush_yd', 0)
        features['week1_rush_td'] = w1.get('rush_td', 0)
        features['week1_rush_att'] = w1.get('rush_att', 0)
        features['week1_rec'] = w1.get('rec', 0)
        features['week1_rec_yd'] = w1.get('rec_yd', 0)
        features['week1_rec_td'] = w1.get('rec_td', 0)
        features['week1_rec_tgt'] = w1.get('rec_tgt', 0)
        features['week1_off_snp'] = w1.get('off_snp', 0)
        
        # Week 2 (second most recent)
        w2 = weekly_stats[1] if len(weekly_stats) > 1 else {}
        features['week2_pts_ppr'] = w2.get('pts_ppr', 0)
        features['week2_pts_half_ppr'] = w2.get('pts_half_ppr', 0)
        features['week2_pts_std'] = w2.get('pts_std', 0)
        features['week2_pass_yd'] = w2.get('pass_yd', 0)
        features['week2_pass_td'] = w2.get('pass_td', 0)
        features['week2_pass_att'] = w2.get('pass_att', 0)
        features['week2_pass_int'] = w2.get('pass_int', 0)
        features['week2_rush_yd'] = w2.get('rush_yd', 0)
        features['week2_rush_td'] = w2.get('rush_td', 0)
        features['week2_rush_att'] = w2.get('rush_att', 0)
        features['week2_rec'] = w2.get('rec', 0)
        features['week2_rec_yd'] = w2.get('rec_yd', 0)
        features['week2_rec_td'] = w2.get('rec_td', 0)
        features['week2_rec_tgt'] = w2.get('rec_tgt', 0)
        features['week2_off_snp'] = w2.get('off_snp', 0)
        
        # Week 3 (third most recent)
        w3 = weekly_stats[2] if len(weekly_stats) > 2 else {}
        features['week3_pts_ppr'] = w3.get('pts_ppr', 0)
        features['week3_pts_half_ppr'] = w3.get('pts_half_ppr', 0)
        features['week3_pts_std'] = w3.get('pts_std', 0)
        features['week3_pass_yd'] = w3.get('pass_yd', 0)
        features['week3_pass_td'] = w3.get('pass_td', 0)
        features['week3_pass_att'] = w3.get('pass_att', 0)
        features['week3_pass_int'] = w3.get('pass_int', 0)
        features['week3_rush_yd'] = w3.get('rush_yd', 0)
        features['week3_rush_td'] = w3.get('rush_td', 0)
        features['week3_rush_att'] = w3.get('rush_att', 0)
        features['week3_rec'] = w3.get('rec', 0)
        features['week3_rec_yd'] = w3.get('rec_yd', 0)
        features['week3_rec_td'] = w3.get('rec_td', 0)
        features['week3_rec_tgt'] = w3.get('rec_tgt', 0)
        features['week3_off_snp'] = w3.get('off_snp', 0)
        
        # TREND ANALYSIS from individual weeks
        pts_trend = []
        tgt_trend = []
        snp_trend = []
        
        for w in weekly_stats:
            if w.get('pts_ppr') is not None:
                pts_trend.append(w['pts_ppr'])
            if w.get('rec_tgt') is not None:
                tgt_trend.append(w['rec_tgt'])
            if w.get('off_snp') is not None:
                snp_trend.append(w['off_snp'])
        
        # Trend features (are they trending up or down?)
        features['pts_trend_up'] = 1 if len(pts_trend) >= 2 and pts_trend[0] > pts_trend[-1] else 0
        features['pts_volatility'] = np.std(pts_trend) if len(pts_trend) > 1 else 0
        features['target_trend_up'] = 1 if len(tgt_trend) >= 2 and tgt_trend[0] > tgt_trend[-1] else 0
        features['snap_consistency'] = 1 - (np.std(snp_trend) / np.mean(snp_trend)) if len(snp_trend) > 1 and np.mean(snp_trend) > 0 else 0
        
        # Games with production
        features['games_with_td'] = sum(1 for w in weekly_stats if (w.get('pass_td', 0) + w.get('rush_td', 0) + w.get('rec_td', 0)) > 0)
        features['games_with_100yd'] = sum(1 for w in weekly_stats if (w.get('pass_yd', 0) + w.get('rush_yd', 0) + w.get('rec_yd', 0)) >= 100)
        
        # Weekly highs and lows
        features['best_week_pts'] = max([w.get('pts_ppr', 0) for w in weekly_stats])
        features['worst_week_pts'] = min([w.get('pts_ppr', 0) for w in weekly_stats])
        features['pts_ceiling'] = features['best_week_pts']
        features['pts_floor'] = features['worst_week_pts']

    else:
        # Zero out all individual week stats if no data
        week_features = ['pts_ppr', 'pts_half_ppr', 'pts_std', 'pass_yd', 'pass_td', 'pass_att', 'pass_int',
                        'rush_yd', 'rush_td', 'rush_att', 'rec', 'rec_yd', 'rec_td', 'rec_tgt', 'off_snp']
        
        for week_num in ['week1', 'week2', 'week3']:
            for stat in week_features:
                features[f'{week_num}_{stat}'] = 0
        
        # Zero trend features
        features.update({
            'pts_trend_up': 0, 'pts_volatility': 0, 'target_trend_up': 0, 'snap_consistency': 0,
            'games_with_td': 0, 'games_with_100yd': 0, 'best_week_pts': 0, 'worst_week_pts': 0,
            'pts_ceiling': 0, 'pts_floor': 0
        })
    
    # =========================================================================
    # OPPONENT WEEKLY DEFENSIVE STATS (Individual weeks, not averages)
    # =========================================================================
    
    opp_def_weekly = get_opponent_defensive_weekly_stats(conn, opponent, week)
    
    if opp_def_weekly and len(opp_def_weekly) > 0:
        # Most recent opponent defensive performances
        od1 = opp_def_weekly[0] if len(opp_def_weekly) > 0 else {}
        features['opp_week1_sacks'] = od1.get('sacks', 0)
        features['opp_week1_qb_hits'] = od1.get('qb_hits', 0)
        features['opp_week1_ints'] = od1.get('ints', 0)
        features['opp_week1_pass_def'] = od1.get('pass_def', 0)
        features['opp_week1_pts_allowed'] = od1.get('pts_allow', 22)
        features['opp_week1_yds_allowed'] = od1.get('yds_allow', 350)
        
        od2 = opp_def_weekly[1] if len(opp_def_weekly) > 1 else {}
        features['opp_week2_sacks'] = od2.get('sacks', 0)
        features['opp_week2_qb_hits'] = od2.get('qb_hits', 0)
        features['opp_week2_ints'] = od2.get('ints', 0)
        features['opp_week2_pass_def'] = od2.get('pass_def', 0)
        features['opp_week2_pts_allowed'] = od2.get('pts_allow', 22)
        features['opp_week2_yds_allowed'] = od2.get('yds_allow', 350)
        
        # Opponent defensive trends
        sack_trend = [w.get('sacks', 0) for w in opp_def_weekly]
        features['opp_def_trending_up'] = 1 if len(sack_trend) >= 2 and sack_trend[0] > sack_trend[-1] else 0
        features['opp_def_volatility'] = np.std(sack_trend) if len(sack_trend) > 1 else 0
        
    else:
        # Default opponent defensive stats
        opp_def_features = ['sacks', 'qb_hits', 'ints', 'pass_def', 'pts_allowed', 'yds_allowed']
        for week_num in ['week1', 'week2']:
            for stat in opp_def_features:
                default_val = 22 if stat == 'pts_allowed' else 350 if stat == 'yds_allowed' else 2
                features[f'opp_{week_num}_{stat}'] = default_val
        
        features['opp_def_trending_up'] = 0
        features['opp_def_volatility'] = 0
    
    # =========================================================================
    # FANTASY POINTS ALLOWED BY POSITION (Weekly, not averaged)
    # =========================================================================
    
    dst_weekly_stats = get_dst_weekly_stats(conn, opponent, week)
    
    if dst_weekly_stats and len(dst_weekly_stats) > 0:
        # Most recent week fantasy points allowed
        dst1 = dst_weekly_stats[0] if len(dst_weekly_stats) > 0 else {}
        features['opp_week1_fan_pts_allow_qb'] = dst1.get('fan_pts_allow_qb', 18)
        features['opp_week1_fan_pts_allow_rb'] = dst1.get('fan_pts_allow_rb', 12)
        features['opp_week1_fan_pts_allow_wr'] = dst1.get('fan_pts_allow_wr', 25)
        features['opp_week1_fan_pts_allow_te'] = dst1.get('fan_pts_allow_te', 10)
        features['opp_week1_fan_pts_allow_k'] = dst1.get('fan_pts_allow_k', 8)
        
        # Second most recent week
        dst2 = dst_weekly_stats[1] if len(dst_weekly_stats) > 1 else {}
        features['opp_week2_fan_pts_allow_qb'] = dst2.get('fan_pts_allow_qb', 18)
        features['opp_week2_fan_pts_allow_rb'] = dst2.get('fan_pts_allow_rb', 12)
        features['opp_week2_fan_pts_allow_wr'] = dst2.get('fan_pts_allow_wr', 25)
        features['opp_week2_fan_pts_allow_te'] = dst2.get('fan_pts_allow_te', 10)
        features['opp_week2_fan_pts_allow_k'] = dst2.get('fan_pts_allow_k', 8)
        
        # Position-specific matchup (most recent week)
        position_matchup_map = {
            'QB': features['opp_week1_fan_pts_allow_qb'],
            'RB': features['opp_week1_fan_pts_allow_rb'],
            'WR': features['opp_week1_fan_pts_allow_wr'],
            'TE': features['opp_week1_fan_pts_allow_te'],
            'K': features['opp_week1_fan_pts_allow_k']
        }
        features['matchup_pts_allowed_week1'] = position_matchup_map.get(position, 15)
        
    else:
        # Default DST weekly stats
        default_dst = {'qb': 18, 'rb': 12, 'wr': 25, 'te': 10, 'k': 8}
        for week_num in ['week1', 'week2']:
            for pos, default_val in default_dst.items():
                features[f'opp_{week_num}_fan_pts_allow_{pos}'] = default_val
        
        features['matchup_pts_allowed_week1'] = default_dst.get(position.lower(), 15)
    
    # =========================================================================
    # BASIC GAME CONTEXT FEATURES
    # =========================================================================
    
    # Week-based features
    features['week_number'] = week
    features['week_sin'] = math.sin(2 * math.pi * week / 18)
    features['week_cos'] = math.cos(2 * math.pi * week / 18)
    features['is_early_season'] = 1 if week <= 6 else 0
    features['is_mid_season'] = 1 if 7 <= week <= 13 else 0
    features['is_late_season'] = 1 if week >= 14 else 0
    
    # Position encoding
    features['is_qb'] = 1 if position == 'QB' else 0
    features['is_rb'] = 1 if position == 'RB' else 0
    features['is_wr'] = 1 if position == 'WR' else 0
    features['is_te'] = 1 if position == 'TE' else 0
    features['is_k'] = 1 if position == 'K' else 0
    features['is_dst'] = 1 if position in ['DEF', 'DST'] else 0
    features['is_db'] = 1 if position in ['CB', 'S', 'SS', 'FS', 'DB'] else 0
    features['is_lb'] = 1 if position in ['LB', 'OLB', 'MLB', 'ILB'] else 0
    features['is_dl'] = 1 if position in ['DE', 'DT', 'DL'] else 0
    
    # Home/Away (would need additional data source or manual input)
    features['is_home'] = 1 if player_data.get('homeAway') == 'HOME' else 0
    
    # Dome stadium (static data)
    dome_teams = ['ATL', 'NO', 'DET', 'MIN', 'ARI', 'LV', 'LAR', 'IND']
    features['is_dome_game'] = 1 if (team in dome_teams or opponent in dome_teams) else 0
    
    # Team strength placeholder
    features['team_off_strength'] = 0.5  # Default neutral strength
    
    # Close database connection
    conn.close()
    
    # =========================================================================
    # BUILD FINAL FEATURE ARRAY
    # =========================================================================
    
    # Define feature order (reduced set for training stability)
    feature_order = [
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
        
        # Opponent defensive stats
        'opp_week1_sacks', 'opp_week1_qb_hits', 'opp_week1_ints', 'opp_week1_pass_def',
        'opp_week1_pts_allowed', 'opp_week1_yds_allowed',
        'opp_week2_sacks', 'opp_week2_qb_hits', 'opp_week2_ints', 'opp_week2_pass_def', 
        'opp_week2_pts_allowed', 'opp_week2_yds_allowed',
        'opp_def_trending_up', 'opp_def_volatility',
        
        # Fantasy points allowed
        'opp_week1_fan_pts_allow_qb', 'opp_week1_fan_pts_allow_rb', 'opp_week1_fan_pts_allow_wr',
        'opp_week1_fan_pts_allow_te', 'opp_week1_fan_pts_allow_k',
        'opp_week2_fan_pts_allow_qb', 'opp_week2_fan_pts_allow_rb', 'opp_week2_fan_pts_allow_wr',
        'opp_week2_fan_pts_allow_te', 'opp_week2_fan_pts_allow_k',
        'matchup_pts_allowed_week1',
        
        # Position and game context
        'is_qb', 'is_rb', 'is_wr', 'is_te', 'is_k',
        'week_number', 'week_sin', 'week_cos', 'is_early_season', 'is_mid_season', 'is_late_season',
        'is_home', 'is_dome_game', 'team_off_strength'
    ]
    
    # Build feature array
    feature_array = []
    for feature_name in feature_order:
        feature_array.append(features.get(feature_name, 0))
    
    print(f"Generated {len(feature_array)} Sleeper-native features for ML model")
    return np.array(feature_array).reshape(1, -1)

def _get_default_features(position, week):
    """Return default feature array when database is unavailable"""
    # Create a basic feature array with defaults
    num_features = 84  # Total number of features in our model
    feature_array = np.zeros(num_features)
    
    # Set some basic defaults based on position
    position_defaults = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8, 'K': 7}
    default_points = position_defaults.get(position, 10)
    
    # Set week 1 points to default
    feature_array[0] = default_points  # week1_pts_ppr
    feature_array[1] = default_points * 0.9  # week1_pts_half_ppr
    feature_array[2] = default_points * 0.8  # week1_pts_std
    
    # Set position encoding
    pos_map = {'QB': 69, 'RB': 70, 'WR': 71, 'TE': 72, 'K': 73}
    if position in pos_map:
        feature_array[pos_map[position]] = 1
    
    # Set week features
    feature_array[74] = week  # week_number
    
    return feature_array.reshape(1, -1)

# =========================================================================
# HELPER FUNCTIONS FOR SLEEPER DATA QUERIES
# =========================================================================

def get_player_weekly_stats_individual(conn, player_name, position, team, week):
    """Get player's individual weekly stats (not averages) for ML training"""
    try:
        query = """
            SELECT w.week, w.pts_ppr, w.pts_half_ppr, w.pts_std,
                   w.pass_yd, w.pass_td, w.pass_att, w.pass_int,
                   w.rush_yd, w.rush_td, w.rush_att,
                   w.rec, w.rec_yd, w.rec_td, w.rec_tgt, w.off_snp
            FROM players p
            JOIN weekly_stats w ON p.player_id = w.player_id
            WHERE p.full_name LIKE ? AND p.position = ? AND p.team = ?
            AND w.season = 2024 AND w.week >= ? AND w.week < ?
            AND w.gp > 0
            ORDER BY w.week DESC
            LIMIT 3
        """
        
        results = conn.execute(query, (f"%{player_name}%", position, team, max(1, week-3), week)).fetchall()
        
        weekly_data = []
        for result in results:
            weekly_data.append({
                'week': result[0],
                'pts_ppr': result[1] or 0,
                'pts_half_ppr': result[2] or 0,
                'pts_std': result[3] or 0,
                'pass_yd': result[4] or 0,
                'pass_td': result[5] or 0,
                'pass_att': result[6] or 0,
                'pass_int': result[7] or 0,
                'rush_yd': result[8] or 0,
                'rush_td': result[9] or 0,
                'rush_att': result[10] or 0,
                'rec': result[11] or 0,
                'rec_yd': result[12] or 0,
                'rec_td': result[13] or 0,
                'rec_tgt': result[14] or 0,
                'off_snp': result[15] or 0
            })
        
        return weekly_data
    except Exception as e:
        print(f"Error getting individual weekly stats: {e}")
        return []

def get_opponent_defensive_weekly_stats(conn, opponent_team, week):
    """Get opponent's individual weekly defensive stats (not averages)"""
    try:
        query = """
            SELECT w.week,
                   COALESCE(SUM(w.idp_sack), 0) as sacks,
                   COALESCE(SUM(w.idp_qb_hit), 0) as qb_hits,
                   COALESCE(SUM(w.idp_int), 0) as ints,
                   COALESCE(SUM(w.idp_pass_def), 0) as pass_def,
                   COALESCE(AVG(w.pts_allow), 22) as pts_allow,
                   COALESCE(AVG(w.yds_allow), 350) as yds_allow
            FROM players p
            JOIN weekly_stats w ON p.player_id = w.player_id
            WHERE p.team = ? AND p.position IN ('CB', 'S', 'SS', 'FS', 'LB', 'DE', 'DT', 'DB', 'DEF')
            AND w.season = 2024 AND w.week >= ? AND w.week < ?
            AND w.gp > 0
            GROUP BY w.week
            ORDER BY w.week DESC
            LIMIT 3
        """
        
        results = conn.execute(query, (opponent_team, max(1, week-3), week)).fetchall()
        
        weekly_def_data = []
        for result in results:
            weekly_def_data.append({
                'week': result[0],
                'sacks': result[1] or 0,
                'qb_hits': result[2] or 0,
                'ints': result[3] or 0,
                'pass_def': result[4] or 0,
                'pts_allow': result[5] or 22,
                'yds_allow': result[6] or 350
            })
        
        return weekly_def_data
    except Exception as e:
        print(f"Error getting opponent weekly defensive stats: {e}")
        return []

def get_dst_weekly_stats(conn, opponent_team, week):
    """Get opponent's DST weekly stats (fantasy points allowed by position)"""
    try:
        # Since Sleeper doesn't have fan_pts_allow_* columns, we'll calculate estimates
        # based on actual points allowed and team defensive performance
        query = """
            SELECT w.week,
                   COALESCE(AVG(w.pts_allow), 22) as pts_allow,
                   COALESCE(AVG(w.yds_allow), 350) as yds_allow,
                   COALESCE(SUM(w.idp_sack), 0) as sacks,
                   COALESCE(SUM(w.idp_int), 0) as ints
            FROM players p
            JOIN weekly_stats w ON p.player_id = w.player_id
            WHERE p.team = ? AND p.position = 'DEF'
            AND w.season = 2024 AND w.week >= ? AND w.week < ?
            AND w.gp > 0
            GROUP BY w.week
            ORDER BY w.week DESC
            LIMIT 3
        """
        
        results = conn.execute(query, (opponent_team, max(1, week-3), week)).fetchall()
        
        weekly_dst_data = []
        for result in results:
            pts_allow = result[1] or 22
            yds_allow = result[2] or 350
            sacks = result[3] or 0
            ints = result[4] or 0
            
            # Estimate fantasy points allowed by position based on defensive performance
            # Better defense = lower points allowed
            defense_strength = min(1.0, (sacks + ints * 2) / 10.0)  # 0-1 scale
            pts_modifier = 1.0 + (defense_strength * 0.3)  # Strong defense allows 30% more resistance
            
            weekly_dst_data.append({
                'week': result[0],
                'fan_pts_allow_qb': max(12, 18 * pts_modifier),
                'fan_pts_allow_rb': max(8, 12 * pts_modifier), 
                'fan_pts_allow_wr': max(15, 25 * pts_modifier),
                'fan_pts_allow_te': max(6, 10 * pts_modifier),
                'fan_pts_allow_k': max(5, 8 * pts_modifier),
                'pts_allow': pts_allow,
                'yds_allow': yds_allow
            })
        
        return weekly_dst_data
    except Exception as e:
        print(f"Error getting DST weekly stats: {e}")
        return []

def generate_sleeper_native_analysis(player_data, prediction):
    """Generate analysis based on actual Sleeper data"""
    analysis = []
    position = player_data.get('position')
    player_name = player_data.get('name', '')
    opponent = player_data.get('opponent', '')
    
    # Position-specific analysis
    if position == 'QB':
        if prediction > 20:
            analysis.append(f"High-end QB1 projection with strong passing upside")
        elif prediction > 16:
            analysis.append(f"Solid QB1 option with good floor and ceiling")
        else:
            analysis.append(f"Streaming QB option in favorable matchups")
    
    elif position == 'RB':
        if prediction > 15:
            analysis.append(f"RB1 projection with strong workload expectations")
        elif prediction > 10:
            analysis.append(f"Reliable RB2 option with decent floor")
        else:
            analysis.append(f"Flex consideration in deeper leagues")
    
    elif position == 'WR':
        if prediction > 13:
            analysis.append(f"WR1 upside with target volume and red zone opportunities")
        elif prediction > 9:
            analysis.append(f"Solid WR2/Flex option with consistent target share")
        else:
            analysis.append(f"Deep league option dependent on game script")
    
    elif position == 'TE':
        if prediction > 10:
            analysis.append(f"Premium TE1 with consistent target share and red zone looks")
        elif prediction > 7:
            analysis.append(f"Streaming TE option with matchup-dependent upside")
        else:
            analysis.append(f"Waiver wire TE in deep leagues only")
    
    elif position == 'K':
        if prediction > 9:
            analysis.append(f"Top-tier kicker with strong offense and dome/good weather")
        elif prediction > 7:
            analysis.append(f"Reliable kicking option with decent offensive support")
        else:
            analysis.append(f"Streaming kicker dependent on game conditions")
    
    elif position in ['DEF', 'DST']:
        if prediction > 12:
            analysis.append(f"Elite DST with strong defensive performance and turnover potential")
        elif prediction > 8:
            analysis.append(f"Solid DST option with consistent floor")
        else:
            analysis.append(f"Risky DST play dependent on matchup and game script")
    
    elif position in ['DB', 'CB', 'S', 'SS', 'FS']:
        if prediction > 8:
            analysis.append(f"High-producing DB with tackle and interception upside")
        elif prediction > 5:
            analysis.append(f"Solid IDP option with consistent tackle production")
        else:
            analysis.append(f"Boom-or-bust DB dependent on big plays")
    
    elif position in ['LB', 'OLB', 'MLB', 'ILB']:
        if prediction > 10:
            analysis.append(f"Elite LB with high tackle volume and sack potential")
        elif prediction > 7:
            analysis.append(f"Reliable LB option with good tackle floor")
        else:
            analysis.append(f"Situational LB play in favorable matchups")
    
    elif position in ['DL', 'DE', 'DT']:
        if prediction > 8:
            analysis.append(f"Premium pass rusher with sack and tackle upside")
        elif prediction > 5:
            analysis.append(f"Solid DL option with pressure and tackle potential")
        else:
            analysis.append(f"Boom-or-bust pass rusher dependent on sacks")
    
    # Add opponent analysis
    analysis.append(f"Matchup vs {opponent} factored into projection")
    
    # Add general analysis based on prediction range
    if prediction > 20:
        analysis.append("High-confidence projection based on strong recent performance and favorable matchup")
    elif prediction > 15:
        analysis.append("Solid projection with good floor based on consistent usage")
    elif prediction > 10:
        analysis.append("Moderate projection - some risk factors present")
    else:
        analysis.append("Low projection due to challenging matchup or inconsistent usage")
    
    return analysis

# Backwards compatibility functions
def prepare_enhanced_features(player_data):
    """Wrapper for backwards compatibility"""
    return prepare_sleeper_native_features(player_data, "sleeper_enhanced.db")

def generate_enhanced_analysis(player_data, prediction):
    """Wrapper for backwards compatibility"""
    return generate_sleeper_native_analysis(player_data, prediction)