"""
Configuration settings for Fantasy Football AI application
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # API Keys for external services
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')
    NFL_API_KEY = os.environ.get('NFL_API_KEY')
    ESPN_API_KEY = os.environ.get('ESPN_API_KEY')
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///fantasy_football.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ML Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'fantasy_football_model.pkl'
    SCALER_PATH = os.environ.get('SCALER_PATH') or 'scaler.pkl'
    FEATURE_NAMES_PATH = os.environ.get('FEATURE_NAMES_PATH') or 'feature_names.pkl'
    
    # Cache settings
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Application settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    JSON_SORT_KEYS = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment variable"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])

# NFL Teams and constants
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

POSITIONS = ['QB', 'RB', 'WR', 'TE']
