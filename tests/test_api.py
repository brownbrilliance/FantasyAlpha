"""
Test suite for Fantasy Football AI API endpoints
"""

import pytest
import json
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask application"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_predict_endpoint(client):
    """Test the player prediction endpoint"""
    test_data = {
        "name": "Josh Allen",
        "position": "QB",
        "team": "BUF",
        "opponent": "MIA",
        "homeAway": "HOME",
        "week": "8",
        "avgPoints": 22.5,
        "vegasTotal": 48.5
    }
    
    response = client.post('/api/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'analysis' in data
    assert 'weather' in data
    assert 'confidence' in data
    
    # Validate prediction is reasonable
    prediction = data['prediction']
    assert isinstance(prediction, (int, float))
    assert 0 <= prediction <= 50
    
    # Validate analysis is a list
    assert isinstance(data['analysis'], list)
    assert len(data['analysis']) > 0
    
    # Validate weather data
    weather = data['weather']
    assert 'temperature' in weather
    assert 'wind_speed' in weather
    assert 'conditions' in weather
    assert 'fantasy_impact' in weather

def test_predict_missing_data(client):
    """Test prediction endpoint with missing required data"""
    test_data = {
        "name": "Josh Allen",
        # Missing required fields
    }
    
    response = client.post('/api/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    # Should still work but with default values
    assert response.status_code == 200

def test_rankings_endpoint(client):
    """Test the rankings generation endpoint"""
    test_data = {
        "position": "QB",
        "week": "8"
    }
    
    response = client.post('/api/rankings',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'rankings' in data
    
    rankings = data['rankings']
    assert isinstance(rankings, list)
    assert len(rankings) > 0
    
    # Validate ranking structure
    for player in rankings:
        assert 'name' in player
        assert 'team' in player
        assert 'opponent' in player
        assert 'points' in player
        assert isinstance(player['points'], (int, float))

def test_rankings_different_positions(client):
    """Test rankings for different positions"""
    positions = ['QB', 'RB', 'WR', 'TE']
    
    for position in positions:
        test_data = {
            "position": position,
            "week": "8"
        }
        
        response = client.post('/api/rankings',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['rankings']) > 0

def test_compare_endpoint(client):
    """Test the player comparison endpoint"""
    test_data = {
        "player1": {
            "name": "Josh Allen",
            "position": "QB",
            "team": "BUF"
        },
        "player2": {
            "name": "Lamar Jackson",
            "position": "QB",
            "team": "BAL"
        }
    }
    
    response = client.post('/api/compare',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'player1' in data
    assert 'player2' in data
    
    # Validate player1 data
    p1 = data['player1']
    assert 'name' in p1
    assert 'prediction' in p1
    assert 'analysis' in p1
    assert isinstance(p1['prediction'], (int, float))
    
    # Validate player2 data
    p2 = data['player2']
    assert 'name' in p2
    assert 'prediction' in p2
    assert 'analysis' in p2
    assert isinstance(p2['prediction'], (int, float))

def test_compare_missing_player(client):
    """Test comparison with missing player data"""
    test_data = {
        "player1": {
            "name": "Josh Allen",
            "position": "QB",
            "team": "BUF"
        }
        # Missing player2
    }
    
    response = client.post('/api/compare',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    # Should return an error
    assert response.status_code == 500

def test_invalid_json(client):
    """Test endpoints with invalid JSON"""
    response = client.post('/api/predict',
                          data="invalid json",
                          content_type='application/json')
    
    assert response.status_code == 500

def test_home_page(client):
    """Test that the home page loads (if templates/index.html exists)"""
    response = client.get('/')
    # This might return 404 if template doesn't exist, which is fine
    assert response.status_code in [200, 404]

if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])
