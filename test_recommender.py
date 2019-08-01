import pytest
from recommender import get_ml_recommendations
from application import app

def test_recommender():
    movie = get_ml_recommendations([['Titanic', 3], ['Inception', 4], ['Shrek',5]])
    assert type(movie) == str

def test_string_entered():
    """fails if the user enters a string instead of a number"""
    with pytest.raises(Exception):
        get_ml_recommendations([3, "dummy", 5])

@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()
    yield client

def test_empty_db(client):
    """Start with a blank database."""
    rv = client.get('/')
    assert b'movie' in rv.data

