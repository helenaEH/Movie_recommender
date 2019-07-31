import pytest
import requests

def test_recommender():
    recommend(3,4,5)
    assert type(movie) == str

def test_string_entered():
    """fails if the user enters a string instead of a number"""
    with pytest.raises(Exception):
        recommend(3, "dummy", 5)

def test_scrape():
    """scrape our own website to see whether the flask server is running"""
    url = "http://localhost:5000"
    page = requests.get(url)
    assert page.status_code == 200
    assert page.text == "hello world"