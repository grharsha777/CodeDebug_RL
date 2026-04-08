from solution import get_config

def test_config():
    config = get_config()
    assert config["host"] == "localhost"
    assert config["port"] == 8080
    assert config["debug"] is False
