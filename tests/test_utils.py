from src.utils import load_image, save_image

def test_load_image():
    result = load_image()
    assert result is None  # For now, just check if the function runs without error

def test_save_image():
    result = save_image()
    assert result is None  # For now, just check if the function runs without error
