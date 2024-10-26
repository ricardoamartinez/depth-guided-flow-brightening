from src.optical_flow import calculate_optical_flow

def test_calculate_optical_flow():
    result = calculate_optical_flow()
    assert result is None  # For now, just check if the function runs without error
