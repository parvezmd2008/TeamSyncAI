import requests
import json

# --- TARGET URL: YOUR MACHINE'S IP + PORT ---
YOUR_API_BASE_URL = "http://10.11.55.152:8000"

def send_health_check():
    """Sends a GET request to your root endpoint."""
    url = f"{YOUR_API_BASE_URL}/"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # This confirms successful connection and response
        return response.json()
        
    except requests.exceptions.RequestException as e:
        # This will catch firewall errors or connection refusals
        print(f"Error connecting to your API: {e}")
        return None

# Execute the call
status = send_health_check()
print(f"API Status: {status}")