import requests
import time

# Your exact Tailscale IP endpoint
BASE_URL = "http://100.80.73.62:1880/api/do1"

def set_digital_output(state):
    """Pass 'on' or 'off' to change the DO 1 hardware line."""
    url = f"{BASE_URL}/{state}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"DO 1 successfully turned {state.upper()}!")
        else:
            print(f"Failed. Server responded with code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Network error trying to reach IRIV Pi over Tailscale: {e}")

if __name__ == "__main__":
    # Example sequence: Turn on for 3 seconds, then turn off
    set_digital_output("on")
    time.sleep(3.0)
    set_digital_output("off")