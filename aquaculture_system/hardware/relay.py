# hardware/relay.py
import time
import queue
import threading
import requests
import config # Import our config file directly

class ExternalRelayWorker:
    """Manages a dedicated thread for the Feeder Relay hardware using Cup portions."""
    def __init__(self, url):
        self.base_url = url
        self.queue = queue.Queue()
        self.thread = None
        self.is_running = False

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def trigger_feeding(self, num_cups, callback_event=None):
        self.queue.put({"cups": num_cups, "event": callback_event})

    def _set_hardware_state(self, state):
        if config.DRY_RUN:
            print(f"🚫 [DRY RUN ACTIVE] Bypassing physical API command. (Feeder would turn {state.upper()})")
            return True
        try:
            r = requests.get(f"{self.base_url}/{state}", timeout=5)
            return r.status_code == 200
        except requests.RequestException as e:
            print(f"💥 [Relay Thread] Tailscale network drop: {e}")
            return False

    def _worker_loop(self):
        while self.is_running:
            try:
                job = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            cups = job["cups"]
            runtime = cups * config.SECONDS_PER_CUP
            
            if runtime > 0:
                print(f"🐟 [Relay Thread] FEEDER ACTIVATED: Dispensing {cups} cup(s) ({runtime}s)...")
                self._set_hardware_state("on")
                time.sleep(runtime)
                self._set_hardware_state("off")
                print(f"✔ [Relay Thread] FEEDER DEACTIVATED: Portion complete.")
            
            if job["event"]:
                job["event"].set()
                
            self.queue.task_done()