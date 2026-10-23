"""Queued feeder control with delivery confirmation and fault handling."""

import math
import queue
import threading
import time
from concurrent.futures import Future

import requests

import config


class ExternalRelayWorker:
    """Confirm timed portions and latch faults until the controller restarts."""

    def __init__(self, url):
        self.base_url = url
        self.queue = queue.Queue()
        self.thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.faulted = False

    def start(self):
        self.stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=(config.RELAY_OFF_RETRIES + 1) * 5 + 2)
        if not self._ensure_off():
            self.faulted = True
            raise RuntimeError("Feeder OFF could not be confirmed during shutdown")

    def trigger_feeding(self, num_cups, callback_event=None):
        future = Future()
        if (
            not math.isfinite(num_cups)
            or num_cups <= 0
            or num_cups > config.MAX_SESSION_CUPS
        ):
            raise ValueError("Invalid feed portion")
        if not self.is_running or self.faulted or self.stop_event.is_set():
            future.set_exception(RuntimeError("Relay is stopped or faulted"))
            return future
        self.queue.put((num_cups, callback_event, future))
        return future

    def _set_hardware_state(self, state):
        if config.DRY_RUN:
            print(f"[DRY RUN] Feeder {state.upper()}")
            return True
        try:
            return (
                requests.get(f"{self.base_url}/{state}", timeout=5).status_code == 200
            )
        except requests.RequestException:
            return False

    def _ensure_off(self):
        return self._off_with_attempts()[0]

    def _off_with_attempts(self):
        for attempt in range(1, config.RELAY_OFF_RETRIES + 1):
            try:
                if self._set_hardware_state("off"):
                    return True, attempt
            except Exception:
                pass
        return False, config.RELAY_OFF_RETRIES

    def _worker_loop(self):
        while self.is_running or not self.queue.empty():
            try:
                cups, event, future = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._dispense_portion(cups)
                future.set_result(cups)
            except Exception as exc:
                self.faulted = True
                future.set_exception(exc)
            finally:
                if event:
                    event.set()
                self.queue.task_done()

    def _dispense_portion(self, cups):
        """Run one timed portion, always attempt OFF, and report uncertainty."""
        error = None
        attempted_on = False
        try:
            if self.stop_event.is_set() or self.faulted:
                raise RuntimeError("Relay stopped or faulted before dispensing")
            runtime = cups * config.SECONDS_PER_CUP
            if not math.isfinite(runtime) or runtime <= 0:
                raise ValueError("SECONDS_PER_CUP must be finite and positive")
            # An ON timeout is ambiguous: never retry ON or claim delivery.
            deadline = time.monotonic() + runtime
            attempted_on = True
            if not self._set_hardware_state("on"):
                raise RuntimeError("Feeder ON unconfirmed; actual delivery is unknown")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "ON acknowledgment exceeded portion deadline; actual delivery is unknown"
                )
            if self.stop_event.wait(remaining):
                raise RuntimeError("Feeding interrupted; actual delivery is unknown")
        except Exception as exc:
            error = exc
        finally:
            if attempted_on:
                off_confirmed, attempts = self._off_with_attempts()
                if not off_confirmed:
                    error = RuntimeError(
                        "Feeder OFF unconfirmed; actual delivery is unknown"
                    )
                elif attempts > 1:
                    error = RuntimeError(
                        "Feeder OFF required retries; actual delivery is unknown"
                    )
        if error:
            raise error
