from __future__ import annotations

import json
import sys
import threading

import dbus
import dbus.exceptions
import dbus.mainloop.glib
from gi.repository import GLib

from ble_adapter import find_adapter, set_adapter_powered
from ble_constants import (
    BLUEZ_SERVICE_NAME,
    GATT_MANAGER_IFACE,
    INFERENCE_CHAR_UUID,
    LE_ADVERTISING_MANAGER_IFACE,
    SERVICE_UUID,
)
from ble_gatt_objects import Advertisement, Application, InferenceCharacteristic, Service


class BleInferenceServer:
    def __init__(self, local_name: str, chunk_bytes: int):
        self.local_name = local_name
        self.chunk_bytes = chunk_bytes
        self.loop: GLib.MainLoop | None = None
        self.thread: threading.Thread | None = None
        self.service_manager = None
        self.advertising_manager = None
        self.application: Application | None = None
        self.advertisement: Advertisement | None = None
        self.characteristic: InferenceCharacteristic | None = None
        self.ready_event = threading.Event()
        self.startup_error: str | None = None
        self.app_ready = False
        self.advertisement_ready = False

    def start(self) -> None:
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SystemBus()
        adapter_path = find_adapter(bus)
        if not adapter_path:
            raise RuntimeError("No Bluetooth adapter with BLE GATT/advertising support was found.")

        set_adapter_powered(bus, adapter_path)
        self.service_manager = dbus.Interface(
            bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
            GATT_MANAGER_IFACE,
        )
        self.advertising_manager = dbus.Interface(
            bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
            LE_ADVERTISING_MANAGER_IFACE,
        )

        self.application = Application(bus)
        inference_service = Service(bus, 0, SERVICE_UUID, primary=True)
        self.characteristic = InferenceCharacteristic(bus, 0, inference_service, self.chunk_bytes)
        inference_service.add_characteristic(self.characteristic)
        self.application.add_service(inference_service)
        self.advertisement = Advertisement(bus, self.local_name, SERVICE_UUID)
        self.loop = GLib.MainLoop()

        self.service_manager.RegisterApplication(
            self.application.get_path(),
            {},
            reply_handler=self._app_registered,
            error_handler=self._registration_failed("Failed to register GATT application"),
        )
        self.advertising_manager.RegisterAdvertisement(
            self.advertisement.get_path(),
            {},
            reply_handler=self._advertisement_registered,
            error_handler=self._registration_failed("Failed to register advertisement"),
        )

        self.thread = threading.Thread(target=self._run_loop, name="ble-gatt-loop", daemon=True)
        self.thread.start()
        if not self.ready_event.wait(timeout=10):
            self.stop()
            raise RuntimeError("Timed out while registering BLE GATT application.")
        if self.startup_error:
            self.stop()
            raise RuntimeError(self.startup_error)

    def _app_registered(self) -> None:
        self.app_ready = True
        print("GATT application registered", flush=True)
        self._mark_ready()

    def _advertisement_registered(self) -> None:
        self.advertisement_ready = True
        print(f"Advertising as {self.local_name}", flush=True)
        print(f"Service: {SERVICE_UUID}", flush=True)
        print(f"Characteristic: {INFERENCE_CHAR_UUID}", flush=True)
        print("Connect from the Flutter app and subscribe to notifications.", flush=True)
        self._mark_ready()

    def _registration_failed(self, prefix: str):
        def handler(error: dbus.exceptions.DBusException) -> None:
            self.startup_error = f"{prefix}: {error}"
            print(self.startup_error, file=sys.stderr, flush=True)
            if "InvalidLength" in str(error):
                print("Try a shorter --ble-name value, for example --ble-name JH.", file=sys.stderr)
            self.ready_event.set()
            if self.loop is not None:
                self.loop.quit()

        return handler

    def _mark_ready(self) -> None:
        if self.app_ready and self.advertisement_ready:
            self.ready_event.set()

    def _run_loop(self) -> None:
        if self.loop is None:
            return
        try:
            self.loop.run()
        finally:
            if self.advertising_manager and self.advertisement:
                try:
                    self.advertising_manager.UnregisterAdvertisement(self.advertisement.get_path())
                except dbus.exceptions.DBusException:
                    pass
            if self.service_manager and self.application:
                try:
                    self.service_manager.UnregisterApplication(self.application.get_path())
                except dbus.exceptions.DBusException:
                    pass
            print("Stopped BLE inference server", flush=True)

    def publish(self, data: dict) -> None:
        if self.characteristic is None:
            return
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
        self.characteristic.notify_text(payload)

    def stop(self) -> None:
        if self.loop is not None:
            GLib.idle_add(self.loop.quit)
        if self.thread is not None:
            self.thread.join(timeout=3)
