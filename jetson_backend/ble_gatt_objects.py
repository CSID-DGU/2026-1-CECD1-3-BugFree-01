from __future__ import annotations

import json
from typing import List

import dbus
import dbus.service
from gi.repository import GLib

from ble_constants import (
    ADVERTISEMENT_PATH,
    APP_PATH,
    DBUS_OM_IFACE,
    DBUS_PROP_IFACE,
    GATT_CHRC_IFACE,
    GATT_SERVICE_IFACE,
    INFERENCE_CHAR_UUID,
    LE_ADVERTISEMENT_IFACE,
)
from ble_utils import InvalidArgsException, byte_array


class Application(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus):
        self.path = APP_PATH
        self.services: list[Service] = []
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_service(self, service: "Service") -> None:
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self) -> dict:
        managed_objects = {}
        for service in self.services:
            managed_objects[service.get_path()] = service.get_properties()
            for characteristic in service.characteristics:
                managed_objects[characteristic.get_path()] = characteristic.get_properties()
        return managed_objects


class Service(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, index: int, uuid: str, primary: bool = True):
        self.path = f"{APP_PATH}/service{index}"
        self.uuid = uuid
        self.primary = primary
        self.characteristics: list[InferenceCharacteristic] = []
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic: "InferenceCharacteristic") -> None:
        self.characteristics.append(characteristic)

    def get_properties(self) -> dict:
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": dbus.Boolean(self.primary),
            }
        }

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict:
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_SERVICE_IFACE]


class InferenceCharacteristic(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, index: int, service: Service, chunk_bytes: int):
        self.path = f"{service.path}/char{index}"
        self.service = service
        self.uuid = INFERENCE_CHAR_UUID
        self.flags = ["read", "notify"]
        self.chunk_bytes = max(20, int(chunk_bytes))
        self.latest_payload = json.dumps(
            {"status": "starting", "message": "waiting_for_inference"},
            separators=(",", ":"),
            ensure_ascii=True,
        )
        self.notifying = False
        self.sequence = 0
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def get_properties(self) -> dict:
        return {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": dbus.Array(self.flags, signature="s"),
            }
        }

    def _frames_for_payload(self, payload: str) -> List[bytes]:
        payload_bytes = payload.encode("ascii")
        total = 1
        while True:
            parts: list[bytes] = []
            offset = 0
            index = 1
            while offset < len(payload_bytes) or (offset == 0 and not payload_bytes):
                prefix = f"#{self.sequence}:{index}/{total}:".encode("ascii")
                content_size = max(1, self.chunk_bytes - len(prefix))
                parts.append(payload_bytes[offset : offset + content_size])
                offset += content_size
                index += 1
            if len(parts) == total:
                return [
                    f"#{self.sequence}:{index}/{total}:".encode("ascii") + part
                    for index, part in enumerate(parts, start=1)
                ]
            total = len(parts)

    def notify_text(self, payload: str) -> None:
        self.latest_payload = payload
        GLib.idle_add(self._notify_latest)

    def _notify_latest(self) -> bool:
        if not self.notifying:
            return False

        self.sequence += 1
        for frame in self._frames_for_payload(self.latest_payload):
            self.PropertiesChanged(
                GATT_CHRC_IFACE,
                dbus.Dictionary({"Value": byte_array(frame)}, signature="sv"),
                dbus.Array([], signature="s"),
            )
        print(
            f"sent BLE inference notification seq={self.sequence} bytes={len(self.latest_payload)}",
            flush=True,
        )
        return False

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict:
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options: dict) -> dbus.Array:
        print("read request: latest inference payload", flush=True)
        return byte_array(self.latest_payload.encode("utf-8"))

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self) -> None:
        if self.notifying:
            return
        self.notifying = True
        print("client subscribed; sending latest inference payload", flush=True)
        self._notify_latest()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self) -> None:
        self.notifying = False
        print("client unsubscribed", flush=True)

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, interface: str, changed: dict, invalidated: list) -> None:
        pass


class Advertisement(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, local_name: str, service_uuid: str):
        self.path = ADVERTISEMENT_PATH
        self.local_name = local_name
        self.service_uuid = service_uuid
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def get_properties(self) -> dict:
        return {
            LE_ADVERTISEMENT_IFACE: {
                "Type": "peripheral",
                "LocalName": dbus.String(self.local_name),
                "ServiceUUIDs": dbus.Array([self.service_uuid], signature="s"),
            }
        }

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict:
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE)
    def Release(self) -> None:
        print("advertisement released", flush=True)
