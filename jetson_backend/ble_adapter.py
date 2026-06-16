from __future__ import annotations

import dbus

from ble_constants import (
    ADAPTER_IFACE,
    BLUEZ_SERVICE_NAME,
    DBUS_OM_IFACE,
    DBUS_PROP_IFACE,
    GATT_MANAGER_IFACE,
    LE_ADVERTISING_MANAGER_IFACE,
)


def find_adapter(bus: dbus.SystemBus) -> str | None:
    object_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
    objects = object_manager.GetManagedObjects()
    for path, interfaces in objects.items():
        if GATT_MANAGER_IFACE in interfaces and LE_ADVERTISING_MANAGER_IFACE in interfaces:
            return path
    return None


def set_adapter_powered(bus: dbus.SystemBus, adapter_path: str) -> None:
    properties = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), DBUS_PROP_IFACE)
    properties.Set(ADAPTER_IFACE, "Powered", dbus.Boolean(True))
