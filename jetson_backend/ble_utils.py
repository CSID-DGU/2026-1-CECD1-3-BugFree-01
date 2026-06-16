from __future__ import annotations

import dbus
import dbus.exceptions


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


def byte_array(data: bytes) -> dbus.Array:
    return dbus.Array([dbus.Byte(byte) for byte in data], signature="y")
