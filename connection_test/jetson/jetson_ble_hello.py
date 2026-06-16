#!/usr/bin/env python3
"""
Advertise the Jetson as a BLE GATT peripheral and send a UTF-8 message.

Test with Android/iOS BLE scanner apps such as nRF Connect:
  1. Run: sudo python3 jetson_ble_hello.py
  2. Connect to the device named JHello.
  3. Read or subscribe to the hello characteristic.
"""

import argparse
import signal
import sys

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib


BLUEZ_SERVICE_NAME = "org.bluez"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
ADAPTER_IFACE = "org.bluez.Adapter1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"

APP_PATH = "/com/bugless/blehello"
ADVERTISEMENT_PATH = "/com/bugless/blehello/advertisement0"
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
HELLO_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


def byte_array(data):
    return dbus.Array([dbus.Byte(b) for b in data], signature="y")


class Application(dbus.service.Object):
    def __init__(self, bus):
        self.path = APP_PATH
        self.services = []
        super().__init__(bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):
        managed_objects = {}
        for service in self.services:
            managed_objects[service.get_path()] = service.get_properties()
            for characteristic in service.characteristics:
                managed_objects[characteristic.get_path()] = characteristic.get_properties()
        return managed_objects


class Service(dbus.service.Object):
    def __init__(self, bus, index, uuid, primary=True):
        self.path = f"{APP_PATH}/service{index}"
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        super().__init__(bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": dbus.Boolean(self.primary),
            }
        }

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_SERVICE_IFACE]


class HelloCharacteristic(dbus.service.Object):
    def __init__(self, bus, index, service, message, repeat_seconds):
        self.path = f"{service.path}/char{index}"
        self.bus = bus
        self.service = service
        self.uuid = HELLO_CHAR_UUID
        self.flags = ["read", "notify"]
        self.message = message
        self.repeat_seconds = repeat_seconds
        self.notify_count = 0
        self.notifying = False
        self.timer_id = None
        super().__init__(bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def get_properties(self):
        return {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": dbus.Array(self.flags, signature="s"),
            }
        }

    def encoded_message(self, message=None):
        return (message if message is not None else self.message).encode("utf-8")

    def send_notification(self):
        self.notify_count += 1
        message = f"{self.message} {self.notify_count}"
        value = byte_array(self.encoded_message(message))
        self.PropertiesChanged(
            GATT_CHRC_IFACE,
            dbus.Dictionary({"Value": value}, signature="sv"),
            dbus.Array([], signature="s"),
        )
        print(f"sent notification: {message}", flush=True)

    def notify_tick(self):
        if not self.notifying:
            return False
        self.send_notification()
        return True

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        print(f"read request: {self.message}", flush=True)
        return byte_array(self.encoded_message())

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):
        if self.notifying:
            return
        self.notifying = True
        self.notify_count = 0
        print("client subscribed; sending hello", flush=True)
        self.send_notification()
        if self.repeat_seconds > 0:
            self.timer_id = GLib.timeout_add_seconds(self.repeat_seconds, self.notify_tick)

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):
        self.notifying = False
        if self.timer_id is not None:
            GLib.source_remove(self.timer_id)
            self.timer_id = None
        print("client unsubscribed", flush=True)

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, interface, changed, invalidated):
        pass


class Advertisement(dbus.service.Object):
    def __init__(self, bus, local_name, service_uuid):
        self.path = ADVERTISEMENT_PATH
        self.bus = bus
        self.local_name = local_name
        self.service_uuid = service_uuid
        super().__init__(bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def get_properties(self):
        return {
            LE_ADVERTISEMENT_IFACE: {
                "Type": "peripheral",
                "LocalName": dbus.String(self.local_name),
                "ServiceUUIDs": dbus.Array([self.service_uuid], signature="s"),
            }
        }

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE)
    def Release(self):
        print("advertisement released", flush=True)


def find_adapter(bus):
    object_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, "/"),
        DBUS_OM_IFACE,
    )
    objects = object_manager.GetManagedObjects()
    for path, interfaces in objects.items():
        if GATT_MANAGER_IFACE in interfaces and LE_ADVERTISING_MANAGER_IFACE in interfaces:
            return path
    return None


def set_adapter_powered(bus, adapter_path):
    properties = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
        DBUS_PROP_IFACE,
    )
    properties.Set(ADAPTER_IFACE, "Powered", dbus.Boolean(True))


def fail(loop, prefix):
    def handler(error):
        print(f"{prefix}: {error}", file=sys.stderr, flush=True)
        if "InvalidLength" in str(error):
            print("Try a shorter --name value, for example --name JH.", file=sys.stderr)
        loop.quit()

    return handler


def main():
    parser = argparse.ArgumentParser(description="Send a BLE hello message from Jetson.")
    parser.add_argument("--name", default="JHello", help="BLE advertising name.")
    parser.add_argument("--message", default="\uc548\ub155", help="UTF-8 message to send.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Repeat notification every N seconds after subscribe. 0 sends once.",
    )
    args = parser.parse_args()

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    adapter_path = find_adapter(bus)
    if not adapter_path:
        print(
            "No Bluetooth adapter with BLE GATT/advertising support was found.",
            file=sys.stderr,
        )
        return 1

    set_adapter_powered(bus, adapter_path)

    service_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
        GATT_MANAGER_IFACE,
    )
    advertising_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
        LE_ADVERTISING_MANAGER_IFACE,
    )

    app = Application(bus)
    hello_service = Service(bus, 0, SERVICE_UUID, primary=True)
    hello_characteristic = HelloCharacteristic(
        bus,
        0,
        hello_service,
        args.message,
        args.repeat,
    )
    hello_service.add_characteristic(hello_characteristic)
    app.add_service(hello_service)
    advertisement = Advertisement(bus, args.name, SERVICE_UUID)

    loop = GLib.MainLoop()

    def app_registered():
        print("GATT application registered", flush=True)

    def advertisement_registered():
        print(f"Advertising as {args.name}", flush=True)
        print(f"Service: {SERVICE_UUID}", flush=True)
        print(f"Characteristic: {HELLO_CHAR_UUID}", flush=True)
        print("Connect with nRF Connect, then read or subscribe to the characteristic.")

    service_manager.RegisterApplication(
        app.get_path(),
        {},
        reply_handler=app_registered,
        error_handler=fail(loop, "Failed to register GATT application"),
    )
    advertising_manager.RegisterAdvertisement(
        advertisement.get_path(),
        {},
        reply_handler=advertisement_registered,
        error_handler=fail(loop, "Failed to register advertisement"),
    )

    def stop(_signum, _frame):
        loop.quit()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        loop.run()
    finally:
        try:
            advertising_manager.UnregisterAdvertisement(advertisement.get_path())
        except dbus.exceptions.DBusException:
            pass
        try:
            service_manager.UnregisterApplication(app.get_path())
        except dbus.exceptions.DBusException:
            pass
        print("Stopped BLE hello server", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
