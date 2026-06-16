from __future__ import annotations

BLUEZ_SERVICE_NAME = "org.bluez"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
ADAPTER_IFACE = "org.bluez.Adapter1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"

APP_PATH = "/com/bugless/bleinference"
ADVERTISEMENT_PATH = "/com/bugless/bleinference/advertisement0"

# EdgeAudioRecognition / Flutter scanner compatible UUIDs.
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
INFERENCE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
