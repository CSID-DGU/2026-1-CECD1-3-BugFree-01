from __future__ import annotations

SAMPLE_RATE = 16000
MODEL_SAMPLE_RATE = 32000
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
MODEL_INPUT_SECONDS = 10
MODEL_INPUT_SAMPLES = MODEL_SAMPLE_RATE * MODEL_INPUT_SECONDS
REQUIRED_INPUT_CHANNELS = 6
MIC_CHANNEL_INDEX = 0
MODEL_NAME = "mn10_as"
AUDIOSET_CLASS_COUNT = 527
DB_EPSILON = 1e-12

DEFAULT_MIN_DB = 30.0
DEFAULT_ENHANCE_THRESHOLD_DB = 35.0
DEFAULT_NOISE_REDUCTION_DB = 18.0
DEFAULT_MAIN_GAIN_DB = 8.0
DEFAULT_ENHANCE_SHARPNESS = 2.0
DEFAULT_MIN_SCORE = 0.05

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"

# EfficientAT AudioSet pretrained models use the official 32 kHz frontend.
N_MELS = 128
WINDOW_SIZE = 800
HOP_SIZE = 320
N_FFT = 1024
FMAX = MODEL_SAMPLE_RATE // 2 - 1000

MIC_NAME_KEYWORDS = ("respeaker", "re speaker", "seeed", "array v3")

LABEL_MAPPING = {
    "construction": ["Tools", "Power tool", "Jackhammer", "Drill", "Chainsaw", "Hammer", "Sawing"],
    "gunshot": ["Gunshot, gunfire"],
    "alarm_siren": ["Siren", "Alarm", "Alarm clock"],
    "horn": ["Vehicle horn, car horn, honking"],
    "water": [
        "Water",
        "Rain",
        "Raindrop",
        "Rain on surface",
        "Stream",
        "Waterfall",
        "Gurgling",
        "Water tap, faucet",
        "Sink (filling or washing)",
        "Liquid",
        "Splash, splatter",
        "Pour",
    ],
    "knock": ["Knock"],
    "appliances": ["Vacuum cleaner"],
    "baby_cry": ["Baby cry, infant cry"],
    "animal_cry": ["Dog", "Cat", "Caterwaul"],
    "glass_shatter": ["Glass", "Shatter"],
}
