"""This checkout's defaults from an optional local_defaults.py beside the engine: the checkpoint and MTP head (None without one) and two opt-ins that are off without one."""

try:
    import local_defaults as _local
except ImportError:
    _local = None

MODEL = getattr(_local, "MODEL", None)
MTP = getattr(_local, "MTP", None)
# The launcher lists the other computers (presets, ~/.ssh/config, Tailscale) and checks each over ssh at start (./larder --devices).
DEVICES = bool(getattr(_local, "DEVICES", False))
# The model scan also reads the top of every other mounted volume, which can raise macOS's removable-volume prompt (./larder --volumes).
VOLUMES = bool(getattr(_local, "VOLUMES", False))
