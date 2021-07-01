from enum import Enum, IntEnum


class NormalizationKind(Enum):
    """
    The normalization method
    """

    linear = "linear"
    log = "log"


class Uint8(IntEnum):
    min = 0
    max = 255


class Uint16(IntEnum):
    min = 0
    max = 65535


class Int16(IntEnum):
    min = -32767
    max = 32767
