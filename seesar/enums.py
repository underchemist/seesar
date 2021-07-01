from enum import Enum

class NormalizationKind(Enum):
    """
    The normalization method
    """
    linear = 'linear'
    log = 'log'