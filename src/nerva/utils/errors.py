"""
NERVA-ML Exception Hierarchy
Structured errors with actionable user hints.
"""

class NervaError(Exception):
    """Base exception for all NERVA-ML runtime errors."""
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\nHINT: {self.hint}"
        return self.message

class ConfigError(NervaError):
    """Raised when configuration values are missing, invalid, or out of range."""
    pass

class DataMissingError(NervaError):
    """Raised when expected data files or directories do not exist."""
    pass

class DataFormatError(NervaError):
    """Raised when file columns or structures fail schema validation."""
    pass

class GeneMappingError(NervaError):
    """Raised when identifier normalization fails critically."""
    pass

class EmptyGeneSetError(NervaError):
    """Raised when no genes remain after universe intersection."""
    pass

class InsufficientSamplesError(NervaError):
    """Raised when sample or donor counts are too low for grouped CV."""
    pass

class ModelFailureError(NervaError):
    """Raised when an estimator fails numerical convergence or evaluation."""
    pass
