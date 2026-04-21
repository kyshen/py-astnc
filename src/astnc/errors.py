class ASTNCError(Exception):
    """Base exception for the package."""


class InvalidWorkpointError(ASTNCError):
    """Raised when an unknown workpoint is requested."""
