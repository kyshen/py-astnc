class ASTNCError(Exception):
    """Base exception for the package."""


class InvalidMethodError(ASTNCError):
    """Raised when an unknown materialization method is requested."""


class InvalidWorkpointError(ASTNCError):
    """Raised when an unknown workpoint is requested."""

