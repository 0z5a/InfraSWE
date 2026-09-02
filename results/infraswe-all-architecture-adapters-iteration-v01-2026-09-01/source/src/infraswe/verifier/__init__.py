from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .verifier import Verifier

__all__ = ["Verifier"]


def __getattr__(name: str) -> Any:
    if name == "Verifier":
        from .verifier import Verifier

        return Verifier
    raise AttributeError(name)
