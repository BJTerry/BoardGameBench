from typing import Protocol, Any, Dict, List, Union


class Serializable(Protocol):
    """Protocol for objects that can be serialized to JSON-compatible dictionaries."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a JSON-serializable dictionary."""
        ...


def serialize_value(value: Any) -> Union[Dict[str, Any], List[Any], Any]:
    """Serialize a value to a JSON-compatible format."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    # For basic types (int, float, str, bool, None), return as is.
    # Assumes these are directly JSON-serializable.
    return value

# Note: A corresponding generic `deserialize_value` is generally not needed.
# Deserialization from the dictionary loaded from JSON should be handled
# by specific `from_dict` classmethods on the target state objects (e.g., AzulState.from_dict),
# as these methods have the necessary context to reconstruct complex types and enums.
