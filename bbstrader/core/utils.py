import configparser
import importlib
import importlib.util
import os
from typing import Any, Dict, List

__all__ = ["load_module", "load_class"]


def load_module(file_path):
    """Load a module from a file path.

    Args:
        file_path: Path to the file to load.

    Returns:
        The loaded module.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Strategy file {file_path} not found. Please create it."
        )
    spec = importlib.util.spec_from_file_location("bbstrader.cli", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_class(module, class_name, base_class):
    """Load a class from a module.

    Args:
        module: The module to load the class from.
        class_name: The name of the class to load.
        base_class: The base class that the class must inherit from.
    """
    if not hasattr(module, class_name):
        raise AttributeError(f"{class_name} not found in {module}")
    class_ = getattr(module, class_name)
    if not issubclass(class_, base_class):
        raise TypeError(f"{class_name} must inherit from {base_class}.")
    return class_


def auto_convert(value):
    """Convert string values to appropriate data types"""
    if value.lower() in {"true", "false"}:  # Boolean
        return value.lower() == "true"
    elif value.lower() in {"none", "null"}:  # None
        return None
    elif value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def dict_from_ini(file_path, sections: str | List[str] = None) -> Dict[str, Any]:
    """Reads an INI file and converts it to a dictionary with proper data types.

    Args:
        file_path: Path to the INI file to read.
        sections: Optional list of sections to read from the INI file.

    Returns:
        A dictionary containing the INI file contents with proper data types.
    """
    try:
        config = configparser.ConfigParser(interpolation=None)
        config.read(file_path)
    except Exception:
        raise
    ini_dict = {}
    for section in config.sections():
        ini_dict[section] = {
            key: auto_convert(value) for key, value in config.items(section)
        }

    if isinstance(sections, str):
        try:
            return ini_dict[sections]
        except KeyError:
            raise KeyError(f"{sections} not found in the {file_path} file")
    if isinstance(sections, list):
        sect_dict = {}
        for section in sections:
            try:
                sect_dict[section] = ini_dict[section]
            except KeyError:
                raise KeyError(f"{section} not found in the {file_path} file")
    return ini_dict
