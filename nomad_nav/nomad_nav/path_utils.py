import os
import sys
from typing import Iterator, Optional


def _iter_ancestors(path: str) -> Iterator[str]:
    current = os.path.abspath(path)
    while True:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def _is_visualnav_root(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "deployment", "config"))
        and os.path.isdir(os.path.join(path, "train"))
    )


def _candidate_roots(start_path: str) -> Iterator[str]:
    seen = set()
    for ancestor in _iter_ancestors(start_path):
        for candidate in (
            ancestor,
            os.path.join(ancestor, "visualnav-transformer"),
            os.path.join(ancestor, "src", "visualnav-transformer"),
        ):
            candidate = os.path.abspath(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def find_visualnav_root(start_path: Optional[str] = None) -> str:
    env_root = os.environ.get("VISUALNAV_ROOT", "").strip()
    if env_root:
        expanded = os.path.abspath(os.path.expanduser(env_root))
        if _is_visualnav_root(expanded):
            return expanded

    search_start = start_path or os.path.dirname(__file__)
    for candidate in _candidate_roots(search_start):
        if _is_visualnav_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate visualnav-transformer root. "
        "Set VISUALNAV_ROOT to your visualnav-transformer directory."
    )


def get_deployment_dir() -> str:
    return os.path.join(find_visualnav_root(), "deployment")


def get_default_robot_config_path() -> str:
    return os.path.join(get_deployment_dir(), "config", "robot.yaml")


def get_default_model_config_path() -> str:
    return os.path.join(get_deployment_dir(), "config", "models.yaml")


def get_default_joy_config_path() -> str:
    return os.path.join(get_deployment_dir(), "config", "joystick.yaml")


def get_default_topomap_images_dir() -> str:
    return os.path.join(get_deployment_dir(), "topomaps", "images")


def ensure_visualnav_python_paths() -> None:
    root = find_visualnav_root()
    for path in (
        os.path.join(root, "train"),
        os.path.join(root, "diffusion_policy"),
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
