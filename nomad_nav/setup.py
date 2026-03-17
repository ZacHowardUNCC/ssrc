import os
from glob import glob
from setuptools import setup, find_packages

package_name = "nomad_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "scripts"), glob("*.sh")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="charmbot",
    maintainer_email="todo@todo.com",
    description="NoMaD goal-conditioned navigation for Scout Mini (ROS2 Humble)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "navigate = nomad_nav.navigate:main",
            "pd_controller = nomad_nav.pd_controller:main",
            "joy_teleop = nomad_nav.joy_teleop:main",
            "create_topomap = nomad_nav.create_topomap:main",
            "collect_trajectory = nomad_nav.collect_trajectory:main",
            "live_viz = nomad_nav.live_viz:main",
        ],
    },
)
