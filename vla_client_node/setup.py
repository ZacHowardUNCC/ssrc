from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vla_client_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='charml',
    maintainer_email='smit.r.patel5@gmail.com',
    description='VLA client node that publishes goal embeddings',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'instruction_to_embedding = vla_client_node.instruction_to_embedding_node:main',
        ],
    },
)