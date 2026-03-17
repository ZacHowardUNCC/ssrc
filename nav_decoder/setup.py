from setuptools import setup

package_name = 'nav_decoder'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/closed_loop_navigation.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='spate308',
    maintainer_email='spate308@example.com',
    description='Closed-loop UniVLA navigation for Scout robot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'navigation_client = nav_decoder.navigation_client:main',
            # Keep existing entries below when merging into your Jetson package:
            # 'nav_decoder_node = nav_decoder.nav_decoder_node:main',
        ],
    },
)
