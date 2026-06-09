from setuptools import find_packages, setup

package_name = 'foot_switch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tp2',
    maintainer_email='kejia.chen@tum.de',
    description='Foot pedal control node for teleoperation bringup and recording services.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'footswitch_node = foot_switch.footswitch_node:main',
        ],
    },
)
