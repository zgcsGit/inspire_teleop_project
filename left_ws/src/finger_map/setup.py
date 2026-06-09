from setuptools import find_packages, setup
from glob import glob

package_name = 'finger_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tp2',
    maintainer_email='kejia.chen@tum.de',
    description='Map retargeted Inspire hand joint states to hand angle commands',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'finger_mapper_node = finger_map.finger_mapper_node:main'
        ],
    },
)
