from setuptools import setup
import os
from glob import glob

package_name = 'inspire_hand_modbus'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],   
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('lib', package_name), glob('scripts/*.py')),
    ],
    install_requires=['setuptools', 'rclpy', 'std_msgs', 'pymodbus'],
    zip_safe=True,
    maintainer='tp2',
    maintainer_email='kejia.chen@tum.de',
    description='Python Modbus interface for Inspire hand',
    entry_points={
        'console_scripts': [
            
            'inspire_hand_modbus_topic = inspire_hand_modbus.inspire_hand_modbus_topic:main',
        ],
    },
)
