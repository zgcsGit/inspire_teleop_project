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
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zz',
    maintainer_email='zz@todo.todo',
    description='Python Modbus interface for Inspire hand',
    license='MIT',
    entry_points={
        'console_scripts': [
            
            'inspire_hand_modbus_topic = inspire_hand_modbus.inspire_hand_modbus_topic:main',
        ],
    },
)
