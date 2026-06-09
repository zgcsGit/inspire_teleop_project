from setuptools import setup
import os
from glob import glob

package_name = 'inspire_launch'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # 安装 package.xml
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装 launch 文件夹下所有文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tp2',
    maintainer_email='tp2@example.com',
    description='Launch package for Inspire nodes',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
        ],
    },
)
