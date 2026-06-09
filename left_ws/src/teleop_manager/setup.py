from setuptools import setup

package_name = 'teleop_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fan',
    maintainer_email='1392588117@qq.com',
    description='Teleoperation bringup manager (start/stop via topic)',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bringup_manager = teleop_manager.bringup_manager:main',
        ],
    },
)
