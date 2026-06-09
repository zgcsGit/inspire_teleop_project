from setuptools import find_packages, setup

package_name = 'vicon_udp_receiver'

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
    description='UDP receiver for publishing Vicon hand keypoints',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'udp_receiver = vicon_udp_receiver.udp_receiver:main',
            'with_subject = vicon_udp_receiver.with_subject:main',
            'udp_lefthand_10 = vicon_udp_receiver.udp_lefthand_10:main',
        ],
    },
)
