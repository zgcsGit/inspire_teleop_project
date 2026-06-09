from setuptools import setup

package_name = 'teleop_viewer'

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
    maintainer_email='fan@example.com',
    description='Lightweight teleoperation image viewer',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'single_image_viewer = teleop_viewer.single_image_viewer:main',
        ],
    },
)
