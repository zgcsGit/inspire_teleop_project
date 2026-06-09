from setuptools import find_packages, setup

package_name = 'inspire_retargeting'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        ],
    zip_safe=True,
    maintainer='tp2',
    maintainer_email='kejia.chen@tum.de',
    description='Inspire hand retargeting from hand keypoints to joint states',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        "detect_from_csv = inspire_retargeting.detect_from_csv:main",
        'pub_csv = inspire_retargeting.pub_csv:main',
        ],
    },
)
