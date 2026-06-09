from setuptools import find_packages, setup

package_name = 'inspire_policy_deploy'

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
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_sub_node = inspire_policy_deploy.image_sub_node:main',
            'obs_monitor_node = inspire_policy_deploy.obs_monitor_node:main',
            'policy_ros_node = inspire_policy_deploy.policy_ros_node:main',
            'dataset_obs_replay_node = inspire_policy_deploy.dataset_obs_replay_node:main',
            'raw_episode_pose_replay_node = inspire_policy_deploy.raw_episode_pose_replay_node:main',
            'episode_init_pose_node = inspire_policy_deploy.episode_init_pose_node:main',
            'debug_action_listener = inspire_policy_deploy.debug_action_listener:main',
        ],
    },
)
