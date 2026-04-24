from setuptools import setup
import os
from glob import glob

package_name = 'uarm'

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
    maintainer='zyw',
    maintainer_email='zyw@todo.todo',
    description='The UArm package for ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cam_pub = uarm.cam_pub:main',
            'episode_recorder = uarm.episode_recorder:main',
            'xarm_pub = uarm.xarm_pub:main',
            'servo2xarm = uarm.servo2xarm:main',
            'servo2dobot = uarm.servo2dobot:main',
            'servo_reader = uarm.servo_reader:main',
            'servo_reader_fixed = uarm.servo_reader_fixed:main',
        ],
    },
) 