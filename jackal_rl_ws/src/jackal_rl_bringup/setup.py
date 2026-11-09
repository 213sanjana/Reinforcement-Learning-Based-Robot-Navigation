from setuptools import setup
from glob import glob

package_name = 'jackal_rl_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Project Maintainer',
    maintainer_email='maintainer@example.com',
    description='Jackal RL Gazebo bringup',
    license='Apache-2.0',
)
