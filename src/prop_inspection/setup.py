from setuptools import setup
import os
from glob import glob

package_name = 'prop_inspection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

        (os.path.join('share', package_name, 'models', 'good_prop'),
            glob('models/good_prop/*.config') + glob('models/good_prop/*.sdf')),
        (os.path.join('share', package_name, 'models', 'good_prop', 'meshes'),
            glob('models/good_prop/meshes/*')),

        (os.path.join('share', package_name, 'models', 'warped_prop'),
            glob('models/warped_prop/*.config') + glob('models/warped_prop/*.sdf')),
        (os.path.join('share', package_name, 'models', 'warped_prop', 'meshes'),
            glob('models/warped_prop/meshes/*')),

        (os.path.join('share', package_name, 'models', 'incomplete_prop'),
            glob('models/incomplete_prop/*.config') + glob('models/incomplete_prop/*.sdf')),
        (os.path.join('share', package_name, 'models', 'incomplete_prop', 'meshes'),
            glob('models/incomplete_prop/meshes/*')),

        (os.path.join('share', package_name, 'models', 'sinkage_prop'),
            glob('models/sinkage_prop/*.config') + glob('models/sinkage_prop/*.sdf')),
        (os.path.join('share', package_name, 'models', 'sinkage_prop', 'meshes'),
            glob('models/sinkage_prop/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adam',
    maintainer_email='adelcoll@umd.edu',
    description='ROS 2 + Gazebo propeller inspection project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spawn_grid = prop_inspection.spawn_grid:main',
            'inspect_camera = prop_inspection.inspect_camera:main',
            'log_results = prop_inspection.log_results:main',
        ],
    },
)
