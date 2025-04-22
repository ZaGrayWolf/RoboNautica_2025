from setuptools import find_packages, setup

package_name = 'robonautic_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='souri',
    maintainer_email='vsouri2705@gmail.com',
    description='ROS2 package for ArUco marker detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco = robonautic_pkg.aruco:main',
            'PID = robonautic_pkg.PID:main'
        ],
    },
)
