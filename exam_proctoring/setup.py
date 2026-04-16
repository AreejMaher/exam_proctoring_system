from setuptools import find_packages, setup

package_name = 'exam_proctoring'

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
    maintainer='rejo',
    maintainer_email='rejo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'face_detection = exam_proctoring.face_detection:main',
            'depth_estimator = exam_proctoring.depth_estimator:main',
            'camera_stream = exam_proctoring.camera_stream:main',
            'object_detector = exam_proctoring.object_detector:main',
            'behavior_node = exam_proctoring.behavior_node:main',
            'rule_evaluation = exam_proctoring.rule_evaluation:main',
            'system_monitor = exam_proctoring.system_monitor:main',
            'alert_node = exam_proctoring.alert_node:main',

        ],
    },
)
