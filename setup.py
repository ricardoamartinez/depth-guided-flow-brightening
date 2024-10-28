# setup.py

from setuptools import setup, find_packages

setup(
    name='motion_analysis_project',
    version='1.0.0',
    description='A video processing system combining optical flow analysis, depth mapping, and highlight generation.',
    author='Ricardo Alexander Martinez',
    author_email='martricardo.a@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python',
        'numpy',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'process_video=motion_analysis.scripts.process_video:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)