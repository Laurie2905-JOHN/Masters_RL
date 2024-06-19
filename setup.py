from setuptools import setup, find_packages

setup(
    name='Masters_RL',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=[
        'gymnasium',
        'numpy',
        'matplotlib',
        'stable-baselines3',
    ],
)
