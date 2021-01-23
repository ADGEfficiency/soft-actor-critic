from setuptools import setup, find_packages


setup(
    name='sac',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['Click'],
    entry_points={
        'console_scripts': [
            'sac=sac.main:cli'
        ],
    }
)
