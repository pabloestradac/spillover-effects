# https://setuptools.pypa.io/en/latest/setuptools.html

from setuptools import setup, find_packages

from spillover_effects import __version__

setup(
    name='spillover-effects',
    version=__version__,

    url='https://github.com/pabloestradac/spillover-effects',
    author='Pablo Estrada',
    author_email='pabloestradace@gmail.com',

    packages=find_packages(),

    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        ],
)