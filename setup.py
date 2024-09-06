from setuptools import setup, find_packages

setup(
    name='three_d_scene_script',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)