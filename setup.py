#!/usr/bin/env python
from setuptools import setup, find_packages

project_name = "dphox"

setup(
    name=project_name,
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19',
        'scipy==1.7.1',
        'shapely>=1.7.1',
        'klamath>=1.2'
    ],
    extras_require={
        'all': [
            'matplotlib>=3.4.2',
            'bokeh==2.2.3',
            'holoviews==1.14.6',
            'trimesh==3.9.30',
            'triangle==20200424'
        ]
    }
)
