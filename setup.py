#!/usr/bin/env python
from setuptools import setup

project_name = "dphox"

setup(
    name=project_name,
    version="0.0.1a3",
    packages=[project_name],
    install_requires=[
        'pydantic==1.8.2',
        'numpy==1.21.2',
        'scipy==1.7.1',
        'matplotlib==3.4.3',
        'shapely==1.7.1',
        'gdspy==1.6.8',
        'descartes==1.1.0',
        'klamath==1.1'
    ],
    extras_require={
        'all': [
            'bokeh==2.2.3',
            'holoviews==1.14.6',
            'trimesh==3.9.30',
            'triangle==20200424'
        ]
    }
)
