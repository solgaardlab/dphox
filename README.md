<p align="center">
<img src="https://user-images.githubusercontent.com/7623867/134089718-b4de5f82-adfb-4b20-9b98-b230748a73f9.png" width=50% height=50% alt="dphox">
</p>

#  

![Build Status](https://img.shields.io/travis/solgaardlab/dphox/main.svg?style=for-the-badge)
![Docs](https://readthedocs.org/projects/dphox/badge/?style=for-the-badge)
![PiPy](https://img.shields.io/pypi/v/dphox.svg?style=for-the-badge)
![CodeCov](https://img.shields.io/codecov/c/github/solgaardlab/dphox/main.svg?style=for-the-badge)

## At a glance

The `dphox` module is a Python 3-based design tool for automating photonic device development.

**Note**: This is a work in progress. Expect features in the code to be unstable until version `0.1.0`. Note the low
test coverage, which will be improved in coming weeks.

### Gallery links

## [Documentation and tutorials](https://dphox.readthedocs.io/en/latest/)

The [documentation](https://dphox.readthedocs.io/en/latest/) contains the API reference for `dphox`
and the tutorials you need to get started.

## Why `dphox`?

### Inspirations for `dphox`
- [`phidl`](https://phidl.readthedocs.io/en/latest/): path calculations, Inkscape-like maneuverability,
functional interface.
- [`nazca`](https://nazca-design.org): ports, cell references, and routing.
- [`gdspy`](https://gdspy.readthedocs.io/en/stable/): parametric implementations, the OG of python GDS automation

### Advantages of `dphox`
- Efficient raw `numpy` implementations for polygon and curve transformations
- Dependence on [`shapely`](https://shapely.readthedocs.io/en/stable/manual.html)
in favor of [`pyclipper`](https://pypi.org/project/pyclipper/) (less actively maintained).
  - `dphox.Curve` ~ `shapely.geometry.MultiLineString`
  - `dphox.Pattern` ~ `shapely.geometry.MultiPolygon`
- The [`klamath`](https://mpxd.net/code/jan/klamath/src/branch/master/klamath/elements.py) module
provides a clean implementation of GDS I/O
- Uses `trimesh` for 3D viewing/export, `blender` figures at your fingertips!
- Plotting using [`holoviews`](https://holoviews.org/) and [`bokeh`](http://docs.bokeh.org/en/latest/),
allowing zoom in/out in a notebook.
- More intuitive representation of GDS cell hierarchy (via `Device`).
- Interface to photonic simulation (such as [`simphox`](https://github.com/fancompute/simphox)
and [`MEEP`](meep.readthedocs.io/)).
- Inverse-designed devices may be incorporated via the `Pattern.replace` function.
- Read and interface with foundry PDKs automatically, even if provided via GDS.

## Installation

You may use `pip` to install `dphox` the usual way:

```
pip install dphox
```

To install all of the dependencies for visualizations as in the above demo, instead run:

```
pip install dphox[all]
```

### Development

When developing, install in your python environment using:

```
git clone git@github.com:solgaardlab/dphox.git
pip install -e dphox
```

You can then change `dphox` if necessary. When importing `dphox`, you can now treat it as any other module. No filepath
setting necessary because `dphox` will be in your environment's `site-packages` directory.


## Requirements



You will need `python>=3.8` as well as the following (note these requirements are automatically installed):



```
numpy==1.21.2
scipy==1.7.1
shapely==1.7.0
klamath==1.1
```

These will be installed via `pip` automatically if not already installed.

### Optional requirements

The following modules are nice-to-have but optional, and are not included in default installation:

```
bokeh==2.2.3
holoviews==1.14.6
trimesh==3.9.30
triangle==20200424
matplotlib==3.4.3
networkx
```

You can also install libraries such as `nazca` and `gdspy`, which can be converted to 
`dphox` objects.