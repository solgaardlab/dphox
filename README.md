<p align="center">
<img src="https://user-images.githubusercontent.com/7623867/134089718-b4de5f82-adfb-4b20-9b98-b230748a73f9.png" width=50% height=50% alt="dphox">
</p>

# 
![Build Status](https://img.shields.io/travis/solgaardlab/dphox/main.svg?style=for-the-badge)
![Docs](https://readthedocs.org/projects/dphox/badge/?style=for-the-badge)
![PiPy](https://img.shields.io/pypi/v/dphox.svg?style=for-the-badge)
![CodeCov](https://img.shields.io/codecov/c/github/solgaardlab/dphox/main.svg?style=for-the-badge)

[Documentation](https://dphox.readthedocs.io/en/latest/)

The `dphox` module is a Python 3-based design tool for automating photonic device development. 

**Note**: This is a work in progress. Expect features in the code to be unstable until
version `0.1.0`. Note the low test coverage, which will be improved in coming weeks.


https://user-images.githubusercontent.com/7623867/134403597-b7c0dd8f-81d0-4b7b-9da4-34b362442486.mp4


## Installation

You may use `pip` to install `dphox` the usual way:
```
pip install dphox
```

### Development

When developing, install in your python environment using:

```
git clone git@github.com:solgaardlab/dphox.git
pip install -e dphox
```

You can then change `dphox` if necessary.
When importing `dphox`, you can now treat it as any other module.
No filepath setting necessary because `dphox` will be in your environment's `site-packages` directory.


## Key features

The `dphox` module has the following features that incorporate the strategies used by 
other open-sourced python-based GDS tools.

### Strongly-typed device cells
In `dphox`, we present a hierarchical format for building highly complex designs based on
[`pydantic`](https://pydantic-docs.helpmanual.io/),
which provides a strongly typed config format for designs that are independent of any GDS file definition.
```python
import dphox as dx

# A waveguide that is 0.5 um wide and 10 um long.
wg = dx.Waveguide((0.5, 10))

# A waveguide that starts out 0.2 um wide and is symmetrically
# cubic-tapered in and out by 0.1 um over the span of
# 5 um (10 um total).
sub_wg = dx.Waveguide(
    extent=(0.2, 10),
    taper=dx.TaperSpec.cubic(-0.1, 5)
)

# equivalent to (wg - sub_wg), but annotated using Pydantic
wg_with_sub_wg_gap = dx.Waveguide(
    extent=(0.5, 10),
    subtract_waveguide=sub_wg
)
```

#### Exporting and importing from json

Exporting and importing from a json object is as simple as:

```python
import dphox as dx
import json
from pydantic.json import pydantic_encoder

# A waveguide with fins that may be used as a double-slot phase shifting element
wg = dx.Waveguide(
    extent=(1, 1),
    taper=dx.TaperSpec.cubic(0.4, 0.3),
    subtract_waveguide=dx.Waveguide(
        extent=(0.8, 1),
        taper=dx.TaperSpec.cubic(0.2, -0.2),
        subtract_waveguide=dx.Waveguide(
            extent=(0.7, 1),
            taper=dx.TaperSpec.cubic(0.2, -0.2),
            subtract_waveguide=dx.Waveguide((0.1, 1))
        )
    )
)

# Export to json
s = json.dumps(wg, indent=4, default=pydantic_encoder)
# Import from json
wg_recover = dx.Waveguide.__pydantic_model__.parse_raw(s)
```

### Port snapping

Like GDS-based tools such as [`phidl`](https://github.com/amccaugh/phidl) and [nazca](https://nazca-design.org),
components can be snapped together at ports, significantly 
simplifying many photonic and electronic routing problems.

### Foundry layer maps and visualization
A [`foundry`](https://github.com/solgaardlab/dphox/blob/main/dphox/foundry.py) module that can be used to import design specs from specific foundries. This module simultaneously provide 3D visualizations of designs, a method for saving designs
to GDS files, and ways to export to external simulators such as [COMSOL](https://www.comsol.com/),
[Lumerical](https://www.lumerical.com/),
[MEEP](https://meep.readthedocs.io/en/latest/), and [`simphox`](https://github.com/fancompute/simphox).

Because foundry layer maps are generally confidential information, we only expose a demonstration `FABLESS`
foundry object whose layer map results in a [KLayout](https://www.klayout.de/) -friendly color scheme and contains "common" foundry layers
one might expect in a typical silicon photonics flow.


### Clean GDS import/export functionality
Clean and efficient GDS import / export using the
[`klamath`](https://mpxd.net/code/jan/klamath/src/branch/master/klamath/elements.py) module.

```python
import dphox as dx

# make a waveguide that has a slab width of 1um and 
rib_wg = dx.WaveguideDevice(
    ridge_waveguide=dx.Waveguide((0.5, 10)),
    slab_waveguide=dx.Waveguide((1, 10))
)

# export to GDS file using klamath
rib_wg.to_gds('rib_wg.gds')

# An unannotated rib waveguide (one that isn't a pydantic model) can be loaded from gds directly
unannotated_rib_wg = dx.Device.from_gds('rib_wg.gds')
```

### Integrated CAD functionality
Adoption of CAD-based alignment/distribute features for easy manipulation of geometries
and fast prototyping of new strongly-typed component designs. Much inspiration for this feature comes from
[`phidl`](https://github.com/amccaugh/phidl). Here, we remove [`gdspy`](https://gdspy.readthedocs.io/en/stable/) and
[`clipper`](https://pypi.org/project/pyclipper/) dependencies in favor of the more actively maintained
[`shapely`](https://shapely.readthedocs.io/en/stable/manual.html) and [`GEOS`](https://trac.osgeo.org/geos) libraries.

Additionally, we use [`trimesh`]() to plot entire 3D stacks, which is useful to ensure complex metal stacks have no
crossings and for exporting to external simulators such as Lumerical and COMSOL. This can also be used to simulate some
fabrication processes, though at the moment we have limited it to patterning, clearouts, and doping (all of which only
require 2D operations).

### Interactive viewing in `holoviews`
Interactively view patterns / components in Holoviews. This can allow for quick and efficient
debugging of component geometries without the need to export to a gds, resulting in a workflow enhancement.

All you need to do is to install holoviews and import the following in a Jupyter notebook:
```python
import holoviews as hv
hv.extension('bokeh')
```

Then for any `Device` or `Pattern`, the command `device.hvplot()` will plot the entire device in HoloViews,
and create a dazzling GUI interface as shown in the above demo video.

Disclaimer: This method of visualizing devices is not recommended for layouts with many polygons, which benefit from
GDS cell references. [KLayout](https://www.klayout.de/) provides a more scalable solution for visualizing these.

## Requirements

You will need `python>=3.8` as well as the following (note these requirements are automatically installed):

```
pydantic==1.8.2
numpy==1.21.2
scipy==1.7.1
matplotlib==3.4.3
shapely==1.7.0
gdspy==1.6.8
descartes==1.1.0
klamath==1.1
```

These will be installed via `pip` automatically if not already installed.

### Optional requirements

The following modules are nice-to-have but optional, and are not included in default installation:
```
nazca
bokeh==2.2.3
holoviews==1.14.6
trimesh==3.9.30
triangle==20200424
```

Note that the [nazca](https://nazca-design.org) library allows one to actually efficiently treat cells
with complex geometries, resulting in fast GDS generation. Also `pip install nazca` will give you the wrong
library. Please install using instructions provided by [nazca](https://nazca-design.org).

If developing and you have a local repo, to install `holoviews`, `trimesh`, `triangle`, run:
```
pip install -r optional_requirements.txt
```
## Upcoming features

Contributions are welcome! Create a branch and [pull request (PR)](https://github.com/solgaardlab/dphox/pulls).

### Path functionality
It is important to `dphox` to remove external dependencies such as [`gdspy`](https://gdspy.readthedocs.io/en/stable/). 
Currently `dphox` relies on
`Path` and `FlexPath` functionality in [`gdspy`](https://gdspy.readthedocs.io/en/stable/). 
In `dphox`, we would ultimately want to add a similar `Path`
object that may be annotated, strongly typed, and fully-traceable, relying only on parametric functions defined 
in pure `numpy`.

### Layout generation
Efficient layouts require defining cell references to efficiently encode geometries in GDS files.
Fairly efficient layout solutions are provided by both 
[gdsfactory](https://gdsfactory.readthedocs.io/en/latest/README.html) and
[nazca](https://nazca-design.org). Ultimately this would also be a feature desired within `dphox`.
For now, a call to `device.nazca_cell(...)` should be one adequate conversion solution. 


## Collaborators

Below are the collaborators who made this library possible:
```
Sunil Pai
Nathnael S Abebe
Rebecca L Hwang
Yu Miao
```
## Git Workflow

### Adding a new feature branch

```
git pull # update local based on remote
git checkout develop # start branch from develop
git checkout -b feature/feature-branch-name
```

Do all work on branch. After your changes, from the root folder, execute the following:

```
git add . && git commit -m 'insert your commit message here'
```


### Rebasing and pull request

First you need to edit your commit history by "squashing" commits. 
You should be in your branch `feature/feature-branch-name`.
First look at your commit history to see how many commits you've made in your feature branch:

```
git log
```
Count the number of commits you've made and call that N.
Now, execute the following:

```
git rebase -i HEAD~N
```
Squash any insignificant commits (or all commits into a single commit if you like).
A good tutorial is provided 
[here](https://medium.com/@slamflipstrom/a-beginners-guide-to-squashing-commits-with-git-rebase-8185cf6e62ec).

Now, you should rebase on top of the `develop` branch by executing:
```
git rebase develop
```
You will need to resolve any conflicts that arise manually during this rebase process.

Now you will force-push this rebased branch using:
```
git push --set-upstream origin feature/feature-branch-name
git push -f
```

Then you must submit a pull request using this [link](https://github.com/solgaardlab/simphox/pulls).

### Updating develop and master

The admin of this repository is responsible for updating `develop` (unstable release)
and `master` (stable release). 
This happens automatically once the admin approves pull request.

```
git checkout develop
git merge feature/feature-branch-name
```

To update master:
```
git checkout master
git merge develop
```

As a rule, only one designated admin should have permissions to do these steps.
