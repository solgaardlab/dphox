# AIM PDK

This code was primarily written by Yu Miao and JP Macclean. First, import the aim pdk module from `simphox`.
```python
from simphox.design import aim
```

You will also need to have a folder in 

## Adding AIM PDK component
all the component cell name is the same as the cell name from the PDK
for example, in order to draw a silicon Y spliter (the cell name of silicon Y junction in AIM PDK is cl_band_splitter_3port_si), type:
```python
aim.cl_band_splitter_3port_si.put()
```

All the pins have been added to the cells, in order to utilize the pin connection, for example connecting a grating coupler to a silicon Y junction:
```python
si_gc = aim.cl_band_vertical_coupler_si.put()
aim.cl_band_splitter_3port_si.put('a0',si_gc.pin['b0'])
```

## Adding single mode waveguide
In AIM_PDK_component python file, defines the single mode waveguide for silicon layer, first nitride layer and nitride slot waveguide. The coresponding cell name for these three is also same as the cell name in AIM PDK document: cl_band_waveguide_si, cl_band_waveguide_FN, cl_band_waveguide_FNSN

Python file parameterized the waveguide by its length, whether need a turn and turn angle. By default, we assume a straight waveguide.
To add a waveguide is same as adding other component, but with input argument waveguide either length or angle.

For example, adding a silicon 20um long straight silicon single mode waveguide:
```python
aim.cl_band_waveguide_si(length = 20).put()
```

Add a turn of 90 degree turn of the silicon waveguide by specifying a non-zero angle:
```python
aim.cl_band_waveguide_si(angle = 90).put()
```
