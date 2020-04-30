# simphox
Design and simulation module for photonics

Install in your python environment using:

`pip install -e simphox`

You can then change `simphox` if necessary.
When importing `simphox`, you can now treat it as any other module.
No filepath setting necessary because `simphox` will be in your environment's `site-packages` directory.

For the AIM PDK imports, please save PDK files
in a separate folder of your choice (or in `simphox/aim_lib/`).
You will always specify these folders when using the PDK 
(see `simphox.gds.aim.AIMPhotonicsChip`). Please do not commit
these files to `simphox` as they tend to inflate contributions
(these are specified via a `.gitignore`).