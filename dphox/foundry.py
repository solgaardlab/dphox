from enum import Enum

import numpy as np
from dataclasses import field, dataclass
from shapely.geometry import box, MultiPolygon, Polygon

from .typing import Dict, Float3, LayerLabel, List, Optional, Callable
from .utils import fix_dataclass_init_docs


@fix_dataclass_init_docs
@dataclass
class Material:
    """Helper class for materials.

    Attributes:
        name: Name of the material.
        eps: Constant epsilon (relative permittivity) assigned for the material.
        facecolor: Facecolor in red-green-blue (RGB) for drawings (default is black or :code:`(0, 0, 0)`).
        alpha: transparency of the material for visualization
    """
    name: str
    eps: float = 1
    color: Float3 = (0, 0, 0)
    alpha: float = 1

    def __str__(self):
        return self.name

    @property
    def n(self):
        return np.sqrt(self.eps)


SILICON = Material('si', 3.4784 ** 2, (0.3, 0.3, 0.3))
POLYSILICON = Material('poly_si', 3.4784 ** 2, (0.5, 0.5, 0.5))
AIR = Material('air')
N_SILICON = Material('n_si', color=(0.4, 0.3, 0), alpha=0.3)
P_SILICON = Material('p_si', color=(0, 0.3, 0.4), alpha=0.3)
NN_SILICON = Material('nn_si', color=(0.4, 0.3, 0), alpha=0.5)
PP_SILICON = Material('pp_si', color=(0, 0.3, 0.4), alpha=0.5)
NNN_SILICON = Material('nnn_si', color=(0.4, 0.3, 0), alpha=0.7)
PPP_SILICON = Material('ppp_si', color=(0, 0.3, 0.4), alpha=0.7)
OXIDE = Material('sio2', 1.4442 ** 2, (0.6, 0, 0))
NITRIDE = Material('si3n4', 1.996 ** 2, (0, 0, 0.7))
LS_NITRIDE = Material('ls_sin', color=(0, 0.4, 1))
LT_OXIDE = Material('lto', 1.4442 ** 2, (0.8, 0.2, 0.2))
COPPER = Material('cu', color=(1, 0.6, 0))
ALUMINUM = Material('al', color=(0, 0.5, 0))  # (1.5785 + 15.658 * 1j) ** 2,
ALCU = Material('alcu', color=(0.2, 0.4, 0))
ALUMINA = Material('al2o3', 1.75, (0.2, 0, 0.2))
HEATER = Material('tin', color=(0.8, 0.8, 0))  # (3.1477 + 5.8429 * 1j) ** 2,
ETCH = Material('etch')
DUMMY = Material('dummy', color=(0.7, 0.7, 0.7))


class ProcessOp(str, Enum):
    """Enumeration for process operations which describe what happens at each step in a foundry process.

    Attributes:
        ISO_ETCH: isotropic etch under a pattern stencil (not yet supported)
        DRI_ETCH: directed-reactive ion etch (simply etch downward under a pattern stencil)
        SAC_ETCH: sacrificial etch (only affects cladding material in a process)
        GROW: grow over the previously deposited layer
        DOPE: dopes the previously deposited layer
        DUMMY: No process step associated with this

    """
    ISO_ETCH = 'iso_etch'
    DRI_ETCH = 'dri_etch'
    SAC_ETCH = 'sac_etch'
    GROW = 'grow'
    DOPE = 'dope'
    DUMMY = 'dummy'


class CommonLayer(str, Enum):
    """Common layers used in foundries. These are just strings representing common layers, no other functionality,

    Attributes:
        RIDGE_SI: Ridge silicon waveguide layer (waveguiding portion of the waveguide).
        RIDGE_SI_2: Another ridge silicon waveguide layer.
        RIB_SI: Rib silicon waveguide layer (slab portion of the waveguide).
        RIB_SI_2: Another rib silicon waveguide layer.
        PDOPED_SI: Lightly P-doped silicon (implants into the crystalline silicon layer).
        NDOPED_SI: Lightly N-doped silicon (implants into the crystalline silicon layer).
        PPDOPED_SI: Medium P-doped silicon (implants into the crystalline silicon layer).
        NNDOPED_SI: Medium N-doped silicon (implants into the crystalline silicon layer).
        PPPDOPED_SI: Highly P-doped silicon (implants into the crystalline silicon layer).
        NNNDOPED_SI: Highly N-doped silicon (implants into the crystalline silicon layer).
        RIDGE_SIN: Silicon nitride ridge layer (usually above silicon).
        ALUMINA: Alumina layer (for etch stop and waveguides, usually done in post-processing).
        POLY_SI_1: Polysilicon layer 1 (typically used in MEMS process).
        POLY_SI_2: Polysilicon layer 2 (typically used in MEMS process).
        POLY_SI_3: Polysilicon layer 3 (typically used in MEMS process).
        VIA_SI_1: Via metal connection from :code:`si` to :code:`metal_1`.
        METAL_1: Metal layer corresponding to an intermediate routing layer (1).
        VIA_1_2: Via metal connection from :code:`metal_1` to :code:`metal_2`.
        METAL_2: Metal layer corresponding to an intermediate routing layer (2).
        VIA_2_PAD: Via metal connection from :code:`metal_2` to :code:`metal_pad`.
        METAL_PAD: Metal layer corresponding to pads that can be wirebonded or solder-bump bonded from the chip surface.
        HEATER: Heater layer (usually titanium nitride).
        VIA_HEATER_2: Via metal connection from :code:`heater` to :code:`metal_2`.
        CLAD: Cladding layer (usually oxide).
        CLEAROUT: Clearout layer for a MEMS release process.
        PHOTONIC_KEEPOUT: A layer specifying where photonics cannot be routed.
        METAL_KEEPOUT: A layer specifying where metal cannot be routed.
        BBOX: Layer for the bounding box of the design.

    """
    RIDGE_SI = 'ridge_si'
    RIDGE_SI_2 = 'ridge_si_2'
    RIB_SI = 'rib_si'
    RIB_SI_2 = 'rib_si_2'
    RIDGE_SIN = 'ridge_sin'
    RIDGE_SIN_2 = 'ridge_sin_2'
    RIB_SIN = 'rib_sin'
    P_SI = 'p_si'
    N_SI = 'n_si'
    PP_SI = 'pp_si'
    NN_SI = 'nn_si'
    PPP_SI = 'pppdoped_si'
    NNN_SI = 'nnndoped_si'
    ALUMINA = 'alumina'
    POLY_SI_1 = 'poly_si_1'
    POLY_SI_2 = 'poly_si_2'
    POLY_SI_3 = 'poly_si_3'
    VIA_SI_1 = 'via_si_1'
    METAL_1 = "metal_1"
    VIA_1_2 = "via_1_2"
    METAL_2 = "metal_2"
    VIA_2_PAD = "via_2_pad"
    PAD = "pad"
    HEATER = "heater"
    VIA_HEATER_2 = "via_heater_2"
    CLAD = "clad"
    OXIDE_OPEN = "oxide_open"
    CLEAROUT = "clearout"
    PHOTONIC_KEEPOUT = "photonic_keepout"
    METAL_KEEPOUT = "metal_keepout"
    BBOX = "bbox"
    BBOX_SI = "bbox_si"
    BBOX_SIN = "bbox_sin"
    BBOX_METAL = "bbox_metal"
    BBOX_LABEL = "bbox_label"
    BBOX_METAL_1 = "bbox_metal_1"
    BBOX_METAL_2 = "bbox_metal_2"
    BBOX_RIB_SI = "bbox_rib_si"
    TRENCH = "trench"


@fix_dataclass_init_docs
@dataclass
class ProcessStep:
    """The :code:`ProcessStep` class is an object that stores all the information about a layer in a foundry process.

    Attributes:
        process_op: Process operation, specified in the enum for :code:`ProcessOp`
        thickness: The thickness spec for the process step.
        mat: Material (relevant to the non-etch :code:`ProcessOp.grow` and :code:`ProcessOp.dope` process ops)
        layer: The device layer corresponding to the process step
            (should NOT vary by foundry, use CommonLayer interface).
        gds_label: The GDS label used for GDS file creation of the device (SHOULD vary by foundry).
        foundry_layer: The device layer corresponding to the process step specified by foundry (SHOULD vary by foundry).
        start_height: The starting height for the process step.

    """
    process_op: ProcessOp
    thickness: float
    mat: Material
    layer: str
    gds_label: LayerLabel
    foundry_layer_name: Optional[str] = None
    start_height: Optional[float] = None


@fix_dataclass_init_docs
@dataclass
class Foundry:
    """The :code:`Foundry` class defines the full stack of process steps.

    For any step where the :code:`start_height` si not specified, the :code:`Foundry` class will assume a start height
    that is directly above the previously deposited layer. Note this does not support conformal deposition as this
    assumes all layers below are planarized.

    Attributes:
        stack: List of process steps that when applied sequentially lead to a full foundry stack
        height: Overall height of the foundry stack.
        cladding: The cladding material used by the foundry (usually OXIDE). This material will more-or-less
            cover the entire
        port_layers: A list of common layers for extracting the appropriate layers.

    """
    stack: List[ProcessStep]
    height: float
    cladding: Material = None
    port_layers: List[CommonLayer] = field(default_factory=list)

    def __post_init__(self):
        start_height = 0
        self.port_fn = lambda d: {} if self.port_fn is None else self.port_fn
        for step in self.stack:
            step.start_height = start_height if step.start_height is None else step.start_height
            start_height = step.start_height + step.thickness if step.process_op == ProcessOp.GROW else step.start_height

    @property
    def layer_to_gds_label(self):
        return {step.layer: step.gds_label for step in self.stack}

    @property
    def gds_label_to_layer(self):
        return {label: layer for layer, label in self.layer_to_gds_label.items()}

    def color(self, layer: str):
        for step in self.stack:
            if step.layer == layer:
                return step.mat.color


def fabricate(layer_to_geom: Dict[str, MultiPolygon], foundry: Foundry, init_device: Optional["Scene"] = None,
              exclude_layer: Optional[List[CommonLayer]] = None) -> "Scene":
    """Fabricate a device based on a layer-to-geometry dict, :code:`Foundry`, and initial device (type :code:`Scene`).

    This method is fairly rudimentary and will not implement things like conformal deposition. At the moment,
    you can implement things like rib etches which can be determined using 2d shape operations. Depositions in
    layers above etched layers will just start from the maximum z extent of the previous layer. This is specified
    by the :code:`Foundry` stack.

    Args:
        layer_to_geom: A dictionary mapping each layer to the `full` Shapely geometry for that layer.
        foundry: The foundry for each layer
        init_device: The initial device on which to start fabrication  (useful for post-processing simulations).
        exclude_layer: Exclude all layers in this list.

    Returns:
        The device :code:`Scene` to visualize.

    """

    try:
        import triangle
    except ImportError:
        raise ImportError("Fabrication requires the triangle module to be compiled with trimesh")
    import trimesh
    from trimesh.creation import extrude_polygon
    from trimesh.scene import Scene

    def _shapely_to_mesh_from_step(_geom: MultiPolygon, _meshes: List["trimesh.Trimesh"], _step: ProcessStep):
        for _poly in _geom.geoms:
            if _poly.area > 1e-8:
                _meshes.append(extrude_polygon(_poly, height=_step.thickness))
        _mesh = trimesh.util.concatenate(_meshes) if len(_meshes) > 0 else trimesh.Trimesh()
        _mesh.visual.face_colors = _step.mat.color
        return _mesh

    device = Scene() if init_device is None else init_device
    prev_mat: Optional[Material] = None
    bound_list = np.array([p.bounds for _, p in layer_to_geom.items()]).T
    xy_extent = np.min(bound_list[0]), np.min(bound_list[1]), np.max(bound_list[2]), np.max(bound_list[3])
    clad_geometry = box(*xy_extent)
    mesh = extrude_polygon(clad_geometry, height=foundry.height)
    exclude_layer = [] if exclude_layer is None else exclude_layer
    prev_si_geom = None
    if 'clad' not in device.geometry and foundry.cladding is not None:
        device.add_geometry(mesh, geom_name='clad')
        mesh.visual.face_colors = (*foundry.cladding.color, 0.5)
    for step in foundry.stack:
        # move the pattern to the previous maximum z height (previous mesh) OR start_height if specified in step.
        dz = step.start_height
        meshes = []
        layer = step.layer
        if layer in exclude_layer:
            continue
        elif layer in layer_to_geom:
            # only silicon (rib/ridge) and metal (via, pad, multilevel) generally have multiple layers
            geom = layer_to_geom[layer]
            mesh_name = f"{step.mat.name}_{layer}" if step.mat.name in {SILICON.name, ALUMINUM.name} else layer
            if step.process_op == ProcessOp.GROW:
                mesh = _shapely_to_mesh_from_step(geom, meshes, step)
                device.add_geometry(mesh.apply_translation((0, 0, dz)), geom_name=mesh_name)
            elif step.process_op == ProcessOp.DRI_ETCH:
                # Directly etch device
                # TODO: convert this to a 2D geometry function
                raise NotImplementedError(f"Fabrication method not yet implemented for `{step.process_op.value}`")
            elif step.process_op == ProcessOp.DOPE:
                # only support to dope silicon at the moment
                if prev_mat != SILICON:
                    raise ValueError("The previous material must be crystalline silicon for dopant implantation.")
                geoms = []
                for poly_si in prev_si_geom.geoms:
                    for poly in geom:
                        mat = poly.intersection(poly_si)
                        if isinstance(mat, Polygon) or isinstance(mat, MultiPolygon):
                            geoms.append(mat)
                mesh = _shapely_to_mesh_from_step(MultiPolygon(geoms), meshes, step)
                device.add_geometry(mesh.apply_translation((0, 0, dz - step.thickness)), geom_name=mesh_name)
            elif step.process_op == ProcessOp.SAC_ETCH:
                if 'clad' not in device.geometry:
                    raise ValueError("The cladding is not in the device geometry / not spec'd by the foundry object.")
                clad_geometry -= geom
                clad_geometry = MultiPolygon([clad_geometry]) if isinstance(clad_geometry, Polygon) else clad_geometry
                for poly in clad_geometry:
                    meshes.append(extrude_polygon(poly, height=step.thickness))
                device.geometry['clad'] = trimesh.util.concatenate(meshes)
                mesh.visual.face_colors = (*foundry.cladding.color, 0.5)
                # raise NotImplementedError(f"Fabrication method not yet implemented for `{step.process_op.value}`")
                # device.geometry['clad'] -= difference(device.geometry['clad'], mesh)
            elif not step.process_op == ProcessOp.DUMMY:
                raise NotImplementedError(f"Fabrication method not yet implemented for `{step.process_op.value}`")
            if step.mat.name == SILICON.name:
                prev_si_geom = geom
        if step.process_op == ProcessOp.GROW:
            prev_mat = step.mat
    return device


# Foundries are generally secretive about their exact stack/gds labels,
# so the below is one example stack for demo purposes.
FABLESS = Foundry(
    stack=[
        # 1. First define the photonic stack
        ProcessStep(ProcessOp.GROW, 0.2, SILICON, CommonLayer.RIDGE_SI, (100, 0), 2),
        ProcessStep(ProcessOp.DOPE, 0.1, P_SILICON, CommonLayer.P_SI, (400, 0)),
        ProcessStep(ProcessOp.DOPE, 0.1, N_SILICON, CommonLayer.N_SI, (401, 0)),
        ProcessStep(ProcessOp.DOPE, 0.1, PP_SILICON, CommonLayer.PP_SI, (402, 0)),
        ProcessStep(ProcessOp.DOPE, 0.1, NN_SILICON, CommonLayer.NN_SI, (403, 0)),
        ProcessStep(ProcessOp.DOPE, 0.1, PPP_SILICON, CommonLayer.PPP_SI, (404, 0)),
        ProcessStep(ProcessOp.DOPE, 0.1, NNN_SILICON, CommonLayer.NNN_SI, (405, 0)),
        ProcessStep(ProcessOp.GROW, 0.1, SILICON, CommonLayer.RIB_SI, (101, 0), 2),
        ProcessStep(ProcessOp.GROW, 0.2, NITRIDE, CommonLayer.RIDGE_SIN, (300, 0), 2.5),
        ProcessStep(ProcessOp.GROW, 0.1, ALUMINA, CommonLayer.ALUMINA, (200, 0), 2.5),
        # 2. Then define the metal connections (zranges).
        ProcessStep(ProcessOp.GROW, 1, COPPER, CommonLayer.VIA_SI_1, (500, 0), 2.2),
        ProcessStep(ProcessOp.GROW, 0.2, COPPER, CommonLayer.METAL_1, (501, 0)),
        ProcessStep(ProcessOp.GROW, 0.5, COPPER, CommonLayer.VIA_1_2, (502, 0)),
        ProcessStep(ProcessOp.GROW, 0.2, COPPER, CommonLayer.METAL_2, (503, 0)),
        ProcessStep(ProcessOp.GROW, 0.5, ALUMINUM, CommonLayer.VIA_2_PAD, (504, 0)),
        # Note: negative means grow downwards (below the ceiling of the device).
        ProcessStep(ProcessOp.GROW, 0.3, ALUMINUM, CommonLayer.PAD, (600, 0)),
        ProcessStep(ProcessOp.GROW, 0.2, HEATER, CommonLayer.HEATER, (700, 0), 3.2),
        ProcessStep(ProcessOp.GROW, 0.5, COPPER, CommonLayer.VIA_HEATER_2, (505, 0)),
        # 3. Finally specify the clearout (needed for MEMS).
        ProcessStep(ProcessOp.SAC_ETCH, 4, ETCH, CommonLayer.CLEAROUT, (800, 0)),
        ProcessStep(ProcessOp.DUMMY, 4, DUMMY, CommonLayer.TRENCH, (41, 0)),
        ProcessStep(ProcessOp.DUMMY, 4, DUMMY, CommonLayer.PHOTONIC_KEEPOUT, (42, 0)),
        ProcessStep(ProcessOp.DUMMY, 4, DUMMY, CommonLayer.METAL_KEEPOUT, (43, 0)),
        ProcessStep(ProcessOp.DUMMY, 4, DUMMY, CommonLayer.BBOX, (44, 0)),
    ],
    height=5
)
