from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ..device import Device, Via
from ..foundry import CommonLayer
from .passive import DC, RibDevice
from ..pattern import Box, MEMSFlexure, Pattern, Port
from ..parametric import straight
from ..transform import GDSTransform
from ..typing import List, Union
from ..utils import fix_dataclass_init_docs


@fix_dataclass_init_docs
@dataclass
class ThermalPS(Device):
    """Thermal phase shifter (e.g. TiN phase shifter).

    Attributes:
        waveguide: Waveguide under the phase shifter
        ps_w: Phase shifter width
        via: Via to connect heater to the top metal layer
        ridge: Waveguide layer
        ps_layer: Phase shifter layer (e.g. TiN)
    """
    waveguide: Pattern
    ps_w: float
    via: Via
    ridge: str = CommonLayer.RIDGE_SI
    heater: str = CommonLayer.HEATER
    name: str = "thermal_ps"

    def __post_init__(self):
        ps = self.waveguide.curve.coalesce().path(self.ps_w)
        left_via = self.via.copy.align(self.waveguide.port['a0'].xy)
        right_via = self.via.copy.align(self.waveguide.port['b0'].xy)

        super(ThermalPS, self).__init__(
            self.name,
            [(self.waveguide, self.ridge), (ps, self.heater)] + left_via.pattern_to_layer + right_via.pattern_to_layer
        )
        self.port = self.waveguide.port
        self.ps = ps
        self.port['gnd'] = Port(self.waveguide.port['a0'].x, 0, -180)
        self.port['pos'] = Port(self.waveguide.port['b0'].x, 0)
        self.wg_path = self.layer_to_polys[self.ridge]


@fix_dataclass_init_docs
@dataclass
class PullOutNemsActuator(Device):
    """Pull out NEMS actuator moves material `away` from the core waveguide to change the phase.

    Attributes:
        pos_pad: Electrode box
        flexure: Connector box
        dope_expand_tuple: Dope expand tuple. This first applies expand, then grow operation on pos pad. For the
            crab-leg flexure, this applies a grow operation that is the sum of the two tuple elements
        ridge: Waveguide layer (usually silicon)
        actuator_dope: Actuator dope setting (set lower than pos_pad dope if possible to improve spring mechanics)
        pos_pad_dope: Electrode dope setting

    """
    pos_pad: Box
    pad_sep: float
    flexure: MEMSFlexure
    connector: Box
    via: Via
    dope_expand_tuple: Tuple[float, float] = (0, 0)
    ridge: str = CommonLayer.RIDGE_SI
    actuator_dope: str = CommonLayer.P_SI
    pos_pad_dope: str = CommonLayer.PPP_SI
    name: str = "pull_out_actuator"

    def __post_init__(self):
        dope_total_offset = self.dope_expand_tuple[0] + self.dope_expand_tuple[1]
        pos_pad = self.pos_pad.copy.vstack(self.flexure, bottom=True).translate(dy=self.pad_sep)
        connectors = [
            (self.connector.copy.vstack(self.flexure).halign(self.flexure.box, left=True), self.ridge),
            (self.connector.copy.vstack(self.flexure).halign(self.flexure.box, left=False), self.ridge)
        ]
        dopes = [
            (pos_pad.copy.expand(self.dope_expand_tuple[0]).buffer(self.dope_expand_tuple[1]), self.pos_pad_dope),
            (self.flexure.copy.buffer(dope_total_offset), self.actuator_dope),
        ]
        via = self.via.copy.align(pos_pad.center)
        super(PullOutNemsActuator, self).__init__(
            self.name, dopes + connectors + [(pos_pad, self.ridge), (self.flexure, self.ridge)] + via.pattern_to_layer
        )
        self.translate(dy=-self.bounds[1])


@fix_dataclass_init_docs
@dataclass
class PullInNemsActuator(Device):
    """Pull in NEMS actuator moves material `toward` the core waveguide to change the phase.

    Actually, this class is really not an actuator, but it provides the electrode that produces the actuation.
    However, in terms of constructing the phase shifter object, it serves the same purpose as the pull-out variety
    in terms of functionality.

    Attributes:
        pos_pad: Pad for the positive terminal for the NEMS actuation
        connector: Connector box for connecting the device
        dope_expand_tuple: Dope expand tuple (first applies expand, then grow operation on ground pad)
        ridge: Waveguide layer (usually silicon)
        dope: Electrode dope setting

    """
    pos_pad: Box
    connector: Box
    via: Via
    dope_expand_tuple: Tuple[float, float] = (0, 0)
    ridge: str = CommonLayer.RIDGE_SI
    dopes: str = CommonLayer.PPP_SI
    name: str = "pull_in_actuator"

    def __post_init__(self):
        via = self.via.align(self.pos_pad.center)
        connectors = [
            (self.connector.copy.halign(self.pos_pad, left=True).valign(self.pos_pad, bottom=False, opposite=True),
             self.ridge),
            (self.connector.copy.halign(self.pos_pad, left=False).valign(self.pos_pad, bottom=False, opposite=True),
             self.ridge)
        ]
        dopes = [
            (self.pos_pad.copy.expand(self.dope_expand_tuple[0]).buffer(self.dope_expand_tuple[1]), self.dopes)
        ]
        super(PullInNemsActuator, self).__init__(
            self.name, connectors + dopes + [(self.pos_pad, self.ridge)] + via.pattern_to_layer
        )


@fix_dataclass_init_docs
@dataclass
class GndAnchorWaveguide(Device):
    """Ground anchor waveguide device useful for connecting a waveguide to the ground plane.

    Attributes:
        rib_waveguide: Transition waveguide for connecting to ground pads
        gnd_pad: Ground pad for ultimately connecting the waveguide to the ground plane
        gnd_connector: Ground connector for connecting the anchor waveguide to the ground pad :code:`gnd_pad`.
        offset_into_rib: Offset of the ground connector into the rib.
        dope_expand_tuple: Dope expand tuple (first applies expand, then grow operation on ground pad)
        ridge: Waveguide layer (usually silicon)
        gnd_pad_dope: Electrode dope setting

    """
    rib_waveguide: RibDevice
    gnd_pad: Box
    gnd_connector: Box
    via: Via
    offset_into_rib: float
    dope_expand_tuple: Tuple[float, float] = (0, 0)
    ridge: str = CommonLayer.RIDGE_SI
    gnd_pad_dope: str = CommonLayer.PPP_SI
    name: str = "gnd_anchor_waveguide"

    def __post_init__(self):
        gnd_connectors = [
            self.gnd_connector.copy.vstack(self.rib_waveguide.slab_waveguide).translate(dy=self.offset_into_rib),
            self.gnd_connector.copy.vstack(self.rib_waveguide.slab_waveguide,
                                           bottom=True).translate(dy=-self.offset_into_rib),
        ]
        gnd_pads = [
            self.gnd_pad.copy.vstack(gnd_connectors[0]),
            self.gnd_pad.copy.vstack(gnd_connectors[1], bottom=True)
        ]
        vias = self.via.copy.align(gnd_pads[0]).pattern_to_layer + self.via.copy.align(gnd_pads[1]).pattern_to_layer
        dopes = [
            (gnd_pads[0].expand(self.dope_expand_tuple[0]).buffer(self.dope_expand_tuple[1]), self.gnd_pad_dope),
            (gnd_pads[1].expand(self.dope_expand_tuple[0]).buffer(self.dope_expand_tuple[1]), self.gnd_pad_dope),
            (gnd_connectors[0].expand(self.dope_expand_tuple[0]).buffer(self.dope_expand_tuple[1]), self.gnd_pad_dope),
            (gnd_connectors[1].expand(self.dope_expand_tuple[0]).buffer(self.dope_expand_tuple[1]), self.gnd_pad_dope),
        ]
        pattern_to_layer = [(p, self.ridge) for p in gnd_connectors + gnd_pads]
        super(GndAnchorWaveguide, self).__init__(
            self.name, pattern_to_layer + dopes + vias + self.rib_waveguide.pattern_to_layer
        )
        self.port = {
            'e0': gnd_pads[0].port['e'].copy,
            'e1': gnd_pads[1].port['e'].copy,
            'w0': gnd_pads[0].port['w'].copy,
            'w1': gnd_pads[1].port['w'].copy,
            'n0': gnd_pads[0].port['n'].copy,
            'n1': gnd_pads[1].port['n'].copy,
            's0': gnd_pads[0].port['s'].copy,
            's1': gnd_pads[1].port['s'].copy,
            'a0': self.rib_waveguide.port['a0'].copy,
            'b0': self.rib_waveguide.port['b0'].copy
        }


@fix_dataclass_init_docs
@dataclass
class Clearout(Device):
    """Clearout device which is generally useful for sacrificial etches in MEMS devices.

    Attributes:
        clearout_etch: THe clearout etch box, which selectively removes silicon dioxide.
        clearout_etch_stop_grow: The etch stop offset factor beyond the size of the clearout etch box
        clearout_layer: The clearout layer
        clearout_etch_stop_layer: The clearout etch stop layer (this can vary more than the clearout layer).

    """
    clearout_etch: Box
    clearout_etch_stop_grow: float
    clearout_layer: str = CommonLayer.CLEAROUT
    clearout_etch_stop_layer: str = CommonLayer.ALUMINA
    name: str = "clearout"

    def __post_init__(self):
        super(Clearout, self).__init__("clearout", [(self.clearout_etch, self.clearout_layer),
                                                    (self.clearout_etch.buffer(self.clearout_etch_stop_grow),
                                                     self.clearout_etch_stop_layer)])


@fix_dataclass_init_docs
@dataclass
class LateralNemsPS(Device):
    """Lateral NEMS phase shifter, which is actuated by the specified :code:`actuator` (pull-in or -out).

    Attributes:
        phase_shifter_waveguide: Phase shifter waveguide (including the nanofins)
        gnd_anchor_waveguide: Ground anchor waveguide connecting the waveguide to ground.
        clearout: Clearout of the oxide cladding material for the MEMS actuation.
        actuator: Actuator (pull-out or pull-in) for controlling the phase shift.
        ridge: Ridge silicon waveguide.
        trace_w: Trace width for the metal routing.
        clearout_pos_sep: Clearout trace separation for the pos terminal avoiding metal accidental etching (HF nasty).
        clearout_gnd_sep: Clearout trace separation for the gnd terminal avoiding metal accidental etching (HF nasty).
        pos_metal_layer: Positive electrode metal layer.
        gnd_metal_layer: Ground electrode metal layer.

    """
    waveguide_w: float
    phase_shifter_waveguide: Pattern
    gnd_anchor_waveguide: GndAnchorWaveguide
    clearout: Clearout
    actuator: Union[PullInNemsActuator, PullOutNemsActuator]
    trace_w: float
    clearout_pos_sep: float
    clearout_gnd_sep: float
    ridge: str = CommonLayer.RIDGE_SI
    pos_metal_layer: str = CommonLayer.METAL_1
    gnd_metal_layer: str = CommonLayer.METAL_2
    name: str = "lateral_nems_ps"

    def __post_init__(self):
        psw = self.phase_shifter_waveguide.copy
        top_actuator = self.actuator.copy.to(Port(psw.center[0], psw.bounds[3], 0))
        bottom_actuator = self.actuator.copy.to(Port(psw.center[0], psw.bounds[1], -180))
        clearout = self.clearout.copy.align(psw)
        left_gnd_waveguide = self.gnd_anchor_waveguide.copy.to(psw.port['a0'])
        right_gnd_waveguide = self.gnd_anchor_waveguide.copy.to(psw.port['b0'])
        pos_metal = Box((self.clearout.size[0] + self.clearout_pos_sep,
                         top_actuator.bounds[3] - bottom_actuator.bounds[1])).hollow(self.trace_w).align(clearout.center)
        gnd_metal = Box((right_gnd_waveguide.port['e0'].x - left_gnd_waveguide.port['e0'].x,
                         self.gnd_anchor_waveguide.size[1] + self.clearout_gnd_sep)).cup(self.trace_w).halign(
            left_gnd_waveguide.port['e0'].x).valign(left_gnd_waveguide.port['e1'].y)

        super(LateralNemsPS, self).__init__(self.name, [(psw, self.ridge), (pos_metal, self.pos_metal_layer),
                                                        (gnd_metal, self.gnd_metal_layer), top_actuator,
                                                        bottom_actuator, clearout, left_gnd_waveguide,
                                                        right_gnd_waveguide])
        self.merge_patterns()
        self.port = {
            'a0': left_gnd_waveguide.port['b0'],
            'b0': right_gnd_waveguide.port['b0']
        }
        self.port['a0'].w = self.port['b0'].w = self.waveguide_w
        self.wg_path = self.phase_shifter_waveguide


PathComponent = Union[Pattern, Device, "MultilayerPath", float]


@fix_dataclass_init_docs
@dataclass
class MultilayerPath(Device):
    """Multilayer path for appending a linear sequence of elements end-to-end

    Attributes:
        waveguide_w: Waveguide width.
        sequence: Sequence of :code:`Device`, :code:`Pattern` or :code:`float`s.
            The :code:`float` corresponds to straight waveguides of such lengths
            and waveguide width :code:`waveguide_w`).
        path_layer: Pattern layer.
        name: Name of the device.
    """
    waveguide_w: float
    sequence: List[PathComponent]
    path_layer: str
    name: str = 'multilayer_path'
    start_port: Port = Port()

    def __post_init__(self):
        waveguided_patterns = []
        if not self.sequence or np.isscalar(self.sequence[0]):
            port = self.start_port
        else:
            start = self.sequence[0].port['a0']
            port = Port(*start.xy, start.a - 180)
        child_to_device = {}
        child_to_ports = defaultdict(list)
        for p in self.sequence:
            if p is not None:
                d = p.copy if isinstance(p, Device) or isinstance(p, Pattern) else straight(p).path(self.waveguide_w)
                if isinstance(d, Device) and d.child_to_device:
                    child_to_device[d.name] = d
                    child_to_ports[d.name].append(port.copy)
                    port = d.dummy_port_pattern.to(port, 'a0').port['b0'].copy
                else:
                    waveguided_patterns.append(d.to(port, 'a0'))
                    port = d.port['b0'].copy
        pattern_to_layer = sum(
            [[(p, self.path_layer)] if isinstance(p, Pattern) else [p] for p in waveguided_patterns],
            [])
        super(MultilayerPath, self).__init__(self.name, pattern_to_layer)
        for child in child_to_device:
            self.place(child_to_device[child], child_to_ports[child], 'a0')
        self.port['a0'] = waveguided_patterns[0].port['a0'].copy if waveguided_patterns else Port(*port.xy, 180)
        self.port['b0'] = port
        self.x_length = self.port['b0'].x - self.port['a0'].x

    def extend(self, path: Union[float, Pattern]):
        """Extend this multilayer path using a path in the :code:`path_layer`.

        Args:
            path: The path to add (either a float for straight segment or a path pattern)

        Returns:
            The updated multilayer path.

        """
        self.sequence.append(path)
        path = straight(path).path(self.waveguide_w) if np.isscalar(path) else path
        segment = path.to(self.port['b0'])
        self.add(segment, self.path_layer)
        self.port['b0'] = segment.port['b0']
        return self


@fix_dataclass_init_docs
@dataclass
class MZI(Device):
    """An MZI with multilayer devices in the arms (e.g., phase shifters and/or grating taps).

    Note:
        This class assumes that the arms begin and end along the horizontal (:math:`x`-axis).

    Attributes:
        coupler: Directional coupler or MMI for MZI
        ridge: Waveguide layer string
        top_internal: Top center arm (waveguide matching bottom arm length if None)
        bottom_internal: Bottom center arm (waveguide matching top arm length if None)
        top_external: Top input (waveguide matching bottom arm length if None)
        bottom_external: Bottom input (waveguide matching top arm length if None)
        name: Name of the MZI
    """
    coupler: DC
    ridge: str = CommonLayer.RIDGE_SI
    top_internal: List[PathComponent] = field(default_factory=list)
    bottom_internal: List[PathComponent] = field(default_factory=list)
    top_external: List[PathComponent] = field(default_factory=list)
    bottom_external: List[PathComponent] = field(default_factory=list)
    name: str = "mzi"

    def __post_init__(self):
        dc_device = self.coupler.device(self.ridge)
        w = self.coupler.waveguide_w
        self.top_input = MultilayerPath(w, self.top_external, self.ridge, name=f'{self.name}_top_input')
        self.bottom_input = MultilayerPath(w, self.bottom_external, self.ridge, name=f'{self.name}_bottom_input')
        self.top_arm = MultilayerPath(w, self.top_internal, self.ridge, name=f'{self.name}_top_arm')
        self.bottom_arm = MultilayerPath(w, self.bottom_internal, self.ridge, name=f'{self.name}_bottom_arm')
        arm_length_diff = self.top_arm.x_length - self.bottom_arm.x_length
        if self.top_arm.x_length > self.bottom_arm.x_length:
            self.bottom_arm.extend(arm_length_diff)
        else:
            self.top_arm.extend(arm_length_diff)
        super(MZI, self).__init__(self.name)
        self.place(self.bottom_input, Port(), 'a0')
        self.place(self.top_input, Port(0, self.coupler.interport_distance), 'a0')
        dc_port = dc_device.dummy_port_pattern.to(self.bottom_input.port['b0'], 'a0').port
        self.place(dc_device, self.top_input.port['b0'])
        self.place(self.bottom_arm, dc_port['b0'], 'a0')
        self.place(self.top_arm, dc_port['b1'], 'a0')
        bottom_arm_port = self.bottom_arm.dummy_port_pattern.to(dc_port['b0'], 'a0').port['b0']
        self.place(dc_device, bottom_arm_port, 'a0')
        self.init_coupler = dc_device
        self.final_coupler = dc_device.copy.to(bottom_arm_port, 'a0')
        self.port = {
            'a0': Port(0, 0, 180, w=w),
            'a1': Port(0, self.coupler.interport_distance, 180, w=w),
            'b0': self.final_coupler.port['b0'].copy,
            'b1': self.final_coupler.port['b1'].copy
        }
        self.interport_distance = self.coupler.interport_distance
        self.waveguide_w = self.coupler.waveguide_w
        self.input_length = self.top_input.x_length
        self.arm_length = self.top_arm.x_length
        self.full_length = self.port['b0'].x - self.port['a0'].x

    def path(self, flip: bool = False):
        first = self.init_coupler.lower_path.copy.reflect() if flip else self.init_coupler.lower_path
        second = self.final_coupler.lower_path.copy.reflect() if flip else self.final_coupler.lower_path
        return MultilayerPath(
            waveguide_w=self.coupler.waveguide_w,
            sequence=[self.bottom_input.copy, first.copy, self.bottom_arm.copy, second.copy],
            path_layer=self.ridge
        )


@fix_dataclass_init_docs
@dataclass
class LocalMesh(Device):
    """A locally interacting rectangular mesh, or triangular mesh if specified.
    Note: triangular meshes can self-configure, but rectangular meshes cannot.

    Attributes:
        mzi: The :code:`MZI` object, which acts as the unit cell for the mesh.
        n: The number of inputs and outputs for the mesh.
        triangular: Triangular mesh, otherwise rectangular mesh
        name: Name of the device
    """

    mzi: MZI
    n: int
    triangular: bool = True
    name: str = 'mesh'

    def __post_init__(self):
        mzi = self.mzi
        n = self.n
        triangular = self.triangular
        num_straight = (n - 1) - np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) - 1 if triangular \
            else np.tile((0, 1), n // 2)[:n]
        n_layers = 2 * n - 3 if triangular else n

        self.upper_path = mzi.path(flip=False)
        self.lower_path = mzi.path(flip=True)
        self.mzi_out = mzi.bottom_input.copy
        self.upper_path.name = 'upper_mzi_path'
        self.lower_path.name = 'lower_mzi_path'
        self.mzi_out.name = 'mzi_out'

        self.upper_transforms = []
        self.lower_transforms = []
        self.out_transforms = []
        self.flip_array = np.zeros((n, n_layers))
        for idx in range(n):  # waveguides
            for layer in range(n_layers):
                flip = idx == n - 1 or (idx - layer % 2 < n and idx > num_straight[layer]) and (idx + layer) % 2
                if flip:
                    self.lower_transforms.append((layer * mzi.full_length, idx * mzi.interport_distance))
                    self.flip_array[idx, layer] = 1
                else:
                    self.upper_transforms.append((layer * mzi.full_length, idx * mzi.interport_distance))
            self.out_transforms.append((n_layers * mzi.full_length, idx * mzi.interport_distance))

        super(LocalMesh, self).__init__(self.name)
        self.place(self.upper_path, np.array(self.upper_transforms))
        self.place(self.lower_path, np.array(self.lower_transforms))
        self.place(self.mzi_out, np.array(self.out_transforms))
        self.port = {
            **{f'a{i}': Port(0, i * mzi.interport_distance, 180, mzi.waveguide_w) for i in range(n)},
            **{f'b{i}': Port(n_layers * mzi.full_length + mzi.input_length,
                             i * mzi.interport_distance, 0, mzi.waveguide_w) for i in range(n)}
        }
        self.interport_distance = mzi.interport_distance
        self.waveguide_w = self.mzi.waveguide_w
        self.n_layers = n_layers

    def demo_polys(self, ps_w_factor: float = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Demo polygons, useful for plotting stuff, using only the polygons in the silicon layer.

        Note:
            This method is generally useless unless used for demo purposes and will be deleted once a cleaner solution
            is found.

        Args:
            ps_w_factor: phase shifter width factor

        Returns:
            A numpy array consisting of lists of polygons (note: NOT numbers)
        """

        geoms = []

        lower_polys = self.lower_path.full_layer_to_polys[self.mzi.ridge]
        upper_polys = self.upper_path.full_layer_to_polys[self.mzi.ridge]
        out_polys = self.mzi_out.full_layer_to_polys[self.mzi.ridge]

        transformed_lower_polys = GDSTransform.parse(np.array(self.lower_transforms))[0].transform_geoms(lower_polys)
        transformed_upper_polys = GDSTransform.parse(np.array(self.upper_transforms))[0].transform_geoms(upper_polys)
        transformed_out_polys = GDSTransform.parse(np.array(self.out_transforms))[0].transform_geoms(out_polys)

        idx_upper = idx_lower = idx_out = 0
        for idx in range(self.n):
            path = []
            for layer in range(self.n_layers):
                if self.flip_array[idx, layer] == 1:
                    path.extend([p[idx_lower] for p in transformed_lower_polys])
                    idx_lower += 1
                else:
                    path.extend([p[idx_upper] for p in transformed_upper_polys])
                    idx_upper += 1
            path.extend([p[idx_out] for p in transformed_out_polys])
            idx_out += 1
            geoms.append(path)

        # TODO: this is a temporary hack to slice the path for the simplest mesh
        sizes = [0, 2] + [4] * (2 * self.n_layers - 1) + [3]
        slices = np.cumsum(sizes, dtype=int)

        path_array = np.array(
            [[geoms[i][slices[s]:slices[s + 1]] for s in range(len(slices) - 1)] for i in range(self.n)],
            dtype=object)
        ps_array = np.array([Pattern(geoms[i][s * 4]).scale(1, ps_w_factor).points.T
                             for i in range(self.n)
                             for s in range(len(slices) - 1)])

        return path_array, ps_array

    def demo_3d_arrays(self, ps_w_factor: float = 4, height=0.22, sep=0.22):
        from trimesh.creation import extrude_polygon
        import trimesh
        path_array, ps_array = self.demo_polys(ps_w_factor)
        ps_array = ps_array.reshape((*path_array.shape, 4, 2))

        def _shapely_to_mesh_from_step(_geom, translation):
            _meshes = []
            for _poly in _geom.geoms:
                _meshes.append(extrude_polygon(_poly, height=height))
            _mesh = trimesh.util.concatenate(_meshes) if len(_meshes) > 0 else trimesh.Trimesh()
            return _mesh.apply_translation((0, 0, translation))

        path_3d_array = []
        ps_3d_array = []
        for i in range(6):
            path_row = []
            ps_row = []
            for j in range(19):
                path_row.append(_shapely_to_mesh_from_step(Pattern(path_array[i, j]).shapely, 0))
                ps_row.append(_shapely_to_mesh_from_step(Pattern(ps_array[i, j].T).shapely, sep + height))
            path_3d_array.append(path_row)
            ps_3d_array.append(ps_row)

        return path_3d_array, ps_3d_array
