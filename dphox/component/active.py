from .multilayer import Multilayer, MultilayerPath
from .passive import DC, MMI, Waveguide
from .pattern import Port
from ..typing import List, Union

import numpy as np


class MZI(Multilayer):
    def __init__(self, coupler: Union[DC, MMI], ridge: str,
                 top_internal: List[Union[Multilayer, float]] = None,
                 bottom_internal: List[Union[Multilayer, float]] = None,
                 top_external: List[Union[Multilayer, float]] = None,
                 bottom_external: List[Union[Multilayer, float]] = None
                 ):
        """An MZI with multilayer devices in the arms (e.g., phase shifters and/or grating taps)

        Args:
            coupler: Directional coupler or MMI for MZI
            top_internal: Top arm (waveguide matching bottom arm length if None)
            bottom_internal: Bottom arm (waveguide matching top arm length if None)
            top_internal: Top input (waveguide matching bottom arm length if None)
            bottom_internal: Bottom input (waveguide matching top arm length if None)
            ridge: Waveguide layer string
        """
        patterns = [coupler]
        port = coupler.port

        if top_external:
            top_input = MultilayerPath(coupler.waveguide_w, top_external, ridge)
            top_input.to(coupler.port['a1'], 'b0')
            port['a1'] = top_input.port['a0']
            patterns.append(top_input)
        if bottom_external:
            bottom_input = MultilayerPath(coupler.waveguide_w, bottom_external, ridge)
            bottom_input.to(coupler.port['a0'], 'b0')
            port['a0'] = bottom_input.port['a0']
            patterns.append(bottom_input)

        if top_internal:
            top_arm = MultilayerPath(coupler.waveguide_w, top_internal, ridge).to(port['b1'])
            port['b1'] = top_arm.port['b0']
        if bottom_internal:
            bottom_arm = MultilayerPath(coupler.waveguide_w, bottom_internal, ridge).to(port['b0'])
            port['b0'] = bottom_arm.port['b0']

        arm_length_diff = port['b1'].x - port['b0'].x

        if arm_length_diff > 0:
            if bottom_internal:
                bottom_arm.append(arm_length_diff)
            else:
                bottom_arm = Waveguide(coupler.waveguide_w, arm_length_diff).to(port['b0'])
            port['b0'] = bottom_arm.port['b0']
        elif arm_length_diff < 0:
            if top_internal:
                top_arm.append(arm_length_diff)
            else:
                top_arm = Waveguide(coupler.waveguide_w, arm_length_diff).to(port['b0'])
            port['b1'] = top_arm.port['b1']

        patterns.extend([top_arm, bottom_arm])

        final_coupler = coupler.copy.to(port['b0'])
        patterns.append(final_coupler)

        pattern_to_layer = sum([[(p, ridge)] if isinstance(p, Pattern) else p.pattern_to_layer for p in patterns], [])

        super(MZI, self).__init__(pattern_to_layer)

        self.init_coupler = coupler
        self.final_coupler = final_coupler
        self.top_arm = top_arm
        self.bottom_arm = bottom_arm
        self.top_input = top_input if top_external else None
        self.bottom_input = bottom_input if bottom_external else None
        self.port = port
        self.ridge = ridge
        self.interport_distance = self.init_coupler.interport_distance
        self.waveguide_w = coupler.waveguide_w

    def path(self, flip: bool = False):
        first = self.init_coupler.lower_path.flip() if flip else self.init_coupler.lower_path
        second = self.final_coupler.lower_path.flip() if flip else self.final_coupler.lower_path
        return MultilayerPath(
            waveguide_w=self.init_coupler.waveguide_w,
            sequence=[self.bottom_input.copy, first.copy, self.bottom_arm.copy, second.copy],
            path_layer=self.ridge
        )


class Mesh(Multilayer):
    def __init__(self, mzi: MZI, n: int, triangular: bool = True):
        """Default rectangular mesh, but does triangular mesh if specified
        Note: triangular meshes can self-configure, but rectangular meshes cannot.

        Args:
            mzi:
            n:
            triangular:
        """
        self.mzi = mzi
        self.n = n
        self.triangular = triangular
        num_straight = (n - 1) - np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) - 1 if triangular \
            else np.tile((0, 1), n // 2)[:n]
        n_layers = 2 * n - 3 if triangular else n
        ports = [Port(0, i * mzi.interport_distance) for i in range(n)]

        paths = []
        for idx in range(n):  # waveguides
            cols = []
            for layer in range(n_layers):
                flip = idx == n - 1 or (idx - layer % 2 < n and idx > num_straight[layer]) and (idx + layer) % 2
                path = mzi.copy.path(flip)
                cols.append(path)
            cols.append(mzi.bottom_arm.copy)
            paths.append(MultilayerPath(self.mzi.waveguide_w, cols, self.mzi.ridge).to(ports[idx], 'a0'))

        pattern_to_layer = sum([path.pattern_to_layer for path in paths], [])
        super(Mesh, self).__init__(pattern_to_layer)
        self.port = {
            f'a{i}': Port(0, i * mzi.interport_distance, -np.pi) for i in range(n)
        }.update({
            f'b{i}': Port(self.size[0], i * mzi.interport_distance) for i in range(n)
        })
        self.interport_distance = mzi.interport_distance
        self.waveguide_w = self.mzi.waveguide_w
        # number of straight waveguide in the column
        self.num_straight = num_straight
        self.paths = paths
        self.num_dummy_polys = (len(self.paths[0].wg_path.polys) - 6 * n_layers) / (2 * n_layers + 1)
        self.num_taps = 2 * n_layers + 1
        self.n_layers = n_layers
        self.num_poly_per_col = self.num_dummy_polys * 2 + 6

    @property
    def path_array(self):
        sizes = [0, self.num_dummy_polys + 2] + [self.num_dummy_polys + 3] * (2 * self.n_layers - 1) + [self.num_dummy_polys + 2]
        slices = np.cumsum(sizes, dtype=int)
        return np.array(
            [[path.wg_path.polys[slices[s]:slices[s + 1]]
              for s in range(len(slices) - 1)] for path in self.paths]
        )

    def phase_shifter_array(self, ps_layer: str):
        """

        Args:
            ps_layer: name of the layer for the polygons

        Returns:
            Phase shifter array polygons

        """
        if ps_layer in self.layer_to_pattern:
            return [ps for ps in self.layer_to_pattern[ps_layer].geoms]
        else:
            raise ValueError(f'The phase shifter layer {ps_layer} is not correct '
                             f'or there is no phase shifter in this mesh')


class ThermalPS(Multilayer):
    def __init__(self, waveguide_w: float, ps_w: float, length: float, via: Via, ridge: str, ps_layer: str):
        """Thermal phase shifter (e.g. TiN phase shifter)
        Args:
            waveguide_w: Waveguide width
            ps_w: Phase shifter width
            via: Via to connect heater to the top metal layer
            ridge: Waveguide layer
            ps_layer: Phase shifter layer (e.g. TiN)
        """

        waveguide = Waveguide(waveguide_w=waveguide_w, length=length)
        ps = Waveguide(waveguide_w=ps_w, length=length)
        left_via, right_via = via.copy.align(ps.port['a0'].xy), via.copy.align(ps.port['b0'].xy)

        super(ThermalPS, self).__init__(
            [(waveguide, ridge), (ps, ps_layer)] + left_via.pattern_to_layer + right_via.pattern_to_layer
        )

        self.ps = ps
        self.waveguide = waveguide
        self.port['a0'] = waveguide.port['a0']
        self.port['b0'] = waveguide.port['b0']
        self.port['gnd'] = Port(self.bounds[0], 0, -np.pi)
        self.port['pos'] = Port(self.bounds[1], 0)
        self.wg_path = self.waveguide
