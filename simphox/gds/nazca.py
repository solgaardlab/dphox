from ..typing import Optional, List, Union, Callable, Tuple
from ..constants import AMF_STACK

import nazca as nd
import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Polygon
from matplotlib.collections import PatchCollection


# NOTE: This file currently contains privileged information. Only distribute to those who have signed NDA with AMF.
# This NDA is a MUTUAL CONFIDENTIALITY AGREEMENT signed on 13th March 2020 by Advanced Micro Foundry Pte. Ltd (Co. Reg.
# No. 20170322R) of 11 Science Park Road Singapore 117685 and Olav Solgaard, Stanford University,
# Stanford, CA 94305, USA. Much contribution to this code (aside from authors of this repo) comes from much work done at
# Politecnico de Milano, specifically by Maziyar Milanizadeh. Any further distribution of this work must be done with
# permission of the authors of this file and that of the Polimi group.


class PhotonicChip:
    def __init__(self, process_stack: dict, waveguide_w: float, accuracy: float = 0.001):
        """

        Args:
            process_stack: The stack, in JSON format
            waveguide_w: The width of the waveguide (μm)
        """
        for layer_name in process_stack['layers']:
            nd.add_layer(name=layer_name, layer=process_stack['layers'][layer_name], accuracy=accuracy, overwrite=True)
        for xs_name in process_stack['cross_sections']:
            for layer_dict in process_stack['cross_sections'][xs_name]:
                nd.add_layer2xsection(
                    xsection=xs_name,
                    accuracy=accuracy,
                    overwrite=True,
                    **layer_dict)
        xs = nd.get_xsection('waveguide_xs')
        xs.os = 0.0  # define strt - bend offset
        xs.minimum_radius = 10
        self.min_radius = 10
        xs = nd.get_xsection('heater_xs')
        xs.os = 0.0
        xs = nd.get_xsection('metal_xs')
        xs.os = 0.0
        xs.minimum_radius = 15
        self.waveguide_w = waveguide_w
        self.waveguide_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='waveguide_xs')
        self.slab_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='slab_xs')
        self.grating_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='grating_xs')
        self.heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='heater_xs')
        self.via_heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius,
                                                           xs='via_heater_xs')
        self.pad_ic = nd.interconnects.Interconnect(width=100, radius=self.min_radius, xs='pad_xs')
        self.trench_ic = nd.interconnects.Interconnect(width=10, radius=self.min_radius, xs='trench_xs')
        self.metal_ic = nd.interconnects.Interconnect(width=15, radius=15, xs='metal_xs')
        self.grating = self._grating()

    # Polimi grating (Maziyar Milanizadeh)
    def _grating(self, pitch: float = 0.315, first_radius: float = 24.350,
                 num_ribs: int = 26, waveguide_w: float = 0.5, theta: float = 22.0,
                 grat_l: float = 29.245, grat_w: float = 23.274, last_rib_w: float = 0.55):
        waveguide_ic, slab_ic, grating_ic = self.waveguide_ic, self.slab_ic, self.grating_ic
        with nd.Cell(name='grating') as grating:
            # Local variables
            taper_cord = first_radius * (1 - np.cos(np.radians(theta / 2)))
            taper_offset = waveguide_w / 2 / np.tan(np.radians(theta / 2))  # Offset from center of circles
            final_taper_w = 2 * first_radius * np.sin(np.radians(theta / 2))
            taper_l = first_radius - taper_cord - taper_offset

            grat_sep = 3 * taper_l / 4  # Separation between gratin mask and input port

            # Geometric definition
            # Create taper with circular head
            tpr_cap = nd.Polygon(layer=10, points=nd.geometries.pie(radius=first_radius, angle=theta))
            tpr_cap.put(-taper_offset, 0, -90 + theta / 2)

            current_radius = first_radius + pitch * 2

            # Create grating
            for i in range(num_ribs):
                rib = nd.Polygon(layer=10,
                                 points=nd.geometries.arc(radius=current_radius,
                                                          angle=theta,
                                                          width=pitch))
                rib.put(-taper_offset, 0, 90 - theta / 2)
                previous_radius = current_radius
                current_radius = previous_radius + 2 * pitch

            # Create last rib
            current_radius = current_radius + last_rib_w / 2
            rib = nd.Polygon(layer=10,
                             points=nd.geometries.arc(radius=current_radius,
                                                      angle=theta,
                                                      width=last_rib_w))
            rib.put(-taper_offset, 0, 90 - theta / 2)

            # Create rib and slab regions
            grating_ic.strt(length=grat_l, width=grat_w).put(grat_sep)
            slab_ic.strt(length=grat_l, width=grat_w).put(grat_sep)

        return grating

    def _mzi_angle(self, gap_w: float, interport_w: float, radius: float):
        return np.arccos(1 - (interport_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

    # Polimi heater (Maziyar Milanizadeh)
    def heater(self, heater_l: float, via_w: float = 2):
        heater_ic, via_ic = self.heater_ic, self.via_heater_ic
        with nd.Cell(f'heater_{heater_l}') as heater:
            htr = heater_ic.strt(length=heater_l).put(0, 0)
            via_ic.strt(width=via_w, length=via_w).put(htr.pin['a0'])
            via_ic.strt(width=via_w, length=via_w).put(htr.pin['b0'])
        return heater

    # Polimi bond pad (Maziyar Milanizadeh)
    def bond_pad(self, pad_w: float = 100, pad_l: float = 100):
        with nd.Cell(name='bond_pad') as bond_pad:
            self.pad_ic.strt(length=pad_l, width=pad_w).put()
        return bond_pad

    # Polimi 1d bond pad array (Maziyar Milanizadeh)
    def bond_pad_array(self, n_pads: int, pitch: float = 200,
                       pad_w: float = 100, pad_l: float = 100, labels: Optional[np.ndarray] = None):
        lattice_bond_pads = []
        with nd.Cell(name=f'bond_pad_array_{n_pads}_{pitch}') as bond_pad_array:
            pad = self.bond_pad(pad_w=pad_w, pad_l=pad_l)
            for i in range(n_pads):
                lattice_bond_pads.append(pad.put(i * pitch, 0, 270))
                message = nd.text(text=f'{i + 1 if labels is None else labels[i]}', align='cc', layer=10, height=50)
                message.put(-pitch / 2 + i * pitch, -pad_l / 2)
        return bond_pad_array

    def tap_notch_path(self, angle: float, radius: float):
        self.waveguide_ic.bend(radius=radius, angle=angle / 8).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle / 8).put()
        tap_waveguide = self.waveguide_ic.bend(radius=radius, angle=-angle / 8).put()
        self.waveguide_ic.bend(radius=radius, angle=angle / 8).put()
        return tap_waveguide

    def coupler_path(self, angle: float, interaction_l: float, radius: float, tap_notch: float = 0,
                     interaction_block: bool = False):
        input_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
        if interaction_block:
            x, y = nd.cp.x(), nd.cp.y()
            self.waveguide_ic.bend(radius=radius, angle=-angle).put(x + interaction_l, y)
            output_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        else:
            self.waveguide_ic.strt(length=interaction_l).put()
            self.waveguide_ic.bend(radius=radius, angle=-angle).put()
            output_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        tap_waveguide = output_waveguide
        if tap_notch != 0:
            self.waveguide_ic.strt(length=25).put()
            tap_waveguide = self.tap_notch_path(np.abs(angle) * tap_notch, radius)
            output_waveguide = self.waveguide_ic.strt(length=25).put()  # to make room for trench
        return input_waveguide.pin['a0'], tap_waveguide.pin['a0'], output_waveguide.pin['b0']

    def mzi_path(self, angle: float, arm_l: float, interaction_l: float, radius: float, trench_gap: float,
                 heater: Optional[nd.Cell] = None, trench: Optional[nd.Cell] = None, grating_tap_w: float = 0,
                 tap_notch: float = 0, input_phase_shift: int = 1, interaction_block: bool = False,
                 ignore_internal_sampling: bool = False):
        def put_heater():
            x, y = nd.cp.x(), nd.cp.y()
            if trench:
                trench.put(x, y + trench_gap)
                trench.put(x, y - trench_gap)
            if heater:
                heater.put(x, y)

        with_grating_taps = (grating_tap_w > 0)
        if input_phase_shift:
            if input_phase_shift == 2:
                tap_waveguide = self.tap_notch_path(np.abs(angle) * tap_notch, radius)
                x, y = nd.cp.x(), nd.cp.y()
                if with_grating_taps:
                    self.bidirectional_grating_tap(tap_waveguide.pin['a0'], radius, np.abs(angle) / 4, grating_tap_w)
                self.waveguide_ic.strt(length=25).put(x, y)
            put_heater()
            self.waveguide_ic.strt(length=arm_l).put()

        i_node, l_node, _ = self.coupler_path(angle, interaction_l, radius, tap_notch=(grating_tap_w > 0) or tap_notch,
                                              interaction_block=interaction_block)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        _, r_node, o_node = self.coupler_path(angle, interaction_l, radius, tap_notch=(grating_tap_w > 0) or tap_notch,
                                              interaction_block=interaction_block)
        if with_grating_taps:
            if not ignore_internal_sampling:
                self.bidirectional_grating_tap(l_node, radius, np.abs(angle) / 4, grating_tap_w)
            self.bidirectional_grating_tap(r_node, radius, np.abs(angle) / 4, grating_tap_w)
        return i_node, o_node

    def coupler(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float, radius: float,
                with_gratings: bool = True, interaction_block: bool = False):
        with nd.Cell(name='coupler') as coupler:
            angle = self._mzi_angle(gap_w, interport_w, radius)
            # upper path
            if with_gratings:
                self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(length=arm_l).put(0, 0, 0)
            self.coupler_path(angle, interaction_l, radius, interaction_block=interaction_block)
            self.waveguide_ic.strt(length=arm_l).put()
            if with_gratings:
                self.grating.put()

            # lower path
            if with_gratings:
                self.grating.put(0, interport_w, 180)
            self.waveguide_ic.strt(length=arm_l).put(0, interport_w, 0)
            self.coupler_path(-angle, interaction_l, radius, interaction_block=interaction_block)
            self.waveguide_ic.strt(length=arm_l).put()
            if with_gratings:
                self.grating.put()

        return coupler

    def mzi(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float, radius: float,
            trench_gap: float, with_gratings: bool = True, with_grating_taps: bool = True, tap_notch: float = 1,
            output_phase_shift: bool = False, input_phase_shift: bool = False,
            interaction_block: bool = False):
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        with nd.Cell(name='mzi') as mzi:
            angle = self._mzi_angle(gap_w, interport_w, radius)

            # upper path
            if with_gratings:
                self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, 0, 0)
            _, o_node = self.mzi_path(angle, arm_l, interaction_l, radius,
                                      trench_gap, heater=heater, trench=trench,
                                      grating_tap_w=with_grating_taps * gap_w, tap_notch=tap_notch,
                                      input_phase_shift=input_phase_shift, interaction_block=interaction_block)
            if output_phase_shift:
                heater.put(o_node.x, o_node.y)
                trench.put(o_node.x, o_node.y + trench_gap)
                trench.put(o_node.x, o_node.y - trench_gap)
            self.waveguide_ic.strt(length=arm_l + output_phase_shift * arm_l).put(o_node)

            if with_gratings:
                self.grating.put()

            # lower path
            if with_gratings:
                self.grating.put(0, interport_w, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, interport_w, 0)
            _, o_node = self.mzi_path(-angle, arm_l, interaction_l, radius,
                                      trench_gap, heater=heater, trench=trench,
                                      grating_tap_w=with_grating_taps * gap_w, tap_notch=tap_notch,
                                      input_phase_shift=input_phase_shift,
                                      interaction_block=interaction_block)
            if output_phase_shift:
                heater.put(o_node.x, o_node.y)
                trench.put(o_node.x, o_node.y + trench_gap)
                trench.put(o_node.x, o_node.y - trench_gap)
            self.waveguide_ic.strt(length=arm_l + output_phase_shift * arm_l).put(o_node)
            if with_gratings:
                self.grating.put()

        return mzi

    def equal_bend_mesh(self, directions: np.ndarray, gap_w: float, interaction_l: float, interport_w: float,
                        arm_l: float, radius: float, trench_gap: float, with_grating_taps: bool = False,
                        tap_notch: float = 1, include_end: bool = True, input_phase_shift: bool = True,
                        interaction_block: bool = False):
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        angle = self._mzi_angle(gap_w, interport_w, radius)
        for idx, direction in enumerate(directions):
            self.waveguide_ic.strt(length=arm_l).put(0, interport_w * idx, 0)
            for j, d in enumerate(direction):
                ips = 1 if j > 0 and input_phase_shift == 2 else input_phase_shift
                self.mzi_path(d * angle, arm_l, interaction_l, radius,
                              trench_gap, heater=heater, trench=trench,
                              grating_tap_w=with_grating_taps * gap_w, tap_notch=tap_notch,
                              input_phase_shift=ips,
                              interaction_block=interaction_block)
            if include_end:
                output = self.waveguide_ic.strt(length=2 * arm_l).put()
                o_node = output.pin['a0']
                heater.put(o_node.x, interport_w * idx)
                trench.put(o_node.x, interport_w * idx + trench_gap)
                trench.put(o_node.x, interport_w * idx - trench_gap)


    def splitter_tree_4(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float,
                        radius: float, trench_gap: float, tap_notch: float = 0,
                        input_phase_shift: bool = True, interaction_block: bool = False):
        directions = np.asarray([(1, 1), (-1, 1), (1, -1), (-1, -1)])
        with nd.Cell(name='splitter_tree_4') as binary_tree_4:
            self.equal_bend_mesh(directions, gap_w, interaction_l, interport_w, arm_l, radius, trench_gap,
                                 tap_notch=tap_notch, include_end=False,
                                 input_phase_shift=input_phase_shift,
                                 interaction_block=interaction_block)
        return binary_tree_4

    def splitter_tree_line_4(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float,
                             radius: float, trench_gap: float, tap_notch: float = 0):
        # directions = np.asarray([(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (-1, -1, -1)])
        directions = np.asarray([(1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, -1)])
        with nd.Cell(name='splitter_line_4') as diagonal_line_4:
            self.equal_bend_mesh(directions, gap_w, interaction_l, interport_w, arm_l, radius, trench_gap,
                                 tap_notch=tap_notch)
        return diagonal_line_4

    def splitter_layer_4(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float,
                         radius: float, trench_gap: float, tap_notch: float = 0):
        directions = np.asarray([[1, -1, 1, -1, 1, -1, 1, -1]]).T
        with nd.Cell(name=f'splitter_layer_4') as layer_4:
            self.equal_bend_mesh(directions, gap_w, interaction_l, interport_w, arm_l, radius, trench_gap,
                                 tap_notch=tap_notch, include_end=False, input_phase_shift=False)
        return layer_4

    def trombone(self, height: float, radius: Optional[float] = None):
        radius = self.min_radius if radius is None else radius
        self.waveguide_ic.bend(radius, 90).put()
        self.waveguide_ic.strt(height).put()
        self.waveguide_ic.bend(radius, -180).put()
        self.waveguide_ic.strt(height).put()
        self.waveguide_ic.bend(radius, 90).put()

    def interposer(self, n: int, period: float, final_period: Optional[float] = None, radius: Optional[float] = None,
                   with_gratings: bool = True, horiz_dist: float = 0):
        final_period = period if final_period is None else final_period
        period_diff = final_period - period
        with nd.Cell(name=f'interposer_{n}_{period}_{final_period}') as interposer:
            for idx in range(n):
                radius = period_diff / 2 if not radius else radius
                angle_r = np.sign(period_diff) * np.arccos(1 - np.abs(period_diff) / 4 / radius)
                angled_length = np.abs(period_diff / np.sin(angle_r))
                x_length = np.abs(period_diff / np.tan(angle_r))
                angle = angle_r * 180 / np.pi
                self.waveguide_ic.strt(length=0).put(0, period * idx, 0)
                mid = int(np.ceil(n / 2))
                max_length_diff = (angled_length - x_length) * (mid - 1)
                num_trombones = int(np.ceil(max_length_diff / 2 / (final_period - 3 * self.min_radius)))
                length_diff = (angled_length - x_length) * idx if idx < mid else (angled_length - x_length) * (
                        n - 1 - idx)
                self.waveguide_ic.strt(horiz_dist).put()
                if idx < mid:
                    self.waveguide_ic.bend(radius, -angle).put()
                    self.waveguide_ic.strt(angled_length * (mid - idx - 1)).put()
                    self.waveguide_ic.bend(radius, angle).put()
                    self.waveguide_ic.strt(x_length * (idx + 1)).put()
                else:
                    self.waveguide_ic.bend(radius, angle).put()
                    self.waveguide_ic.strt(angled_length * (mid - n + idx)).put()
                    self.waveguide_ic.bend(radius, -angle).put()
                    self.waveguide_ic.strt(x_length * (n - idx)).put()
                for _ in range(num_trombones):
                    self.trombone(length_diff / 2 / num_trombones)
                self.grating.put()
            if with_gratings:
                x, y = nd.cp.x(), nd.cp.y()
                self.grating.put(x, y - n * final_period)
                self.grating.put(x, y + final_period)
                grating_length = self.grating.bbox[2] - self.grating.bbox[0]
                if radius > 50:
                    radius = self.min_radius  # hack for now
                self.waveguide_ic.bend(radius=radius, angle=-180).put(x, y + final_period, -180)
                self.waveguide_ic.strt(length=grating_length + 5).put()
                self.waveguide_ic.bend(radius=radius, angle=-90).put()
                self.waveguide_ic.strt(length=final_period * (n + 1) + 2 * radius).put()
                self.waveguide_ic.bend(radius=radius, angle=-90).put()
                self.waveguide_ic.strt(length=grating_length + 5).put()
                self.waveguide_ic.bend(radius=radius, angle=-180).put()
        return interposer

    def grating_array(self, n: int, period: float, turn_radius: float = 0, connector_x: float = 0,
                      connector_y: float = 0):
        with nd.Cell(name=f'grating_array_{n}_{period}') as gratings:
            for idx in range(n):
                self.waveguide_ic.strt(length=connector_x).put(0, period * idx)
                if turn_radius > 0:
                    self.waveguide_ic.bend(turn_radius, angle=90).put()
                self.waveguide_ic.strt(length=connector_y).put()
                self.grating.put()
        return gratings

    def sensor_connector(self, n: int, radius: float, curr_period: float, final_period: float,
                         connector_x: float = 0, connector_y: float = 0, sensor_x: float = 0, sensor_y: float = 0,
                         wrap_l=0):
        with nd.Cell(f'sensor_connector_{n}_{radius}') as sensor_connector:
            for idx in range(n):
                if sensor_x > 0 or sensor_y > 0:
                    self.waveguide_ic.bend(radius, angle=-90).put(0, idx * curr_period)
                    self.waveguide_ic.strt(wrap_l).put()
                    self.waveguide_ic.bend(radius, angle=-90).put()
                    self.waveguide_ic.strt(sensor_x + idx * radius).put()
                    self.waveguide_ic.bend(radius, angle=90).put()
                    self.waveguide_ic.strt(sensor_y + idx * (curr_period + radius)).put()
                    self.waveguide_ic.bend(radius, angle=90).put()
                    self.waveguide_ic.strt(connector_x + 2 * idx * radius).put()
                    self.waveguide_ic.bend(radius, angle=90).put()
                    self.waveguide_ic.strt(
                        connector_y + (n - 1 - idx) * (final_period - radius) + 5 * radius + wrap_l).put()
                    self.waveguide_ic.bend(radius, angle=-90).put()
                    self.waveguide_ic.strt((n - 1 - idx) * radius).put()
                else:
                    self.waveguide_ic.strt((n - 1 - idx) * radius).put(0, idx * curr_period)
                    self.waveguide_ic.bend(radius, angle=90).put()
                    self.waveguide_ic.strt((n - 1 - idx) * (curr_period - final_period)).put()
                    self.waveguide_ic.bend(radius, angle=-180).put()
                    self.waveguide_ic.bend(radius, angle=90).put()
                    self.waveguide_ic.strt(connector_x + idx * radius).put()
        return sensor_connector

    def u_connector(self, radius: float, connector_xl: float = 0, connector_xr: float = 0, connector_y: float = 0):
        with nd.Cell(f'u_connector_{radius}') as u_connector:
            self.waveguide_ic.strt(connector_xl).put(0, 0, 0)
            self.waveguide_ic.bend(radius, angle=90).put()
            self.waveguide_ic.strt(connector_y).put()
            self.waveguide_ic.bend(radius, angle=90).put()
            self.waveguide_ic.strt(connector_xl + connector_xr).put()
        return u_connector

    def autoroute_turn(self, n: Union[int, List, np.ndarray], period: float, final_period: float, pin_prefix: str,
                       connector_x: float = 0, connector_y: float = 0, turn_radius: float = 0, overlap: float = 1):
        with nd.Cell(f'autoroute_turn_{period}_{final_period}') as autoroute_turn:
            route_arr = np.ones(n) if isinstance(n, int) else np.asarray(n)
            route_num = 0
            for m, route_idx in enumerate(route_arr):
                if route_idx > 0:
                    self.metal_ic.strt(length=overlap).put(-overlap, m * period)
                    if turn_radius > 0:
                        self.metal_ic.bend(radius=turn_radius, angle=90).put()
                        self.metal_ic.bend(radius=turn_radius, angle=-90).put()
                    # self.metal_ic.strt(route_num * final_period + connector_x - turn_radius * 2).put()
                    # self.metal_ic.bend(radius=15, angle=-90).put()
                    self.metal_ic.strt(route_num * final_period + connector_x - turn_radius * 2 + 15).put()
                    # self.metal_ic.bend(radius=15, angle=-90).put()
                    self.metal_ic.strt(connector_y + 22.5).put(nd.cp.x(), nd.cp.y() + 7.5, -90)
                    output = self.metal_ic.strt(m * period + turn_radius * 2).put()
                    nd.Pin(f'{pin_prefix}{int(route_num)}', pin=output.pin['b0']).put()
                    route_num += 1
            nd.put_stub([], length=0)
        return autoroute_turn

    def autoroute_interposer(self, end_x: np.ndarray, start_y: float, end_y: float, widths: np.ndarray,
                             start_length: float, end_lengths: np.ndarray, period: float = 100,
                             stagger_offset: float = 200, p2p_radius: float = 30):
        with nd.Cell(f'autoroute_interposer_{period}') as autoroute_interposer:
            for idx, xwl in enumerate(zip(end_x, widths, end_lengths)):
                x, width, length = xwl
                start = self.metal_ic.strt(width=width, length=stagger_offset * (1 - idx % 2)).put(
                    idx * period, start_y + stagger_offset * (idx % 2), 90)
                self.metal_ic.bend_strt_bend_p2p(start.pin['b0'], (x, end_y, 90),
                                                 width=width, radius=p2p_radius, length1=start_length,
                                                 length2=length).put()
        return autoroute_interposer

    def bidirectional_grating_tap(self, node: nd.Node, radius: float, angle: float, gap_w: float):
        vert_angle = 90 - angle if angle > 0 else -90 + angle
        y_offset = self.waveguide_w + gap_w
        self.waveguide_ic.bend(radius=radius, angle=angle).put(node.x, node.y + y_offset)
        self.waveguide_ic.bend(radius=self.min_radius, angle=vert_angle).put()
        self.grating.put(nd.cp.x(), nd.cp.y(), 90 * np.sign(angle))
        self.waveguide_ic.bend(radius=radius, angle=-angle).put(node.x, node.y + y_offset, -180)
        self.waveguide_ic.bend(radius=self.min_radius, angle=-vert_angle).put()
        self.grating.put(nd.cp.x(), nd.cp.y(), 90 * np.sign(angle))

    def triangular_mesh(self, n: int, arm_l: float, gap_w: float, interaction_l: float, interport_w: float,
                        radius: float,
                        trench_gap: float, with_grating_taps: float = 1, tap_notch: float = 1,
                        ignore_internal_sampling: bool = False):

        num_straight = (n - 1) - (np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) + 1)
        bend_angle = self._mzi_angle(gap_w, interport_w, radius)

        with nd.Cell(name=f'triangular_mesh_{n}_{with_grating_taps}') as triangular_mesh:
            heater = self.heater(heater_l=arm_l)
            trench = self.trench_ic.strt(length=arm_l)
            for idx in range(n):
                self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, interport_w * idx, 0)
                for layer in range(2 * n - 3):
                    angle = -bend_angle if idx == n - 1 or (idx - layer % 2 < n and idx > num_straight[layer]) and (
                            idx + layer) % 2 else bend_angle
                    i_node, o_node = self.mzi_path(angle, arm_l, interaction_l, radius, trench_gap, heater=heater,
                                                   trench=trench, grating_tap_w=with_grating_taps * gap_w,
                                                   tap_notch=tap_notch, ignore_internal_sampling=ignore_internal_sampling,
                                                   input_phase_shift=2 if layer == 0 else 1)
                    self.waveguide_ic.strt(length=0).put(o_node)
                output = self.waveguide_ic.strt(length=2 * arm_l).put()
                o_node = output.pin['a0']
                heater.put(o_node.x, interport_w * idx)
                trench.put(o_node.x, interport_w * idx + trench_gap)
                trench.put(o_node.x, interport_w * idx - trench_gap)

        return triangular_mesh

    def cutback_mzi_test(self, n: int, arm_l: float, gap_w: float, interaction_l: float, interport_w: float,
                         radius: float, trench_gap: float, with_grating_taps: float = 1, tap_notch: float = 1):
        angle = self._mzi_angle(gap_w, interport_w, radius)
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        with nd.Cell(name=f'cutback_mzi_test_{n}_{with_grating_taps}') as cutback_mzi_test:
            for idx in range(n):
                self.waveguide_ic.strt(length=0).put(0, interport_w * idx, 0)
                for _ in range(idx + 1):
                    self.waveguide_ic.strt(length=0).put()
                    _, o_node = self.mzi_path(angle, arm_l, interaction_l, radius, trench_gap, heater=heater,
                                              trench=trench,
                                              grating_tap_w=with_grating_taps * gap_w, tap_notch=tap_notch)
                    self.waveguide_ic.strt(length=0).put(o_node)
                self.waveguide_ic.bend(radius=10, angle=90).put()
                self.grating.put()
        return cutback_mzi_test

    def cutback_coupler_test(self, n: int, gap_w: float, interaction_l: float, interport_w: float, radius: float,
                             turn: bool = False, factor: int = 2):
        angle = self._mzi_angle(gap_w, interport_w, radius)
        with nd.Cell(name=f'cutback_mzi_test_{n}') as cutback_coupler_test:
            for idx in range(n):
                self.waveguide_ic.strt(length=0).put(0, interport_w * idx, 0)
                for _ in range(idx + 1):
                    for _ in range(factor):
                        self.coupler_path(angle, interaction_l, radius)
                if turn:
                    self.waveguide_ic.bend(radius=10, angle=90).put()
                self.grating.put()
        return cutback_coupler_test

    def sampling_test(self, gap_ws: Union[List[float], np.ndarray], arm_l: float, gap_w, interaction_l: float,
                      interport_w: float, radius: float, trench_gap: float, with_grating_taps: float = 1):
        angle = self._mzi_angle(gap_w, interport_w, radius)
        with nd.Cell(name=f'sampling_test') as sampling_test:
            for idx, tap_gap_w in enumerate(gap_ws):
                self.waveguide_ic.strt(length=0).put(0, interport_w * idx, 0)
                _, o_node = self.mzi_path(angle, arm_l, interaction_l, radius, trench_gap=trench_gap,
                                          grating_tap_w=with_grating_taps * tap_gap_w, tap_notch=1)
                self.waveguide_ic.strt(length=arm_l).put(o_node)
        return sampling_test

    def chiplet_trench(self, y_lines: List[float], x_lines: List[float] = (50, 2950),
                       x_line_w: float = 100, y_line_w: float = 200, tot_height: float = 2800,
                       tot_width: float = 16000):
        with nd.Cell(name=f'chiplet_trench') as chiplet_trench:
            for y in x_lines:
                self.trench_ic.strt(width=x_line_w, length=tot_width).put(0, y)
            for x in y_lines:
                self.trench_ic.strt(width=y_line_w, length=tot_height).put(x, x_line_w, 90)
        return chiplet_trench

    def drop_port_array(self, n: Union[int, List[int]], period: float, final_taper_width: float):
        with nd.Cell(name=f'drop_port_array_{n}_{period}') as drop_port_array:
            idxs = np.arange(n) if isinstance(n, int) else n
            for idx in idxs:
                self.waveguide_ic.taper(length=50, width1=self.waveguide_w,
                                        width2=final_taper_width).put(0, period * float(idx))
                self.waveguide_ic.bend(radius=10, angle=180, width=final_taper_width).put()
        return drop_port_array


class NazcaVisualizer:
    def __init__(self):
        self.poly_generator = nd.Netlist().celltree_iter
        self.bbox = [1e6, 1e6, -1e6, -1e6]
        self.layer2color = {
            'waveguide': 'black',
            'grating': 'purple',
            'via': 'brown',
            'via_heater': 'yellow',
            'mt_heater': 'green',
            'heater': 'red',
            'slab': 'orange',
            'pad': 'blue',
            'ox_open': 'cyan',
            'trench': 'gray'
        }
        self.patches = []

    def add_cell(self, cell: nd.Cell):
        for params in self.poly_generator(cell, hierarchy='flat'):
            for poly, xy, bbox in params.iters["polygon"]:
                xy = np.asarray(xy)
                layer = poly.layer
                if layer in self.layer2color:
                    self.patches.append(PolygonPatch(Polygon(xy), fc=self.layer2color[layer], ec='none'))
                if self.bbox[0] > bbox[0]:
                    self.bbox[0] = bbox[0]
                if self.bbox[2] < bbox[2]:
                    self.bbox[2] = bbox[2]
                if self.bbox[1] > bbox[1]:
                    self.bbox[1] = bbox[1]
                if self.bbox[3] < bbox[3]:
                    self.bbox[3] = bbox[3]

    def plot(self, ax):
        p = PatchCollection(self.patches, match_original=True)
        ax.add_collection(p)
        ax.set_xlim(self.bbox[0], self.bbox[2])
        ax.set_ylim(self.bbox[1], self.bbox[3])
        ax.set_aspect('equal')
