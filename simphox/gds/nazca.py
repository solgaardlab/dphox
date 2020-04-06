from ..typing import Optional, List, Union, Callable, Tuple
from ..constants import AMF_STACK

import nazca as nd
import numpy as np


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
                message = nd.text(text=f'{i if labels is None else labels[i]}', align='cc', layer=10, height=50)
                message.put(pitch / 2 + i * pitch, -pad_l / 2)
        return bond_pad_array

    def tap_notch_path(self, angle: float, radius: float):
        self.waveguide_ic.bend(radius=radius, angle=angle / 8).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle / 8).put()
        tap_waveguide = self.waveguide_ic.bend(radius=radius, angle=-angle / 8).put()
        self.waveguide_ic.bend(radius=radius, angle=angle / 8).put()
        return tap_waveguide

    def coupler_path(self, angle: float, interaction_l: float, radius: float, tap_notch: float = 0):
        input_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
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
                 tap_notch: float = 0):
        def put_heater():
            x, y = nd.cp.x(), nd.cp.y()
            if trench:
                trench.put(x, y + trench_gap)
                trench.put(x, y - trench_gap)
            if heater:
                heater.put(x, y)

        with_grating_taps = (grating_tap_w > 0)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        i_node, l_node, _ = self.coupler_path(angle, interaction_l, radius, tap_notch=(grating_tap_w > 0) or tap_notch)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        _, r_node, o_node = self.coupler_path(angle, interaction_l, radius, tap_notch=(grating_tap_w > 0) or tap_notch)
        if with_grating_taps:
            self.bidirectional_grating_tap(l_node, radius, np.abs(angle) / 4, grating_tap_w)
            self.bidirectional_grating_tap(r_node, radius, np.abs(angle) / 4, grating_tap_w)
        return i_node, o_node

    def coupler(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float, radius: float,
                with_gratings: bool = True):
        with nd.Cell(name='coupler') as coupler:
            angle = self._mzi_angle(gap_w, interport_w, radius)
            # upper path
            if with_gratings:
                self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(length=arm_l).put(0, 0, 0)
            self.coupler_path(angle, interaction_l, radius)
            self.waveguide_ic.strt(length=arm_l).put()
            if with_gratings:
                self.grating.put()

            # lower path
            if with_gratings:
                self.grating.put(0, interport_w, 180)
            self.waveguide_ic.strt(length=arm_l).put(0, interport_w, 0)
            self.coupler_path(-angle, interaction_l, radius)
            self.waveguide_ic.strt(length=arm_l).put()
            if with_gratings:
                self.grating.put()

        return coupler

    def mzi(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float, radius: float,
            trench_gap: float, with_gratings: bool = True, with_grating_taps: bool = True, tap_notch: float = 1,
            output_phase_shift: bool = False):
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
                                      grating_tap_w=with_grating_taps * gap_w, tap_notch=tap_notch)
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
                                      grating_tap_w=with_grating_taps * gap_w, tap_notch=tap_notch)
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
                        tap_notch: float = 1):
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        angle = self._mzi_angle(gap_w, interport_w, radius)
        for idx, direction in enumerate(directions):
            self.waveguide_ic.strt(length=arm_l).put(0, interport_w * idx, 0)
            for d in direction:
                self.mzi_path(d * angle, arm_l, interaction_l, radius,
                              trench_gap, heater=heater, trench=trench,
                              grating_tap_w=with_grating_taps * gap_w, tap_notch=tap_notch)
            output = self.waveguide_ic.strt(length=2 * arm_l).put()
            o_node = output.pin['a0']
            heater.put(o_node.x, interport_w * idx)
            trench.put(o_node.x, interport_w * idx + trench_gap)
            trench.put(o_node.x, interport_w * idx - trench_gap)

    def splitter_tree_4(self, gap_w: float, interaction_l: float, interport_w: float, arm_l: float,
                        radius: float, trench_gap: float, tap_notch: float = 0):
        directions = np.asarray([(1, 1), (-1, 1), (1, -1), (-1, -1)])
        with nd.Cell(name='splitter_tree_4') as binary_tree_4:
            self.equal_bend_mesh(directions, gap_w, interaction_l, interport_w, arm_l, radius, trench_gap,
                                 tap_notch=tap_notch)
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
                                 tap_notch=tap_notch)
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
                       connector_x: float = 0, connector_y: float = 0, turn_offset: bool = False):
        with nd.Cell(f'autoroute_turn_{period}_{final_period}') as autoroute_turn:
            route_arr = np.ones(n) if isinstance(n, int) else np.asarray(n)
            route_num = 0
            # tot_routes = np.sum(route_arr)
            for m, route_idx in enumerate(route_arr):
                if route_idx > 0:
                    self.metal_ic.strt(length=0).put(0, m * period)
                    if turn_offset:
                        self.metal_ic.bend(radius=period / 4, angle=-90).put()
                        self.metal_ic.bend(radius=period / 4, angle=90).put()
                    self.metal_ic.strt(route_num * final_period + connector_x - turn_offset * period / 2).put()
                    self.metal_ic.bend(radius=15, angle=-90).put()
                    self.metal_ic.strt(connector_y).put()
                    output = self.metal_ic.strt(m * period - turn_offset * period / 2).put()
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
                        trench_gap: float, with_grating_taps: float = 1, tap_notch: float = 1):

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
                                                   tap_notch=tap_notch)
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

    def cutback_coupler_test(self, n: int, gap_w: float, interaction_l: float, interport_w: float, radius: float):
        angle = self._mzi_angle(gap_w, interport_w, radius)
        with nd.Cell(name=f'cutback_mzi_test_{n}') as cutback_mzi_test:
            for idx in range(n):
                self.waveguide_ic.strt(length=0).put(0, interport_w * idx, 0)
                for _ in range(idx + 1):
                    self.coupler_path(angle, interaction_l, radius)
                self.waveguide_ic.bend(radius=10, angle=90).put()
                self.grating.put()
        return cutback_mzi_test

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
                       x_line_w: float = 100, y_line_w: float = 200, tot_height: float = 3000,
                       tot_width: float = 16000):
        with nd.Cell(name=f'chiplet_trench') as chiplet_trench:
            for y in x_lines:
                self.trench_ic.strt(width=x_line_w, length=tot_width).put(0, y)
            for x in y_lines:
                self.trench_ic.strt(width=y_line_w, length=tot_height).put(x, 0, 90)
        return chiplet_trench


if __name__ == 'main':
    waveguide_w = 0.5
    interport_w = 70
    arm_l = 80
    gap_w = 0.3
    interaction_l = 40
    cp_radius = 35
    trench_gap = 10

    mzi_kwargs = {
        'gap_w': gap_w,
        'interaction_l': interaction_l,
        'interport_w': interport_w,
        'arm_l': arm_l,
        'radius': cp_radius,
        'trench_gap': trench_gap
    }

    chip = PhotonicChip(AMF_STACK, waveguide_w)
    CHIPLET_SEP = 200

    with nd.Cell('meshes_chiplet') as meshes_chiplet:
        # useful constants
        cp_len = 308.937
        min_dist = 163.5
        ground_dist = 87

        # triangular meshes

        mesh = chip.triangular_mesh(n=6, **mzi_kwargs).put(0, 0)
        chip.triangular_mesh(n=6, **mzi_kwargs, with_grating_taps=False).put(0, 15 * interport_w, flip=True)

        # small test structures

        splitter_tree = chip.splitter_tree_4(**mzi_kwargs, tap_notch=1)
        splitter_tree.put(0, 6 * interport_w)
        mzi_no_tap = chip.mzi(**mzi_kwargs, with_grating_taps=False, with_gratings=False, output_phase_shift=True)
        mzi_tap = chip.mzi(**mzi_kwargs, with_grating_taps=True, with_gratings=False, output_phase_shift=True)
        mzi_no_tap.put(mesh.bbox[2] - mzi_no_tap.bbox[2], 6 * interport_w)
        mzi_tap.put(mesh.bbox[2] - mzi_tap.bbox[2], 8 * interport_w)

        gratings_turn = chip.grating_array(4, period=70, connector_x=20, turn_radius=10)
        gratings_turn.put(splitter_tree.bbox[2], 6 * interport_w)
        gratings_turn.put(mesh.bbox[2] - mzi_no_tap.bbox[2], 6 * interport_w, flop=True)

        gratings_3 = chip.grating_array(3, period=70, connector_x=20, turn_radius=10)  # length=20)
        gratings_4 = chip.grating_array(4, period=70, connector_x=10)

        chip.cutback_mzi_test(3, **mzi_kwargs, with_grating_taps=False).put(cp_len * 5 + arm_l, 7 * interport_w)
        gratings_3.put(cp_len * 5 + arm_l, 7 * interport_w, flop=True)
        chip.sampling_test(gap_ws=[0.3, 0.35, 0.4, 0.45], **mzi_kwargs).put(cp_len * 13 + arm_l, 6 * interport_w)
        gratings_turn.put(cp_len * 13 + arm_l, 6 * interport_w, flop=True)
        gratings_turn.put(cp_len * 15 + 2 * arm_l, 6 * interport_w)

        # gratings and interposers

        interposer = chip.interposer(16, period=70, final_period=127, radius=75, horiz_dist=300)
        interposer.put(0, 0, flop=True)
        interposer.put(mesh.bbox[2])

        connection_array = np.asarray([
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ])

        end_x = []
        end_lengths = []
        widths = []

        num_layers = connection_array.shape[0]

        for idx, cxns in enumerate(connection_array):
            num_cxns = np.sum(cxns)
            start_x = min_dist + idx * cp_len
            chip.autoroute_turn(cxns, period=70, final_period=20,
                                pin_prefix=f'v{idx // 2}.{idx % 2}.').put(start_x, 0)
            chip.autoroute_turn(cxns, period=70, final_period=15,
                                pin_prefix=f'g{idx // 2}.{idx % 2}.', connector_x=65).put(start_x - ground_dist, 0,
                                                                                          flop=True)
            chip.autoroute_turn(cxns, period=70, final_period=20,
                                pin_prefix=f'v{idx // 2}.{idx % 2}.').put(start_x, 15 * interport_w, flip=True)
            chip.autoroute_turn(cxns, period=70, final_period=15,
                                pin_prefix=f'g{idx // 2}.{idx % 2}.', connector_x=65).put(start_x - ground_dist,
                                                                                          15 * interport_w,
                                                                                          flop=True, flip=True)
            end_x.append(start_x - (num_cxns + 1) * 7.5 - 65 - ground_dist)
            end_x.append(start_x + np.arange(num_cxns) * 20 + 15)
            if idx < num_layers // 2:
                end_lengths.append((num_cxns + 1) * 10)
                end_lengths.append(np.arange(num_cxns) * 10)
            else:
                end_lengths.append(0)
                end_lengths.append(np.flipud(np.arange(num_cxns)) * 10)
            widths.append(num_cxns * 15)
            widths.append(np.ones(num_cxns) * 15)

        end_x, end_lengths, widths = np.hstack(end_x), np.hstack(end_lengths), np.hstack(widths)

        # bond pad arrays
        pad_start = -850
        bond_pad_array_top = chip.bond_pad_array(n_pads=38)
        bond_pad_array_bot = chip.bond_pad_array(n_pads=37)
        bond_pad_array_bot.put(pad_start + 100, -7 * interport_w)
        bond_pad_array_top.put(pad_start, -7 * interport_w - 200)
        bond_pad_array_bot.put(pad_start + 100, 23 * interport_w + 30)
        bond_pad_array_top.put(pad_start, 23 * interport_w + 230)

        pad_routes = chip.autoroute_interposer(end_x=end_x - pad_start - 200, widths=widths,
                                               start_length=100,
                                               end_lengths=end_lengths, start_y=-7 * interport_w - 200, end_y=-15)

        pad_routes.put(pad_start + 200, 0)
        pad_routes.put(pad_start + 200, 15 * interport_w, flip=True)

    meshes_chiplet.put(CHIPLET_SEP / 2 - meshes_chiplet.bbox[0], 200 - meshes_chiplet.bbox[1])

    meshes_chiplet_x = meshes_chiplet.bbox[2] - meshes_chiplet.bbox[0] + CHIPLET_SEP / 2

    with nd.Cell('sensor_and_inv_design_chiplet') as sensor_and_inv_design_chiplet:
        # sensor network and splitter trees

        radius = 10
        cp_len = 317.282
        ground_dist = 87
        inter_heater_dist = 240.782

        sensor_layer = chip.splitter_layer_4(**mzi_kwargs, tap_notch=0).put()
        splitter_tree_dist = sensor_layer.bbox[2] + 9 * radius
        splitter_tree = chip.splitter_tree_4(**mzi_kwargs, tap_notch=0)
        splitter_tree.put(splitter_tree_dist)
        splitter_tree.put(splitter_tree_dist, 4 * interport_w)

        # test structures
        splitter_tree_x = sensor_layer.bbox[2] + 9 * radius
        splitter_tree.put(splitter_tree_x, 11 * interport_w)
        mzi = chip.mzi(**mzi_kwargs, with_grating_taps=False, with_gratings=False, output_phase_shift=True, tap_notch=0)
        mzi.put(0, 11 * interport_w)
        chip.cutback_coupler_test(4, gap_w, interaction_l, interport_w, cp_radius).put(-500, 11 * interport_w)
        coupler = chip.coupler(gap_w, interaction_l, interport_w, arm_l, cp_radius, with_gratings=False)
        coupler.put(cp_len, 13 * interport_w)

        # gratings and interposers
        interposer_8 = chip.interposer(6, period=140, final_period=127, radius=32, horiz_dist=500)
        interposer_8.put(0, 0, flop=True)

        gratings_turn_4 = chip.grating_array(4, period=70, turn_radius=10)
        gratings_4 = chip.grating_array(4, period=70)
        gratings_turn_2 = chip.grating_array(2, period=70, turn_radius=10)
        gratings_2 = chip.grating_array(2, period=70)
        gratings_turn_4.put(splitter_tree_x + splitter_tree.bbox[2], 11 * interport_w)
        gratings_4.put(splitter_tree_x, 11 * interport_w, flop=True)
        gratings_4.put(-500, 11 * interport_w, flop=True)
        gratings_turn_2.put(mzi.bbox[2], 11 * interport_w)
        gratings_2.put(0, 11 * interport_w, flop=True)
        gratings_turn_2.put(coupler.bbox[2] + cp_len, 13 * interport_w)
        gratings_2.put(cp_len, 13 * interport_w, flop=True)

        # sensor-to-fiber connectors

        sensor_connector = chip.sensor_connector(n=4, radius=radius, wrap_l=5, curr_period=140, final_period=70,
                                                 sensor_x=sensor_layer.bbox[2] + 685, sensor_y=0,
                                                 connector_x=sensor_layer.bbox[2] + 685 + 4 * radius,
                                                 connector_y=0)
        mux_connector = chip.sensor_connector(n=4, radius=radius, curr_period=140, final_period=70,
                                              connector_x=4 * radius, connector_y=0)
        u_connector_top = chip.u_connector(radius=radius, connector_xl=4 * radius,
                                           connector_xr=splitter_tree_dist + splitter_tree.bbox[2],
                                           connector_y=2 * interport_w - 2 * radius)
        u_connector_bot = chip.u_connector(radius=radius, connector_xl=5 * radius,
                                           connector_xr=splitter_tree_dist + splitter_tree.bbox[2],
                                           connector_y=8 * interport_w - 2 * radius)

        sensor_connector.put(sensor_layer.bbox[2], 0)
        mux_connector.put(sensor_layer.bbox[2], interport_w)
        u_connector_bot.put(splitter_tree_dist + splitter_tree.bbox[2], 2 * interport_w)
        u_connector_top.put(splitter_tree_dist + splitter_tree.bbox[2], 6 * interport_w)

        # traces
        cxns = [1, 0, 1, 0, 1, 0, 1, 0]
        cxns_tree = [0, 1, 0, 0, 0, 1, 0, 0]

        start_splitter_x = splitter_tree_x + 2 * arm_l + 3.5

        end_x = []
        end_lengths = []
        widths = []

        chip.autoroute_turn(cxns, period=70, final_period=25, pin_prefix=f'v0.').put(arm_l + cp_len + 7, 0)
        chip.autoroute_turn(cxns, period=70, final_period=15, pin_prefix=f'g0.').put(arm_l + cp_len + 7 - ground_dist,
                                                                                     0, flop=True)
        end_x.extend([arm_l + cp_len + 7 - ground_dist - 37.5, arm_l + cp_len + 7 + 25 * np.arange(4) + 15])
        end_lengths.extend([0, 25 * np.arange(4)])
        widths.extend([60, np.ones(4) * 15])
        chip.autoroute_turn(cxns, period=70, final_period=15, pin_prefix=f'v1.').put(start_splitter_x, 0)
        chip.autoroute_turn(cxns, period=70, final_period=25, pin_prefix=f'g1.').put(start_splitter_x - ground_dist, 0,
                                                                                     flop=True)
        chip.autoroute_turn(cxns, period=70, final_period=25, pin_prefix=f'v2.').put(
            start_splitter_x + inter_heater_dist, 0)
        chip.autoroute_turn(cxns, period=70, final_period=15, pin_prefix=f'g2.').put(
            start_splitter_x + inter_heater_dist - ground_dist, 0, flop=True)
        end_x.extend([start_splitter_x - 25 * np.flipud(np.arange(4)) - ground_dist - 15, start_splitter_x + 37.5,
                      start_splitter_x + inter_heater_dist - 37.5 - ground_dist,
                      start_splitter_x + inter_heater_dist + 25 * np.arange(4) + 15])
        end_lengths.extend([25 * np.arange(4), 0, 0, 25 * np.flipud(np.arange(4))])
        widths.extend([np.ones(4) * 15, 60, 60, np.ones(4) * 15])
        chip.autoroute_turn(cxns_tree, period=70, final_period=25, pin_prefix=f'v3.').put(
            start_splitter_x + inter_heater_dist * 2, 0)
        chip.autoroute_turn(cxns_tree, period=70, final_period=15, pin_prefix=f'g3.').put(
            start_splitter_x + inter_heater_dist * 2 - ground_dist, 0, flop=True)
        chip.autoroute_turn(cxns_tree, period=70, final_period=25, pin_prefix=f'v4.').put(
            start_splitter_x + inter_heater_dist * 3, 0)
        chip.autoroute_turn(cxns_tree, period=70, final_period=15, pin_prefix=f'g4.').put(
            start_splitter_x + inter_heater_dist * 3 - ground_dist, 0, flop=True)
        end_x.extend([start_splitter_x + inter_heater_dist * 2 - ground_dist - 22.5,
                      start_splitter_x + inter_heater_dist * 2 + 25 * np.arange(2) + 15,
                      start_splitter_x + inter_heater_dist * 3 - ground_dist - 22.5,
                      start_splitter_x + inter_heater_dist * 3 + 25 * np.arange(2) + 15])
        end_lengths.extend([0, 25 * np.flipud(np.arange(2)), 0, 25 * np.flipud(np.arange(2))])
        widths.extend([30, np.ones(2) * 15, 30, np.ones(2) * 15])

        # bond pad arrays
        end_x, end_lengths, widths = np.hstack(end_x), np.hstack(end_lengths), np.hstack(widths)
        pad_start = 100
        bond_pad_array_top = chip.bond_pad_array(n_pads=11)
        bond_pad_array_bot = chip.bond_pad_array(n_pads=10)
        bond_pad_array_bot.put(pad_start + 100, -3 * interport_w)
        bond_pad_array_top.put(pad_start, -3 * interport_w - 200)
        bond_pad_array_bot.put(pad_start + 100, 18 * interport_w + 30)

        pad_routes = chip.autoroute_interposer(end_x=end_x - pad_start, widths=widths, start_length=0,
                                               end_lengths=end_lengths, start_y=-3 * interport_w - 200, end_y=-15)

        pad_routes.put(pad_start, 0)

    sensor_and_inv_design_chiplet_x = sensor_and_inv_design_chiplet.bbox[2] - sensor_and_inv_design_chiplet.bbox[
        0] + meshes_chiplet_x + 2 * CHIPLET_SEP

    sensor_and_inv_design_chiplet.put(sensor_and_inv_design_chiplet_x - sensor_and_inv_design_chiplet.bbox[2],
                                      -sensor_and_inv_design_chiplet.bbox[1] + 200)

    chip.chiplet_trench([meshes_chiplet_x + CHIPLET_SEP, sensor_and_inv_design_chiplet_x + CHIPLET_SEP]).put(0, 0)

    nd.export_gds(filename='amf.gds')
