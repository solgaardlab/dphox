from ..typing import Optional, List
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

        self.waveguide_w = waveguide_w
        self.waveguide_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='waveguide_xs')
        self.slab_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='slab_xs')
        self.grating_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='grating_xs')
        self.heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius, xs='heater_xs')
        self.via_heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=self.min_radius,
                                                           xs='via_heater_xs')
        self.pad_ic = nd.interconnects.Interconnect(width=100, radius=self.min_radius, xs='pad_xs')
        self.trench_ic = nd.interconnects.Interconnect(width=10, radius=self.min_radius, xs='trench_xs')

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
            in_tpr = waveguide_ic.taper(length=taper_l,
                                        width1=waveguide_w,
                                        width2=final_taper_w).put(0, 0)
            tpr_cap = nd.Polygon(layer=10,
                                 points=nd.geometries.pie(radius=first_radius,
                                                          angle=theta))

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
                       pad_w: float = 100, pad_l: float = 100):
        lattice_bond_pads = []
        with nd.Cell(name=f'bond_pad_array_{n_pads}_{pitch}') as bond_pad_array:
            pad = self.bond_pad(pad_w=pad_w, pad_l=pad_l)
            for i in range(n_pads):
                lattice_bond_pads.append(pad.put(i * pitch, 0, 270))
                message = nd.text(text=f'{i}', align='cc', layer=10, height=50)
                message.put(pitch / 2 + i * pitch, -pad_l / 2)
        return bond_pad_array

    def tap_notch_path(self, angle: float, radius: float):
        self.waveguide_ic.bend(radius=radius, angle=angle / 8).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle / 8).put()
        tap_waveguide = self.waveguide_ic.bend(radius=radius, angle=-angle / 8).put()
        self.waveguide_ic.bend(radius=radius, angle=angle / 8).put()
        return tap_waveguide

    def coupler_path(self, angle: float, interaction_l: float, radius: float, tap_notch: float = False):
        input_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
        self.waveguide_ic.strt(length=interaction_l).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
        output_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        tap_waveguide = output_waveguide
        if tap_notch:
            tap_waveguide = self.tap_notch_path(np.abs(angle), radius)
            output_waveguide = self.waveguide_ic.strt(length=25).put()  # to make room for trench
        return input_waveguide.pin['a0'], tap_waveguide.pin['a0'], output_waveguide.pin['b0']

    def mzi_path(self, angle: float, arm_l: float, interaction_l: float, radius: float, trench_gap: float,
                 heater: Optional[nd.Cell] = None, trench: Optional[nd.Cell] = None, with_grating_taps: bool = True):
        def put_heater():
            x, y = nd.cp.x(), nd.cp.y()
            if trench:
                trench.put(x, y + trench_gap)
                trench.put(x, y - trench_gap)
            if heater:
                heater.put(x, y)

        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        i_node, lt_node, l_node = self.coupler_path(angle, interaction_l, radius, tap_notch=with_grating_taps)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        r_node, ot_node, o_node = self.coupler_path(angle, interaction_l, radius, tap_notch=with_grating_taps)
        return i_node, l_node, lt_node, r_node, ot_node, o_node

    def mzi(self, gap_w: float, interaction_l: float, mzi_w: float, arm_l: float, radius: float,
            trench_gap: float, with_grating_taps: bool = False):
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        with nd.Cell(name='mzi') as mzi:
            angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

            # upper path
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, 0, 0)
            self.mzi_path(angle, arm_l, interaction_l, radius,
                          trench_gap, heater=heater, trench=trench, with_grating_taps=with_grating_taps)
            self.waveguide_ic.strt(length=arm_l).put()
            self.grating.put()

            # lower path
            self.grating.put(0, mzi_w, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w, 0)
            self.mzi_path(-angle, arm_l, interaction_l, radius,
                          trench_gap, heater=heater, trench=trench, with_grating_taps=with_grating_taps)
            self.waveguide_ic.strt(length=arm_l).put()
            self.grating.put()

        return mzi

    def equal_bend_mesh(self, directions, gap_w: float, interaction_l: float, mzi_w: float, arm_l: float,
                        radius: float, trench_gap: float, with_grating_taps: bool = True):
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi
        for idx, direction in enumerate(directions):
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w * idx, 0)
            for d in direction:
                self.mzi_path(d * angle, arm_l, interaction_l, radius,
                              trench_gap, heater=heater, trench=trench,
                              with_grating_taps=with_grating_taps)
            self.waveguide_ic.strt(length=arm_l).put()

    def splitter_tree_4(self, gap_w: float, interaction_l: float, mzi_w: float, arm_l: float,
                        radius: float, trench_gap: float):
        directions = np.asarray([(1, 1), (-1, 1), (1, -1), (-1, -1)])
        with nd.Cell(name='splitter_tree_4') as binary_tree_4:
            self.equal_bend_mesh(directions, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap)
        return binary_tree_4

    def splitter_line_4(self, gap_w: float, interaction_l: float, mzi_w: float, arm_l: float,
                        radius: float, trench_gap: float):
        directions = np.asarray([(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (-1, -1, -1)])
        with nd.Cell(name='splitter_line_4') as diagonal_line_4:
            self.equal_bend_mesh(directions, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap)
        return diagonal_line_4

    def splitter_layer_4(self, gap_w: float, interaction_l: float, mzi_w: float, arm_l: float,
                        radius: float, trench_gap: float):
        directions = np.asarray([[1, -1, 1, -1, 1, -1, 1, -1]]).T
        with nd.Cell(name=f'splitter_layer_4') as layer_4:
            self.equal_bend_mesh(directions, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap)
        return layer_4

    def grating_array(self, n: int, period: float, final_period: Optional[float] = None):
        final_period = period if final_period is None else final_period
        period_diff = final_period - period

        grating_length = self.grating.bbox[2] - self.grating.bbox[0]
        out_nodes = []

        use_interposer = (period_diff > 0)
        with nd.Cell(name=f'grating_array_{n}_{period}_{final_period}') as gratings:
            for idx in range(n + 2 * use_interposer):
                self.grating.put(0, final_period * idx, flop=True)
                if use_interposer:
                    radius = period_diff / 2  # need to change
                    angle = np.arccos(1 - period_diff / 4 / radius) * 180 / np.pi
                    if idx == 0:
                        self.waveguide_ic.bend(radius=radius, angle=-180).put(0, 0)
                        self.waveguide_ic.strt(length=grating_length + 5).put()
                        self.waveguide_ic.bend(radius=radius, angle=-90).put()
                        self.waveguide_ic.strt(length=final_period * (n + 1) + 2 * radius).put()
                        self.waveguide_ic.bend(radius=radius, angle=-90).put()
                        self.waveguide_ic.strt(length=grating_length + 5).put()
                        self.waveguide_ic.bend(radius=radius, angle=-180).put()
                    elif idx < n + 1:
                        self.waveguide_ic.strt().put(0, final_period * idx, 0)
                        for i in range(int(np.ceil(n / 2))):
                            if idx < int(np.ceil(n / 2)):
                                self.waveguide_ic.bend(radius, angle).put()
                                self.waveguide_ic.bend(radius, -angle).put()
                                self.waveguide_ic.bend(radius, angle if i >= idx else -angle).put()
                                out = self.waveguide_ic.bend(radius, -angle if i >= idx else angle).put()
                            else:
                                self.waveguide_ic.bend(radius, -angle).put()
                                self.waveguide_ic.bend(radius, angle).put()
                                self.waveguide_ic.bend(radius, -angle if i >= n - idx else angle).put()
                                out = self.waveguide_ic.bend(radius, angle if i >= n - idx else -angle).put()
                        out_nodes.append(out.pin['b0'])
        return gratings, out_nodes

    # def multipath_turn(self, n: int, radius: float, angle: float, curr_period: float, final_period: float):
    #

    def bidirectional_grating_tap(self, node: nd.Node, radius: float, angle: float, gap_w: float):
        vert_angle = 90 - angle if angle > 0 else -90 + angle
        y_offset = self.waveguide_w + gap_w
        self.waveguide_ic.bend(radius=radius, angle=angle).put(node.x, node.y + y_offset)
        self.waveguide_ic.bend(radius=self.min_radius, angle=vert_angle).put()
        self.grating.put(nd.cp.x(), nd.cp.y(), 90 * np.sign(angle))
        self.waveguide_ic.bend(radius=radius, angle=-angle).put(node.x, node.y + y_offset, -180)
        self.waveguide_ic.bend(radius=self.min_radius, angle=-vert_angle).put()
        self.grating.put(nd.cp.x(), nd.cp.y(), 90 * np.sign(angle))

    def triangular_mesh(self, n: int, arm_l: float, gap_w: float, interaction_l: float, mzi_w: float, radius: float,
                        trench_gap: float, with_grating_taps: bool = True):

        num_straight = (n - 1) - (np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) + 1)
        bend_angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

        with nd.Cell(name=f'triangular_mesh_{n}_{with_grating_taps}') as triangular_mesh:
            heater = self.heater(heater_l=arm_l)
            trench = self.trench_ic.strt(length=arm_l)
            # mesh
            for idx in range(n):
                self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w * idx, 0)
                for layer in range(2 * n - 3):
                    angle = -bend_angle if idx == n - 1 or (idx - layer % 2 < n and idx > num_straight[layer]) and (
                            idx + layer) % 2 else bend_angle
                    i_node, l_node, lt_node, r_node, ot_node, o_node = self.mzi_path(angle, arm_l, interaction_l,
                                                                                     radius,
                                                                                     trench_gap, heater=heater,
                                                                                     trench=trench)

                    if with_grating_taps:
                        self.bidirectional_grating_tap(lt_node, radius, np.abs(angle) / 4, gap_w)
                        self.bidirectional_grating_tap(ot_node, radius, np.abs(angle) / 4, gap_w)
                    self.waveguide_ic.strt(length=0).put(o_node)
                output = self.waveguide_ic.strt(length=2 * arm_l).put()
                o_node = output.pin['a0']
                heater.put(o_node.x, mzi_w * idx)
                trench.put(o_node.x, mzi_w * idx + trench_gap)
                trench.put(o_node.x, mzi_w * idx - trench_gap)

        return triangular_mesh


if __name__ == 'main':
    waveguide_w = 0.5
    interport_w = 70
    mzi_kwargs = {
        'gap_w': 0.3,
        'interaction_l': 40,
        'mzi_w': interport_w,
        'arm_l': 80,
        'radius': 35,
        'trench_gap': 10
    }

    chip = PhotonicChip(AMF_STACK, waveguide_w)

    # triangular meshes

    mesh = chip.triangular_mesh(n=6, **mzi_kwargs).put(0, 0)
    chip.triangular_mesh(n=6, **mzi_kwargs, with_grating_taps=False).put(0, 11 * interport_w, flip=True)

    # small test structures

    splitter_tree = chip.splitter_tree_4(**mzi_kwargs)
    splitter_tree.put(0, -4 * interport_w)
    splitter_tree.put(0, 20 * interport_w)
    splitter_line = chip.splitter_line_4(**mzi_kwargs)
    splitter_line.put(mesh.bbox[2], -4 * interport_w, flop=True)
    splitter_line.put(mesh.bbox[2], 20 * interport_w, flop=True)
    chip.mzi(**mzi_kwargs).put(mesh.bbox[2] / 2, 20 * interport_w, flop=True)
    chip.mzi(**mzi_kwargs).put(mesh.bbox[2] / 2, -4 * interport_w, flop=True)

    # gratings and interposers

    interposer, out_nodes = chip.grating_array(16, period=70, final_period=127)
    idx = 4
    interposer.put(-out_nodes[idx].x, -out_nodes[idx].y)
    interposer.put(mesh.bbox[2] + out_nodes[idx].x, -out_nodes[idx].y, flop=True)

    gratings_4, _ = chip.grating_array(4, period=70)
    gratings_4.put(splitter_tree.bbox[2], -4 * interport_w, flop=True)
    gratings_4.put(splitter_tree.bbox[2], 20 * interport_w, flop=True)
    gratings_4.put(0, 20 * interport_w)
    gratings_4.put(mesh.bbox[2] - splitter_line.bbox[2], -4 * interport_w)
    gratings_4.put(mesh.bbox[2], 20 * interport_w, flop=True)
    gratings_4.put(mesh.bbox[2] - splitter_line.bbox[2], 20 * interport_w)

    # bond pad arrays
    bond_pad_array = chip.bond_pad_array(n_pads=26)
    bond_pad_array.put(100, -6 * interport_w)
    bond_pad_array.put(200, -6 * interport_w - 200)
    bond_pad_array.put(100, 14 * interport_w)
    bond_pad_array.put(200, 14 * interport_w + 200)

    nd.export_gds(filename='amf.gds')
