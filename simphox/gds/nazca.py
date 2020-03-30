from ..typing import Tuple, Optional
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
        xs = nd.get_xsection('heater_xs')
        xs.os = 0.0

        self.waveguide_w = waveguide_w
        self.waveguide_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='waveguide_xs')
        self.slab_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='slab_xs')
        self.grating_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='grating_xs')
        self.heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='heater_xs')
        self.via_heater_ic = nd.interconnects.Interconnect(width=waveguide_w, radius=10, xs='via_heater_xs')
        self.pad_ic = nd.interconnects.Interconnect(width=100, radius=10, xs='pad_xs')
        self.trench_ic = nd.interconnects.Interconnect(width=10, radius=10, xs='trench_xs')

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
            # Export nodes
            nd.Pin(f'g').put(in_tpr.pin['a0'])

            nd.put_stub([], length=0)

        return grating

    # Polimi heater (Maziyar Milanizadeh)
    @nd.hashme('heater', 'heater_l')
    def heater(self, heater_l: float, via_w: float = 2):
        heater_ic, via_ic = self.heater_ic, self.via_heater_ic
        with nd.Cell(hashme=True) as heater:
            htr = heater_ic.strt(length=heater_l).put(0, 0)
            via_in = via_ic.strt(width=via_w, length=via_w).put(htr.pin['a0'])
            via_out = via_ic.strt(width=via_w, length=via_w).put(htr.pin['b0'])

            # add pins to the building block
            nd.Pin('hl').put(htr.pin['a0'])
            nd.Pin('hr').put(htr.pin['b0'])
            nd.Pin('pl').put(via_in.pin['a0'])
            nd.Pin('pr').put(via_out.pin['b0'])

            nd.put_stub([], length=0)
        return heater

    # Polimi bond pad (Maziyar Milanizadeh)
    def bond_pad(self, pad_w: float = 100, pad_l: float = 100):
        with nd.Cell(name='bond_pad') as bond_pad:
            pad = self.pad_ic.strt(length=pad_l, width=pad_w).put()
            # add pins to the building block
            nd.Pin('a0', pin=pad.pin['a0']).put()
            nd.Pin('b0', pin=pad.pin['b0']).put()
        return bond_pad

    # Polimi 1d bond pad array (Maziyar Milanizadeh)
    @nd.hashme('bond_pad_array', 'n_pads', 'pitch')
    def bond_pad_array(self, n_pads: int, pitch: float = 200,
                       pad_w: float = 100, pad_l: float = 100):
        lattice_bond_pads = []
        with nd.Cell(hashme=True) as bond_pad_array:
            pad = self.bond_pad(pad_w=pad_w, pad_l=pad_l)
            for i in range(n_pads):
                lattice_bond_pads.append(pad.put(i * pitch, 0, 270))
                nd.Pin(f'pdt{i}').put(lattice_bond_pads[i].pin['a0'])
                nd.Pin(f'pdb{i}').put(lattice_bond_pads[i].pin['b0'])
                message = nd.text(text=f'{i}', align='cc', layer=10, height=50)
                message.put(pitch / 2 + i * pitch, -pad_l / 2)
            nd.put_stub([], length=0)
        return bond_pad_array

    def coupler_path(self, angle, interaction_l, radius):
        input_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
        self.waveguide_ic.strt(length=interaction_l).put()
        self.waveguide_ic.bend(radius=radius, angle=-angle).put()
        output_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
        return input_waveguide.pin['a0'], output_waveguide.pin['b0']

    def mzi_path(self, angle, arm_l, interaction_l, radius, trench_gap,
                 heater: Optional[nd.Cell] = None, trench: Optional[nd.Cell] = None):
        def put_heater():
            x, y = nd.cp.x(), nd.cp.y()
            if trench:
                trench.put(x, y + trench_gap)
                # trench.put(x, y - trench_gap)
            if heater:
                heater.put(x, y)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        i_node, l_node = self.coupler_path(angle, interaction_l, radius)
        put_heater()
        self.waveguide_ic.strt(length=arm_l).put()
        r_node, o_node = self.coupler_path(angle, interaction_l, radius)
        return i_node, l_node, r_node, o_node

    @nd.hashme('mzi', 'n_pads', 'pitch')
    def mzi(self, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap, with_grating_taps=True):
        heater = self.heater(heater_l=arm_l)
        trench = self.trench_ic.strt(length=arm_l)
        with nd.Cell(hashme=True) as mzi:
            angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

            # upper path
            self.grating.put(0, 0, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, 0, 0)
            self.mzi_path(angle, arm_l, interaction_l, radius,
                          trench_gap, heater=heater, trench=trench)
            self.waveguide_ic.strt(length=arm_l).put()
            self.grating.put()

            # lower path
            self.grating.put(0, mzi_w, 180)
            self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w, 0)
            self.mzi_path(-angle, arm_l, interaction_l, radius,
                          trench_gap, heater=heater, trench=trench)
            self.waveguide_ic.strt(length=arm_l).put()
            self.grating.put()
        return mzi

    @nd.hashme('binary_tree_4', 'n_pads', 'pitch', 'sensor')
    def splitter_tree_4(self, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap, right_facing=True,
                        sensor=0):
        directions = np.asarray([(1, 1), (1, -1), (-1, 1), (-1, -1)]) if right_facing else \
            np.asarray([(1, 1), (-1, 1), (1, -1), (-1, -1)])

        with nd.Cell(hashme=True) as binary_tree_4:
            heater = self.heater(heater_l=arm_l)
            trench = self.trench_ic.strt(length=arm_l)
            angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi
            for idx, direction in enumerate(directions):
                self.grating.put(0, mzi_w * idx, 180)
                self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w * idx, 0)
                self.mzi_path(direction[0] * angle, arm_l, interaction_l, radius,
                              trench_gap, heater=heater, trench=trench)
                self.mzi_path(direction[1] * angle, arm_l, interaction_l, radius,
                              trench_gap, heater=heater, trench=trench)
                self.waveguide_ic.strt(length=arm_l).put()
                if sensor != 0:
                    c = idx if sensor < 0 else 4 - idx
                    self.waveguide_ic.strt(length=arm_l * (c + 0.25 * (1 + sensor))).put()
                    self.waveguide_ic.bend(radius=radius, angle=sensor * 90).put()
                    if sensor < 0:
                        self.waveguide_ic.strt(length=((4.5 - idx) * mzi_w - 2 * radius)).put()
                    else:
                        self.waveguide_ic.strt(length=((idx + 1.5) * mzi_w - 2 * radius)).put()
                    self.waveguide_ic.bend(radius=radius, angle=-sensor * 90).put()
                    self.waveguide_ic.strt(length=arm_l * (4.25 - c - 0.25 * sensor)).put()
                    self.mzi_path(sensor * angle, arm_l, interaction_l, radius,
                                  trench_gap, heater=heater, trench=trench)
                self.grating.put()

        return binary_tree_4

    @nd.hashme('triangular_mesh', 'n', 'arm_l', 'gap_w', 'interaction_l')
    def triangular_mesh(self, n, arm_l, gap_w, interaction_l, mzi_w, radius, trench_gap,
                        with_grating_taps=True):

        num_straight = (n - 1) - (np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) + 1)
        bend_angle = np.arccos(1 - (mzi_w - gap_w - self.waveguide_w) / 4 / radius) * 180 / np.pi

        angles, pos_i, pos_l, pos_r, pos_o = [], [], [], [], []

        with nd.Cell(hashme=True) as triangular_mesh:
            heater = self.heater(heater_l=arm_l)
            trench = self.trench_ic.strt(length=arm_l)
            # mesh
            for idx in range(n):
                self.grating.put(0, mzi_w * idx, 180)
                self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(0, mzi_w * idx, 0)
                # heater.put(arm_l, mzi_w * idx)
                # trench.put(arm_l, mzi_w * idx + trench_gap)
                # trench.put(arm_l, mzi_w * idx - trench_gap)
                # self.waveguide_ic.strt(width=self.waveguide_w, length=arm_l).put(arm_l, mzi_w * idx)
                for layer in range(2 * n - 3):
                    angle = -bend_angle if idx - layer % 2 < n and idx >= num_straight[layer] and (
                            idx + layer) % 2 else bend_angle
                    angle = -angle if idx < num_straight[layer] else angle
                    angles.append(angle)
                    i_node, l_node, r_node, o_node = self.mzi_path(angle, arm_l, interaction_l, radius,
                                                                   trench_gap, heater=heater, trench=trench)

                    nd.Pin(f'i{idx}{layer}').put(i_node)
                    nd.Pin(f'l{idx}{layer}').put(l_node)
                    nd.Pin(f'r{idx}{layer}').put(r_node)
                    nd.Pin(f'o{idx}{layer}').put(o_node)

                    y_offset = (self.waveguide_w + gap_w) * np.sign(angle)

                    pos_i.append((i_node.x, i_node.y - y_offset))
                    pos_l.append((l_node.x, l_node.y - y_offset))
                    pos_r.append((r_node.x, r_node.y - y_offset))
                    pos_o.append((o_node.x, o_node.y - y_offset))
                output = self.waveguide_ic.strt(length=2 * arm_l).put()
                self.grating.put()
                o_node = output.pin['a0']
                heater.put(o_node.x, mzi_w * idx)
                trench.put(o_node.x, mzi_w * idx + trench_gap)

            # grating taps
            if with_grating_taps:
                for (angle, pi, pl, pr, po) in zip(angles, pos_i, pos_l, pos_r, pos_o):
                    for p in (pi, pr):
                        self.waveguide_ic.bend(width=self.waveguide_w, radius=radius, angle=-angle / 4).put(p[0], p[1])
                        self.waveguide_ic.bend(radius=radius, angle=angle / 2).put()
                        self.waveguide_ic.bend(radius=radius, angle=-angle / 4).put()
                        self.waveguide_ic.strt(length=15).put()
                        self.grating.put(nd.cp.x(), nd.cp.y())

                    for p in (pl, po):
                        self.waveguide_ic.bend(width=self.waveguide_w, radius=radius, angle=angle / 4).put(p[0], p[1], -180)
                        self.waveguide_ic.bend(radius=radius, angle=-angle / 2).put()
                        self.waveguide_ic.bend(radius=radius, angle=angle / 4).put()
                        if angle < 0:
                            self.waveguide_ic.bend(radius=57 / 4, angle=np.sign(angle) * 90).put()
                            self.waveguide_ic.strt(length=5).put()
                            self.waveguide_ic.bend(radius=57 / 4, angle=np.sign(angle) * 90).put()
                            self.waveguide_ic.strt(length=45).put()
                            self.grating.put(nd.cp.x(), nd.cp.y())

            nd.put_stub([], length=0)
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

    chip.triangular_mesh(n=6, **mzi_kwargs).put(0, 0)
    chip.triangular_mesh(n=6, **mzi_kwargs, with_grating_taps=False).put(0, 900)

    # splitter trees
    chip.splitter_tree_4(**mzi_kwargs, sensor=1).put(0, 1800)
    chip.splitter_tree_4(**mzi_kwargs, sensor=-1).put(0, 1800 + 6 * interport_w)
    chip.mzi(**mzi_kwargs).put(0, 3000)

    # bond pad arrays

    bond_pad_array = chip.bond_pad_array(n_pads=23)
    bond_pad_array.put(0, -2 * interport_w)
    bond_pad_array.put(100, -2 * interport_w)
    bond_pad_array.put(0, -2 * interport_w + 900)
    bond_pad_array.put(100, -2 * interport_w + 900)

    nd.export_gds(filename='amf.gds')
