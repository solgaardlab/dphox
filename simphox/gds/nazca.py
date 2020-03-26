import nazca as nd
import numpy as np

# NOTE: This file currently contains privileged information. Only distribute to those who have signed NDA with AMF.
# This NDA is a MUTUAL CONFIDENTIALITY AGREEMENT signed at 13th March 2020 by Advanced Micro Foundry Pte. Ltd (Co. Reg.
# No. 20170322R) of 11 Science Park Road Singapore 117685 and Olav Solgaard, Stanford University,
# Stanford, CA 94305, USA. Much contribution to this code (aside from authors of this repo) comes from much work done at
# Politecnico de Milano, specifically by Maziyar Milanizadeh.


class PhotonicChip:
    def __init__(self, process_stack: dict, waveguide_w: float):
        """

        Args:
            process_stack: The stack, in JSON format (see below example)
            waveguide_w: The width of the waveguide (μm)
        """
        for layer_name in process_stack['layers']:
            nd.add_layer(name=layer_name, layer=process_stack['layers'][layer_name], accuracy=0.001, overwrite=True)
        for xs_name in process_stack['cross_sections']:
            for layer_dict in process_stack['cross_sections'][xs_name]:
                nd.add_layer2xsection(
                    xsection=xs_name,
                    accuracy=0.001,
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
            # nd.Pin(f'g{idx}{layer}').put(in_tpr.pin['a0'])
            nd.Pin(f'g').put(in_tpr.pin['a0'])

            nd.put_stub([], length=0)

        return grating

    # Polimi heater (Maziyar Milanizadeh)
    @nd.hashme('heater', 'heater_l')
    def _heater(self, heater_l: float, via_w: float = 2):
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
    def _bond_pad(self, pad_w: float = 100, pad_l: float = 100):
        with nd.Cell(name='bond_pad') as bond_pad:
            pad = self.pad_ic.strt(length=pad_l, width=pad_w).put()
            # add pins to the building block
            nd.Pin('a0', pin=pad.pin['a0']).put()
            nd.Pin('b0', pin=pad.pin['b0']).put()
        return bond_pad

    # Polimi 1d bond pad array (Maziyar Milanizadeh)
    @nd.hashme('bond_pad_array', 'n_pads', 'pitch')
    def _bond_pad_array(self, n_pads: int, pitch: float = 200,
                        pad_w: float = 100, pad_l: float = 100):
        lattice_bond_pads = []
        with nd.Cell(hashme=True) as bond_pad_array:
            pad = self._bond_pad(pad_w=pad_w, pad_l=pad_l)
            for i in range(n_pads):
                lattice_bond_pads.append(pad.put(i * pitch, 0, 270))
                nd.Pin(f'pdt{i}').put(lattice_bond_pads[i].pin['a0'])
                nd.Pin(f'pdb{i}').put(lattice_bond_pads[i].pin['b0'])
                message = nd.text(text=f'{i}', align='cc', layer=10, height=50)
                message.put(pitch / 2 + i * pitch, -pad_l / 2)
            nd.put_stub([], length=0)
        return bond_pad_array

    def triangular_mesh(self, n, gap_w, interaction_l, mzi_w, arm_l, radius, trench_gap, n_pads,
                        with_grating_taps=True, with_upper_bond_pads=True, with_lower_bond_pads=False, y_shift=0):

        waveguide_w = self.waveguide_w

        num_straight = (n - 1) - (np.hstack([np.arange(1, n), np.arange(n - 2, 0, -1)]) + 1)
        bend_angle = np.arccos(1 - (mzi_w - gap_w - waveguide_w) / 4 / radius) * 180 / np.pi

        angles, pos_i, pos_l, pos_r, pos_o = [], [], [], [], []

        grating = self.grating
        heater = self._heater(heater_l=arm_l)
        bond_pad_array = self._bond_pad_array(n_pads=n_pads)
        trench = self.trench_ic.strt(length=arm_l)

        if with_lower_bond_pads:
            bond_pad_array.put(0, - 2 * mzi_w + y_shift)
            bond_pad_array.put(100, - 2 * mzi_w - 200 + y_shift)
        if with_upper_bond_pads:
            bond_pad_array.put(0, (n + 2) * mzi_w + y_shift)
            bond_pad_array.put(100, (n + 2) * mzi_w + 200 + y_shift)

        # mesh
        for idx in range(n):
            self.waveguide_ic.strt(width=waveguide_w, length=arm_l).put(0, mzi_w * idx + y_shift)
            grating.put(0, mzi_w * idx + y_shift, 180)
            heater.put(arm_l, mzi_w * idx + y_shift)
            trench.put(arm_l, mzi_w * idx + trench_gap + y_shift)
            trench.put(arm_l, mzi_w * idx - trench_gap + y_shift)
            self.waveguide_ic.strt(width=waveguide_w, length=arm_l).put(arm_l, mzi_w * idx + y_shift)
            for layer in range(2 * n - 3):
                angle = -bend_angle if idx - layer % 2 < n and idx >= num_straight[layer] and (
                        idx + layer) % 2 else bend_angle
                angle = -angle if idx < num_straight[layer] else angle
                angles.append(angle)
                input_waveguide = self.waveguide_ic.bend(radius=radius, angle=angle).put()
                self.waveguide_ic.bend(radius=radius, angle=-angle).put()
                self.waveguide_ic.strt(length=interaction_l).put()
                self.waveguide_ic.bend(radius=radius, angle=-angle).put()
                self.waveguide_ic.bend(radius=radius, angle=angle).put()
                x, y = nd.cp.x(), nd.cp.y()
                trench.put(x, y + trench_gap)
                trench.put(x, y - trench_gap)
                heater.put(x, y)
                arm_waveguide = self.waveguide_ic.strt(length=arm_l).put()
                self.waveguide_ic.bend(radius=radius, angle=angle).put()
                self.waveguide_ic.bend(radius=radius, angle=-angle).put()
                self.waveguide_ic.strt(length=interaction_l).put()
                self.waveguide_ic.bend(radius=radius, angle=-angle).put()
                self.waveguide_ic.bend(radius=radius, angle=angle).put()
                x, y = nd.cp.x(), nd.cp.y()
                trench.put(x, y + trench_gap)
                trench.put(x, y - trench_gap)
                heater.put(x, y)
                output_waveguide = self.waveguide_ic.strt(length=arm_l).put()
                i_node = input_waveguide.pin['a0']
                l_node = arm_waveguide.pin['a0']
                r_node = arm_waveguide.pin['b0']
                o_node = output_waveguide.pin['a0']

                nd.Pin(f'i{idx}{layer}').put(i_node)
                nd.Pin(f'l{idx}{layer}').put(l_node)
                nd.Pin(f'r{idx}{layer}').put(r_node)
                nd.Pin(f'o{idx}{layer}').put(o_node)

                y_offset = (waveguide_w + gap_w) * np.sign(angle)

                pos_i.append((i_node.x, i_node.y - y_offset))
                pos_l.append((l_node.x, l_node.y - y_offset))
                pos_r.append((r_node.x, r_node.y - y_offset))
                pos_o.append((o_node.x, o_node.y - y_offset))
            self.waveguide_ic.strt(width=waveguide_w, length=arm_l).put()
            grating.put()

        # grating taps
        if with_grating_taps:
            for (angle, pi, pl, pr, po) in zip(angles, pos_i, pos_l, pos_r, pos_o):
                for p in (pi, pr):
                    self.waveguide_ic.bend(width=waveguide_w, radius=radius, angle=-angle / 4).put(p[0], p[1])
                    self.waveguide_ic.bend(width=waveguide_w, radius=radius, angle=angle / 2).put()
                    self.waveguide_ic.bend(width=waveguide_w, radius=radius, angle=-angle / 4).put()
                    self.waveguide_ic.strt(length=15).put()
                    grating.put(nd.cp.x(), nd.cp.y())

                for p in (pl, po):
                    self.waveguide_ic.bend(width=waveguide_w, radius=radius, angle=angle / 4).put(p[0], p[1], -180)
                    self.waveguide_ic.bend(width=waveguide_w, radius=radius, angle=-angle / 2).put()
                    self.waveguide_ic.bend(width=waveguide_w, radius=radius, angle=angle / 4).put()
                    if angle < 0:
                        self.waveguide_ic.bend(radius=57 / 4, angle=np.sign(angle) * 90).put()
                        self.waveguide_ic.strt(length=5).put()
                        self.waveguide_ic.bend(radius=57 / 4, angle=np.sign(angle) * 90).put()
                        self.waveguide_ic.strt(length=45).put()
                        grating.put(nd.cp.x(), nd.cp.y())

        nd.put_stub([], length=0)


if __name__ == 'main':
    chip = PhotonicChip({
        'layers': {
            'waveguide': 10,
            'grating': 11,
            'via': 100,
            'via_heater': 120,
            'mt_heater': 125,
            'heater': 115,
            'slab': 12,
            'pad': 150,
            'trench': 160
        },
        'cross_sections': {
            'heater_xs': [
                {
                    'layer': 115,  # heater
                    'growx': 0.755,  # (waveguide_w - heater_w) / 2 + 0.005
                    'growy': 0.005
                },
                #             {
                #                 'layer': 10,  # waveguide
                #                 'growy': 0.001
                #             }
            ],
            'via_heater_xs': [
                {
                    'layer': 120  # via_heater
                },
                {
                    'layer': 125,  # mt_heater
                    'growx': 1.5,
                    'growy': 1.5
                }
            ],
            'grating_xs': [
                {
                    'layer': 11  # grating
                }
            ],
            'waveguide_xs': [
                {
                    'layer': 10,  # waveguide
                    'growy': 0.004
                }
            ],
            'pad_xs': [
                {
                    'layer': 125  # mt_heater
                },
                {
                    'layer': 150,  # pad
                    'growx': -2,
                    'growy': -2
                }
            ],
            'trench_xs': [
                {
                    'layer': 160  # trench
                }
            ],
            'slab_xs': [
                {
                    'layer': 12  # slab
                }
            ]
        }
    }, 0.5)

    with nd.Cell('meshes') as meshes:
        chip.triangular_mesh(
            n=6,
            gap_w=0.3,
            interaction_l=40,
            mzi_w=70,
            arm_l=80,
            radius=35,
            n_pads=23,
            trench_gap=10
        )
        chip.triangular_mesh(
            n=6,
            gap_w=0.3,
            interaction_l=40,
            mzi_w=70,
            arm_l=80,
            radius=35,
            n_pads=23,
            trench_gap=10,
            y_shift=900,
            with_grating_taps=False
        )

    nd.export_gds(filename='mesh.gds', topcells=[meshes])
