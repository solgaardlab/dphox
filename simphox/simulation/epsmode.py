import numpy as np


def wg_xsection(size, spacing, wg_x, wg_y, substrate_height, sub_eps, wg_eps):
    nx = int(size[0] / spacing)
    ny = int(size[1] / spacing)
    center = nx // 2
    xr = (center - int((wg_x / 2) / spacing), center + int((wg_x / 2) / spacing))
    yr = (int(wg_y[0] / spacing), int(wg_y[1] / spacing))
    eps = np.ones((nx, ny))
    eps[:, :int(substrate_height / spacing)] = sub_eps
    eps[xr[0]:xr[1], yr[0]:yr[1]] = wg_eps
    return eps


def dc_xsection(size, spacing, wg_x, wg_y, gap, substrate_height, sub_eps, wg_eps):
    nx = int(size[0] / spacing)
    ny = int(size[1] / spacing)
    center = nx // 2
    xr_l = (center - int((gap / 2 + wg_x) / spacing), center - int(gap / 2 / spacing))
    xr_r = (center + int((gap / 2) / spacing), center + int((gap / 2 + wg_x) / spacing))
    yr = (int(wg_y[0] / spacing), int(wg_y[1] / spacing))
    eps = np.ones((nx, ny))
    eps[:, :int(substrate_height / spacing)] = sub_eps
    eps[xr_l[0]:xr_l[1], yr[0]:yr[1]] = wg_eps
    eps[xr_r[0]:xr_r[1], yr[0]:yr[1]] = wg_eps
    return eps


def tdc_xsection(size, spacing, wg_x, wg_y, ps_x, ps_y, gap, substrate_height, sub_eps, wg_eps, ps_eps=None, ps_y_2=None):
    nx = int(size[0] / spacing)
    ny = int(size[1] / spacing)
    center = nx // 2

    ps_eps = wg_eps if ps_eps is None else ps_eps

    xr_l = (center - int((gap / 2 + wg_x) / spacing), center - int(gap / 2 / spacing))
    xr_r = (center + int((gap / 2) / spacing), center + int((gap / 2 + wg_x) / spacing))
    xrps_l = (xr_l[0], xr_l[0] + int(ps_x / spacing))
    xrps_r = (xr_r[1] - int(ps_x / spacing), xr_r[1])
    yr = (int(wg_y[0] / spacing), int(wg_y[1] / spacing))
    yr_ps = (int(ps_y[0] / spacing), int(ps_y[1] / spacing))
    eps = np.ones((nx, ny))
    eps[:, :int(substrate_height / spacing)] = sub_eps
    eps[xr_l[0]:xr_l[1], yr[0]:yr[1]] = wg_eps
    eps[xr_r[0]:xr_r[1], yr[0]:yr[1]] = wg_eps
    eps[xrps_l[0]:xrps_l[1], yr_ps[0]:yr_ps[1]] = ps_eps
    if ps_y_2:
        yr_ps2 = (int(ps_y_2[0] / spacing), int(ps_y_2[1] / spacing))
        eps[xrps_r[0]:xrps_r[1], yr_ps2[0]:yr_ps2[1]] = ps_eps
    return eps


def ps_xsection(size, spacing, wg_x, wg_y, ps_x, ps_y, substrate_height, sub_eps, wg_eps, ps_eps=None):
    nx = int(size[0] / spacing)
    ny = int(size[1] / spacing)
    center = nx // 2

    ps_eps = wg_eps if ps_eps is None else ps_eps

    xr_wg = (center - int(wg_x / 2 / spacing), center + int(wg_x / 2 / spacing))
    xr_ps = (center - int(ps_x / 2 / spacing), center + int(ps_x / 2 / spacing))
    yr_wg = (int(wg_y[0] / spacing), int(wg_y[1] / spacing))
    yr_ps = (int(ps_y[0] / spacing), int(ps_y[1] / spacing))

    eps = np.ones((nx, ny))
    eps[:, :int(substrate_height / spacing)] = sub_eps
    eps[xr_wg[0]:xr_wg[1], yr_wg[0]:yr_wg[1]] = wg_eps
    eps[xr_ps[0]:xr_ps[1], yr_ps[0]:yr_ps[1]] = ps_eps

    return eps
