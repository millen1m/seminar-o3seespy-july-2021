import sfsimodels as sm
import all_paths as ap
import os


def generate_pier_ssi_system():

    sl = sm.Soil()
    sl.unit_dry_weight = 16.5e3  # N/m3
    sl.specific_gravity = 2.65
    sl.phi = 0.0
    sl.cohesion = 60.0e3  # Pa
    sl.g_mod = 25.0e6  # Pa
    sl.poissons_ratio = 0.3
    sp = sm.SoilProfile()
    sp.add_layer(0, sl)
    import numpy as np
    sp.x_angles = [0.0, 0.0]
    sp.height = 12.0  # m

    fd = sm.RaftFoundation()
    fd.id = 1
    fd.width = 5.0  # m
    fd.length = 7.0  # m
    fd.height = 2.0  # m
    fd.depth = 1.5  # m
    fd.mass = 80.0e3  # kg
    fd.ip_axis = 'length'

    bd = sm.SDOFBuilding()
    bd.h_eff = 10.0  # m
    bd.mass_eff = 300.0e3
    bd.xi = 0.05
    bd.t_fixed = 1.1  # s
    bd.inputs.append('xi')
    bd.set_foundation(fd, x=0)
    ecp_out = sm.Output()
    ecp_out.add_to_dict(sp)
    ecp_out.add_to_dict(bd)
    ecp_out.add_to_dict(fd)
    ofolder = ap.MODULE_DATA_PATH + 'pier/'
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    ecp_out.to_file(ofolder + 'pier_ecp.json')
    return bd, sp


if __name__ == '__main__':
    generate_pier_ssi_system()
