import numpy as np
import o3seespy as o3
import geofound as gf
import scipy.integrate


def set_bnwf2d_via_millen_et_al_2021(osi, sl, fd, soil_node, fd_node, ip_axis, dettach=True, mtype='pro', pro_params=None,
                                     b=0.01, btype='mn-only', phi_if=None, qv_max=0.13, sand_like=0, match_km=True, c_min=0.1,
                                     at_edge=False, n_springs=20, f_tension_ratio=1.0):
    """
    Set a Beam on nonlinear Winker Foundation between two nodes

    Parameters
    ----------
    osi
    sl: sfsimodels.Soil object
    fd: sfsimodels.Foundation object
    soil_node: o3.Node
        The soil node
    fd_node: o3.Node
        The base of the building node
    ip_axis: str
        The axis that is in the plane of deformation
    dettach: bool
        If true, then use zero tension springs
    mtype: str
        Type of material that should be used for modelling the vertical springs
        'lin' - linear
        'ep' - elastoplastic
        'pro' - progressive yielding
        'ebl' - ElasticBilin
    qv_max: float
        Ratio of maximum shear strength to ultimate bearing capacity (default=0.13 from Gottardi)
    Returns
    -------

    """

    # TODO: add q_spring dist
    # Note: fd_node is actual foundation base - account for foundation height in exterior model
    k_rot = gf.stiffness.calc_rotational_via_gazetas_1991(sl, fd, ip_axis=ip_axis, f_contact=0.0)
    k_vert = gf.stiffness.calc_vert_via_gazetas_1991(sl, fd, f_contact=0.0)
    print('k_rot: ', k_rot)
    l_ip = getattr(fd, ip_axis)
    nk = k_rot / k_vert / l_ip ** 2

    xn = np.linspace(-0.5, 0.5, n_springs + 1)
    if at_edge:  # TODO: put spring on end?
        xn_pos = np.linspace(-0.5, 0.5, n_springs)
    else:
        dx = 0.5 / n_springs
        xn_pos = np.linspace(dx - 0.5, 0.5 - dx, n_springs)
    pos = xn_pos * l_ip
    nk_lim = -c_min / 15 + 3. / 20
    if nk < nk_lim:
        an = 180.0 * nk - 15.0
        cn = 2.25 - 15.0 * nk
    else:
        if match_km:
            an = 80.0 * nk - 20. * c_min / 3
            cn = c_min
        else:
            an = 80.0 * nk_lim - 20. * c_min / 3
            cn = c_min

    ff = (an * xn ** 2 + cn)
    nk_springs = (ff[1:] + ff[:-1]) / 2
    k_spring = nk_springs * k_vert / n_springs
    kn_rat = np.mean(nk_springs)
    km_spring = np.sum(k_spring * xn_pos ** 2) * l_ip ** 2
    kn_spring = np.sum(k_spring)
    print('km_spring: ', km_spring, km_spring / k_rot)
    print('kn_spring: ', kn_spring, kn_spring / k_vert)

    if dettach:
        k_ten = 1.0e-5 * k_spring
    else:
        k_ten = k_spring
    # use 10 springs - have the exterior spring
    if ip_axis == 'width':
        oop_axis = 'length'
    else:
        oop_axis = 'width'
    l_oop = getattr(fd, oop_axis)

    if mtype == 'lin':
        f_ult = None
        f_spring = [None] * n_springs
    else:
        q_ult = gf.capacity_salgado_2008(sl, fd)
        f_ult = q_ult * fd.area
        if sand_like:  # TODO: should detect based on soil object
            sf_at_incs = 1.375 - 4.5 * xn ** 2
            sf = (sf_at_incs[1:] + sf_at_incs[:-1]) / 2
            nn = sand_like
            sf += nn  # / len(sf)
            sf /= (nn + 1)
            sf_mean = np.mean(sf)
            print('sf_mean: ', sf_mean)
        else:
            sf = np.ones(n_springs)
        f_spring = f_ult / n_springs * sf
    print('f_ult: ', sum(f_spring) / f_ult)
    spring_mats = []
    for i in range(n_springs):
        if mtype == 'lin':
            spring_mats.append(o3.uniaxial_material.Elastic(osi, k_spring[i], eneg=k_ten[i]))
        elif mtype == 'ep':
            int_spring_mat_1 = o3.uniaxial_material.Steel01(osi, f_spring[i], k_spring[i], b=b)
            if dettach:
                mat_obj2 = o3.uniaxial_material.Elastic(osi, 1000 * k_spring[i], eneg=0.0001 * k_spring[i])
                spring_mats.append(o3.uniaxial_material.Series(osi, [int_spring_mat_1, mat_obj2]))
            else:
                spring_mats.append(int_spring_mat_1)
        elif mtype == 'pro':
            if pro_params is None:
                pro_params = [20, 0., 0.15]
            int_spring_mat_1 = o3.uniaxial_material.SteelMPF(osi, f_spring[i], f_tension_ratio * f_spring[i], k_spring[i], b, b, params=pro_params)
            if dettach:
                mat_obj2 = o3.uniaxial_material.Elastic(osi, 1000 * k_spring[i], eneg=0.0001 * k_spring[i])
                spring_mats.append(o3.uniaxial_material.Series(osi, [int_spring_mat_1, mat_obj2]))
            else:
                spring_mats.append(int_spring_mat_1)
        else:
            raise ValueError(f"mtype must be 'lin', 'ep' or 'pro', not {mtype}")

    from o3seespy.command.element.soil_foundation import gen_shallow_foundation_bnwf
    fd_area = l_oop * fd.height
    fd_emod = 30.0e12
    fd_iz = l_oop * fd.height ** 3 / 12
    if btype == 'mn-only':
        return gen_shallow_foundation_bnwf(osi, soil_node, fd_node, sf_mats=spring_mats, pos=pos, fd_area=fd_area,
                                           fd_e_mod=fd_emod, fd_iz=fd_iz)

    k_shear = gf.stiffness.calc_shear_via_gazetas_1991(sl, fd, axis=ip_axis, f_contact=0)
    # xi = 1000.
    # k_shear *= 10000
    if '-f' in btype:
        if phi_if is None:
            phi_if = sl.phi
        mu = np.tan(np.radians(phi_if))
        sf_frn = o3.friction_model.Coulomb(osi, mu)
    else:
        sf_frn = None
    if btype == 'mn-v1e':
        sf_horz_mats = o3.uniaxial_material.Elastic(osi, k_shear)
    elif 'mn-v1i' in btype:
        if mtype == 'ep':
            sf_horz_mats = o3.uniaxial_material.Steel01(osi, qv_max * f_ult, k_shear, b=0.1)
        elif mtype == 'pro':
            sf_horz_mats = o3.uniaxial_material.SteelMPF(osi, qv_max * f_ult, qv_max * f_ult, k_shear, 0.1, 0.1, params=pro_params)
        else:
            sf_horz_mats = o3.uniaxial_material.Elastic(osi, qv_max * f_ult, k_shear)
    elif 'mn-vi' in btype:
        sf_horz_mats = []
        for i in range(n_springs):
            if mtype == 'ep':
                sf_horz_mats.append(o3.uniaxial_material.Steel01(osi, qv_max * f_spring[i], k_shear / n_springs, b=0.05))
            elif mtype == 'pro':
                sf_horz_mats.append(o3.uniaxial_material.SteelMPF(osi, qv_max * f_spring[i], qv_max * f_spring[i],
                                                                  k_shear / n_springs, b, b, params=pro_params))
            else:
                sf_horz_mats.append(o3.uniaxial_material.Elastic(osi, k_shear / n_springs))
    else:
        raise ValueError('btype must be one of [mn-only, mn-v1e, mn-v1i, mn-vi, mn-vi-f]')
    return gen_shallow_foundation_bnwf(osi, soil_node, fd_node, sf_mats=spring_mats,
                                       sf_horz_mats=sf_horz_mats, pos=pos, sf_frn=sf_frn,
                                       fd_area=fd_area, fd_e_mod=fd_emod, fd_iz=fd_iz)
