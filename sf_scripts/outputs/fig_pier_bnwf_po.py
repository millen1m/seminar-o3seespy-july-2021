import geofound as gf
import o3seespy as o3
import numpy as np
import engformat as ef

from bwplot import cbox
import all_paths as ap
import loc_o3soil_bnwf
import settings as ops
import sfsimodels as sm
import matplotlib.pyplot as plt


def get_mom_from_rot_bnwf(bd, sl, mtype='lin', dettach=True, target_rots=None,
                          b=0.001, pro_params=None, rec_ext_springs=False, c_min=0.1):
    osi = o3.OpenSeesInstance(ndm=2, state=3)
    fd = bd.fd
    height = bd.h_eff

    # Establish nodes
    top_ss_node = o3.node.Node(osi, x=0, y=height)
    bot_ss_node = o3.node.Node(osi, 0, 0)
    bot_fd_node = o3.node.Node(osi, 0, -fd.height)
    sl_node = o3.node.Node(osi, 0, -fd.height)

    # Fix bottom node
    o3.Fix3DOF(osi, sl_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    # nodal mass (weight / g):
    o3.Mass(osi, top_ss_node, bd.mass_eff, 0., 0.)
    o3.Mass(osi, bot_ss_node, fd.mass, 0., 0.)

    # Define material
    transf = o3.geom_transf.Linear2D(osi, [])
    area = 1.0
    e_mod = 100.0e6
    iz = bd.k_eff * height ** 3 / (3 * e_mod)
    ele_nodes = [bot_ss_node, top_ss_node]

    vert_ele = o3.element.ElasticBeamColumn2D(osi, ele_nodes, area=area, e_mod=e_mod, iz=iz, transf=transf)

    fd_vert_ele = o3.element.ElasticBeamColumn2D(osi, [bot_fd_node, bot_ss_node], area=area, e_mod=e_mod, iz=iz, transf=transf)

    bnwf = loc_o3soil_bnwf.set_bnwf2d_via_millen_et_al_2021(osi, sl, fd, sl_node, bot_fd_node, ip_axis=fd.ip_axis, mtype=mtype,
                                                       dettach=dettach, b=b, pro_params=pro_params, btype='mn-v1e',
                                                         c_min=c_min)

    # set damping based on first eigen mode
    angular_freq = o3.get_eigen(osi, solver='fullGenLapack', n=1)[0] ** 0.5
    response_period = 2 * np.pi / angular_freq
    print('response_period: ', response_period)
    beta_k = 2 * bd.xi / angular_freq
    o3.rayleigh.Rayleigh(osi, alpha_m=0.0, beta_k=beta_k, beta_k_init=0.0, beta_k_comm=0.0)
    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, top_ss_node, [0, -bd.mass_eff * 9.8, 0])
    o3.Load(osi, bot_ss_node, [0, -fd.mass * 9.8, 0])

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.SparseGeneral(osi)
    n_steps_gravity = 10
    d_gravity = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, d_gravity, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3.analyze(osi, num_inc=n_steps_gravity)
    o3.load_constant(osi, time=0.0)
    print('init_disp: ', o3.get_node_disp(osi, bot_ss_node, o3.cc.DOF2D_Y))

    # Start cyclic loading
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.SparseGeneral(osi)
    n_steps_hload = 10
    d_hload = 1. / n_steps_hload
    o3.integrator.LoadControl(osi, d_hload, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)

    approx_m_cap = (bd.mass_eff + fd.mass) * 9.8 * getattr(fd, fd.ip_axis) * 0.5
    o3.extensions.to_py_file(osi)
    o3.record(osi)
    drots = np.diff(target_rots)
    rot = []
    mom = []
    sfs_f = [[], []]
    sfs_d = [[], []]
    for rr, drot in enumerate(drots):
        # Start horizontal load
        ts0 = o3.time_series.Linear(osi, factor=1)
        o3.pattern.Plain(osi, ts0)
        # o3.Load(osi, bot_fd_node, [0, 0, -dmom])
        sgn = np.sign(drot)
        o3.Load(osi, top_ss_node, [0.001 * approx_m_cap / bd.h_eff * sgn, 0, 0])
        print(o3.get_ele_response(osi, fd_vert_ele, 'force'))

        for i in range(500000):
            o3.analyze(osi, num_inc=20)
            curr_rot = -o3.get_node_disp(osi, bot_fd_node, o3.cc.DOF2D_ROTZ)
            rot.append(curr_rot)
            o3.gen_reactions(osi)
            mom.append(o3.get_ele_response(osi, fd_vert_ele, 'force')[2])
            max_mom = np.max(mom)
            min_mom = np.min(mom)
            if rec_ext_springs:
                sfs_f[0].append(o3.get_ele_response(osi, bnwf.sf_eles[0], 'force'))
                sfs_f[1].append(o3.get_ele_response(osi, bnwf.sf_eles[-1], 'force'))
                sfs_d[0].append(o3.get_ele_response(osi, bnwf.sf_eles[0], 'deformation'))
                sfs_d[1].append(o3.get_ele_response(osi, bnwf.sf_eles[-1], 'deformation'))
            if sgn * rot[-1] > sgn * target_rots[rr + 1]:
                break
            # mom.append(o3.get_node_reaction(osi, bot_fd_node, o3.cc.DOF2D_ROTZ))
        o3.load_constant(osi, time=0)
    if rec_ext_springs:
        return np.array(mom), np.array(rot), np.array(sfs_f), np.array(sfs_d)
    return np.array(mom), np.array(rot)


def get_moment_rotation(bd, sl, mtype='lin', dettach=True, mom_demands=None, n_steps=100, ip_axis='width',
                        b=0.001, pro_params=None, sand_like=False, rec_ext_springs=False, c_min=0.1):
    osi = o3.OpenSeesInstance(ndm=2, state=3)
    fd = bd.fd
    height = bd.h_eff - fd.height  # TODO: fix
    # Establish nodes

    top_ss_node = o3.node.Node(osi, 0, height)
    bot_ss_node = o3.node.Node(osi, 0, 0)
    bot_fd_node = o3.node.Node(osi, 0, -fd.height)
    sl_node = o3.node.Node(osi, 0, -fd.height)  # TODO: add fd height

    # Fix bottom node
    o3.Fix3DOF(osi, sl_node, o3.cc.FIXED, o3.cc.FIXED, o3.cc.FIXED)

    # nodal mass (weight / g):
    o3.Mass(osi, top_ss_node, bd.mass_eff, 0., 0.)
    o3.Mass(osi, bot_ss_node, fd.mass, 0., 0.)

    # Define material
    transf = o3.geom_transf.Linear2D(osi, [])
    area = 1.0
    e_mod = 100.0e6
    iz = bd.k_eff * height ** 3 / (3 * e_mod)
    ele_nodes = [bot_ss_node, top_ss_node]

    vert_ele = o3.element.ElasticBeamColumn2D(osi, ele_nodes, area=area, e_mod=e_mod, iz=iz, transf=transf)
    # TODO: replace with rigid link
    fd_vert_ele = o3.element.ElasticBeamColumn2D(osi, [bot_fd_node, bot_ss_node], area=area, e_mod=e_mod, iz=iz, transf=transf)

    # TODO: if sl.g_mod is stress dependent, then account for foundation load and depth increase using pg2-18 of NIST
    # TODO: Implement the stiffness using soil_profile into geofound that accounts for fd.q_load

    bnwf = loc_o3soil_bnwf.set_bnwf2d_via_millen_et_al_2021(osi, sl, fd, sl_node, bot_fd_node, ip_axis=ip_axis, mtype=mtype,
                                                       dettach=dettach, b=b, pro_params=pro_params, btype='mn-v1e', c_min=c_min)
    # import o3seespy.extensions
    # o3.extensions.to_py_file(osi)

    # set damping based on first eigen mode
    angular_freq = o3.get_eigen(osi, solver='fullGenLapack', n=1)[0] ** 0.5
    response_period = 2 * np.pi / angular_freq
    print('response_period: ', response_period)
    beta_k = 2 * bd.xi / angular_freq
    o3.rayleigh.Rayleigh(osi, alpha_m=0.0, beta_k=beta_k, beta_k_init=0.0, beta_k_comm=0.0)
    ts0 = o3.time_series.Linear(osi, factor=1)
    o3.pattern.Plain(osi, ts0)
    o3.Load(osi, top_ss_node, [0, -bd.mass_eff * 9.8, 0])
    o3.Load(osi, bot_ss_node, [0, -fd.mass * 9.8, 0])

    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.SparseGeneral(osi)
    n_steps_gravity = 10
    d_gravity = 1. / n_steps_gravity
    o3.integrator.LoadControl(osi, d_gravity, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)
    o3.analyze(osi, num_inc=n_steps_gravity)
    o3.load_constant(osi, time=0.0)
    print('init_disp: ', o3.get_node_disp(osi, bot_ss_node, o3.cc.DOF2D_Y))

    # Start cyclic loading
    o3.constraints.Transformation(osi)
    o3.test_check.NormDispIncr(osi, tol=1.0e-6, max_iter=35, p_flag=0)
    o3.algorithm.Newton(osi)
    o3.numberer.RCM(osi)
    o3.system.SparseGeneral(osi)
    n_steps_hload = 500
    d_hload = 1. / n_steps_hload
    o3.integrator.LoadControl(osi, d_hload, num_iter=10)
    # o3.rayleigh.Rayleigh(osi, a0, a1, 0.0, 0.0)
    o3.analysis.Static(osi)

    o3.record(osi)
    dmoms = np.diff(mom_demands)
    rot = []
    mom = []
    sfs_f = [[], []]
    sfs_d = [[], []]
    for dmom in dmoms:
        # Start horizontal load
        ts0 = o3.time_series.Linear(osi, factor=1)
        o3.pattern.Plain(osi, ts0)
        # o3.Load(osi, bot_fd_node, [0, 0, -dmom])
        o3.Load(osi, top_ss_node, [dmom / bd.h_eff, 0, 0])

        for i in range(n_steps_hload):
            o3.analyze(osi, num_inc=1)
            rot.append(o3.get_node_disp(osi, bot_fd_node, o3.cc.DOF2D_ROTZ))
            o3.gen_reactions(osi)
            mom.append(o3.get_ele_response(osi, fd_vert_ele, 'force')[2])
            if rec_ext_springs:
                sfs_f[0].append(o3.get_ele_response(osi, bnwf.sf_eles[0], 'force'))
                sfs_f[1].append(o3.get_ele_response(osi, bnwf.sf_eles[-1], 'force'))
                sfs_d[0].append(o3.get_ele_response(osi, bnwf.sf_eles[0], 'deformation'))
                sfs_d[1].append(o3.get_ele_response(osi, bnwf.sf_eles[-1], 'deformation'))
            # mom.append(o3.get_node_reaction(osi, bot_fd_node, o3.cc.DOF2D_ROTZ))
        o3.load_constant(osi, time=0)
    if rec_ext_springs:
        return np.array(mom), -np.array(rot), np.array(sfs_f), np.array(sfs_d)
    return np.array(mom), -np.array(rot)


def create(show=0, save=0):
    ecp = sm.load_json(ap.MODULE_DATA_PATH + 'pier/pier_ecp.json')
    bd = ecp['building'][1]
    sp = ecp['soil_profile'][1]
    sl = sp.layer_objects[0]
    fd = bd.fd
    gf.capacity_salgado_2008(sl, fd)
    fd_cap = fd.q_ult * fd.area
    print('N_ult: ', fd_cap)
    bf, ax = plt.subplots(nrows=1)
    ax = [ax]
    rots = [0, 0.05]

    fd_load = (bd.mass_eff + fd.mass) * 9.8
    print('N_load: ', fd_load)
    print('FOS: ', fd_cap / fd_load)

    mom_cap = fd_load * fd.lip / 2 * (1 - fd_load / fd_cap)
    pro_params = [2, 0., 0.15]
    mom, rot = get_mom_from_rot_bnwf(bd, sl, mtype='pro', dettach=True, b=0.001, target_rots=rots, pro_params=pro_params)

    ax[0].plot(rot, mom, label='Nonlinear-w-dettach (Clay)', ls=':', c=cbox(0))
    ax[0].axhline(mom_cap, c='k', label='Approx. moment cap.')


    ef.revamp_legend((ax[0]), loc='lower center')
    # ef.xy(ax[0], x_origin=True, y_origin=True)
    plt.tight_layout()
    name = __file__.replace('.py', '')
    name = name.split("figure_")[-1]
    extension = ''

    if save:
        ffp = ap.PUB_FIG_PATH + name + extension + ops.PUB_FIG_FILE_TYPE
        bf.savefig(ffp, dpi=ops.PUB_FIG_DPI)
        print(ef.latex_for_figure(ap.FIG_FOLDER, name + extension, ops.PUB_FIG_FILE_TYPE))
    if show:
        plt.show()


if __name__ == '__main__':
    create(show=1, save=0)

