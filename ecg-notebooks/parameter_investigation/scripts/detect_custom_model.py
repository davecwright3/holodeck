import numpy as np
import holodeck as holo
import argparse
from holodeck import detstats
from datetime import datetime
from tqdm import tqdm
import os

from holodeck.sams import cyutils as sam_cyutils
from holodeck.constants import YR


# sample
DEF_SHAPE = None
DEF_NLOUDEST = 10
DEF_NREALS = 100
DEF_NFREQS = 40
DEF_NVARS = 21
DEF_PTA_DUR = holo.librarian.DEF_PTA_DUR

# pta calibration
DEF_NSKIES = 100
DEF_NPSRS = 40
DEF_RED_AMP = None
DEF_RED_GAMMA = None
DEF_RED2WHITE = None

DEF_TOL = 0.01
DEF_MAXBADS = 5
GAMMA_RHO_GRID_PATH = '/Users/emigardiner/GWs/holodeck/output/rho_gamma_grids' # modify for system
ANATOMY_PATH = '/Users/emigardiner/GWs/holodeck/output/anatomy'

# settings to vary
DEF_CONSTRUCT = False
DEF_DETSTATS = False


def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', action='store', type=str,
                        help="target parameter to vary")
    parser.add_argument('param_space', type=str,
                        help="Parameter space class name, found in 'holodeck.param_spaces'.")
    # parser.add_argument('--grid_path', action='store', dest ='grid_path', type=str, default=GAMMA_RHO_GRID_PATH,
    #                     help="gamma-rho interpolation grid path")
    
    # sample models setup
    
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int, default=DEF_NFREQS,
                        help='number of frequency bins')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int, default=DEF_NREALS,
                        help='number of strain realizations')
    parser.add_argument('-l', '--nloudest', action='store', dest='nloudest', type=int, default=DEF_NLOUDEST,
                        help='number of loudest single sources')
    parser.add_argument('-v', '--nvars', action='store', dest='nvars', type=int, default=DEF_NVARS,
                        help='number of variations on target param')
    parser.add_argument('--shape', action='store', dest='shape', type=int, default=DEF_SHAPE,
                        help='sam shape')
    parser.add_argument('-d', '--dur', action='store', dest='pta_dur', type=int, default=DEF_PTA_DUR,
                        help='pta duration in yrs')


    
    # pta setup
    parser.add_argument('-p', '--npsrs', action='store', dest='npsrs', type=int, default=DEF_NPSRS,
                        help='number of pulsars in pta')
    parser.add_argument('-s', '--nskies', action='store', dest='nskies', type=int, default=DEF_NSKIES,
                        help='number of ss sky realizations')
    parser.add_argument('--red_amp', action='store', dest='red_amp', type=float, default=DEF_RED_AMP,
                        help='Red noise amplitude')
    parser.add_argument('--red_gamma', action='store', dest='red_gamma', type=float, default=DEF_RED_GAMMA,
                        help='Red noise gamma')
    parser.add_argument('--red2white', action='store', dest='red2white', type=float, default=DEF_RED2WHITE,
                        help='Red noise amplitude to white noise amplitude ratio.')
    
    # pta calibration settings
    parser.add_argument('--sigstart', action='store', dest='sigstart', type=float, default=1e-7,
                        help='starting sigma if for realization calibration')
    parser.add_argument('--sigmin', action='store', dest='sigmin', type=float, default=1e-10,
                        help='sigma minimum for calibration')
    parser.add_argument('--sigmax', action='store', dest='sigmax', type=float, default=1e-4,
                        help='sigma maximum for calibration')
    parser.add_argument('--thresh', action='store', dest='thresh', type=float, default=0.5,
                        help='threshold for detection fractions')
    parser.add_argument('-t', '--tol', action='store', dest='tol', type=float, default=DEF_TOL,
                         help='tolerance for BG DP calibration')
    parser.add_argument('-b', '--maxbads', action='store', dest='maxbads', type=int, default=DEF_MAXBADS,
                         help='number of bad sigmas to try before expanding the search range')
    
    # general settings
    parser.add_argument('--construct', action='store_true', default=DEF_CONSTRUCT,
                        help='construct data and detstats for each varying param')
    parser.add_argument('--detstats', action='store_true', default=DEF_DETSTATS,
                        help='construct detstats, using saved data')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='print steps along the way')
    
    # rarely need changing
    parser.add_argument('--snr_cython', action='store_true', default=True,
                        help='Use cython for ss snr calculations')
    parser.add_argument('--save_ssi', action='store_true', default=True,
                        help="Save 'gamma_ssi', the detprob of each single source.")
    parser.add_argument('--clbrt', action='store_true', default=True,
                        help="Whether or not to calibrate the PTA for individual realizations.")
    parser.add_argument('--grid_path', action='store', dest ='grid_path', type=str, default=GAMMA_RHO_GRID_PATH,
                        help="gamma-rho interpolation grid path")
    parser.add_argument('--anatomy_path', action='store', dest ='anatomy_path', type=str, default=ANATOMY_PATH,
                        help="path to load and save anatomy files")
    parser.add_argument('--load_file', action='store', dest ='load_file', type=str, default=None,
                        help="file to load sample data and params, excluding .npz suffice")
    parser.add_argument('--save_file', action='store', dest ='save_file', type=str, default=None,
                        help="file to save sample data, excluding .npz suffix")
    
    args = parser.parse_args()
    return args


def run_model(sam, hard, nreals, nfreqs, nloudest=5,
              pta_dur = DEF_PTA_DUR,
              gwb_flag=True, details_flag=False, singles_flag=False, params_flag=False):
    """Run the given modeling, storing requested data
    """
    fobs_cents, fobs_edges = holo.utils.pta_freqs(pta_dur)
    if nfreqs is not None:
        fobs_edges = fobs_edges[:nfreqs+1]
        fobs_cents = fobs_cents[:nfreqs]
    fobs_orb_cents = fobs_cents / 2.0     # convert from GW to orbital frequencies
    fobs_orb_edges = fobs_edges / 2.0     # convert from GW to orbital frequencies

    data = dict(fobs_cents=fobs_cents, fobs_edges=fobs_edges)

    redz_final, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(
        fobs_orb_cents, sam, hard, holo.cosmo
    )
    use_redz = redz_final
    edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges]
    number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num)
    if details_flag:
        data['static_binary_density'] = sam.static_binary_density
        data['number'] = number
        data['redz_final'] = redz_final
        data['coalescing'] = (redz_final > 0.0)

        gwb_pars, num_pars, gwb_mtot_redz_final, num_mtot_redz_final = holo.librarian._calc_model_details(edges, redz_final, number)

        data['gwb_params'] = gwb_pars
        data['num_params'] = num_pars
        data['gwb_mtot_redz_final'] = gwb_mtot_redz_final
        data['num_mtot_redz_final'] = num_mtot_redz_final

    # calculate single sources and/or binary parameters
    if singles_flag or params_flag:
        nloudest = nloudest if singles_flag else 1

        vals = holo.single_sources.ss_gws_redz(
            edges, use_redz, number, realize=nreals,
            loudest=nloudest, params=params_flag,
        )
        if params_flag:
            hc_ss, hc_bg, sspar, bgpar = vals
            data['sspar'] = sspar
            data['bgpar'] = bgpar
        else:
            hc_ss, hc_bg = vals

        if singles_flag:
            data['hc_ss'] = hc_ss
            data['hc_bg'] = hc_bg

    if gwb_flag:
        gwb = holo.gravwaves._gws_from_number_grid_integrated_redz(edges, use_redz, number, nreals)
        data['gwb'] = gwb

    return data

# # construct a param_space instance, note that `nsamples` and `seed` don't matter here for how we'll use this
# pspace = holo.param_spaces.PS_Uniform_09B(holo.log, nsamples=1, sam_shape=SHAPE, seed=None)

def vary_parameter(
        target_param,    # the name of the parameter, has to exist in `param_names`
        params_list,  # the values we'll check
        nreals, nfreqs, nloudest,
        pspace,
        pars=None, debug=True, pta_dur=DEF_PTA_DUR,
        ):

    # get the parameter names from this library-space
    param_names = pspace.param_names
    num_pars = len(pspace.param_names)
    if debug: print(f"{num_pars=} :: {param_names=}")

    # choose each parameter to be half-way across the range provided by the library
    if pars is None:
        pars = 0.5 * np.ones(num_pars) 
    # Choose parameter to vary
    param_idx = param_names.index(target_param)

    data = []
    params = []
    for ii, par in enumerate(tqdm(params_list)):
        pars[param_idx] = par
        if debug: print(f"{ii=}, {pars=}")
        # _params = pspace.param_samples[0]*pars
        _params = pspace.normalized_params(pars)
        params.append(_params)
        # construct `sam` and `hard` instances based on these parameters
        sam, hard = pspace.model_for_params(_params, pspace.sam_shape)
        # run this model, retrieving binary parameters and the GWB
        _data = run_model(sam, hard, nreals, nfreqs, nloudest=nloudest, pta_dur=pta_dur,
                        gwb_flag=False, singles_flag=True, params_flag=True, details_flag=True)
        data.append(_data)

    return (data, params)



def main():
    start_time = datetime.now()
    print("-----------------------------------------")
    print(f"starting at {start_time}")
    print("-----------------------------------------")

    # set up args
    args = _setup_argparse()
    print("NREALS = %d, NSKIES = %d, NPSRS = %d, target = %s, NVARS=%d"
          % (args.nreals, args.nskies, args.npsrs, args.target, args.nvars))
    
    # set up output folder
    output_path = args.anatomy_path+f'/{args.param_space}_{args.target}_v{args.nvars}_r{args.nreals}_d{args.pta_dur}'
    # check if output folder already exists, if not, make it.
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    # set up load and save locations
    if args.load_file is None:
        load_data_from_file = output_path+'/data_params'
    else:
        load_data_from_file = args.load_file

    if args.save_file is None:
        save_data_to_file =  output_path+'/data_params'
    else:
        save_data_to_file = args.save_file

    save_dets_to_file = output_path+f'/detstats_s{args.nskies}_ssn'
    if args.red2white is not None and args.red_gamma is not None:
        save_dets_to_file = save_dets_to_file+f'_r2w{args.red2white:.1f}_rg{args.red_gamma:.1f}'
    elif args.red_amp is not None and args.red_gamma is not None:
        save_dets_to_file = save_dets_to_file+f'_ra{args.red_amp:.1e}_rg{args.red_gamma:.1f}'
    else: 
        save_dets_to_file = save_dets_to_file+f'_white'

    if args.red2white is not None and args.red_amp is not None:
        print(f"{args.red2white=} and {args.red_amp} both provided. red_amp will be overriden by red2white ratio.")

    print(f"{load_data_from_file=}.npz")
    print(f"{save_data_to_file=}.npz")
    print(f"{save_dets_to_file=}.npz")


    space_class = getattr(holo.param_spaces, args.param_space)
    
    # calculate model and/or detstats
    if args.construct or args.detstats:
        if args.construct:
            params_list = np.linspace(0,1,args.nvars)
            data, params, = vary_parameter(
                target_param=args.target, params_list=params_list,
                nfreqs=args.nfreqs, nreals=args.nreals, nloudest=args.nloudest,
                pspace = space_class(holo.log, nsamples=1, sam_shape=args.shape, seed=None),
                pta_dur=args.pta_dur)
            np.savez(save_data_to_file+'.npz', data=data, params=params) # save before calculating detstats, in case of crash
        else:
            file = np.load(load_data_from_file+'.npz', allow_pickle=True)
            print('loaded files:', file.files)
            data = file['data']
            params = file['params']
            file.close()

        fobs_cents = data[0]['fobs_cents']

        # get dsdat for each data/param
        dsdat = []
        for ii, _data in enumerate(tqdm(data)):
            if args.debug: print(f"on var {ii=} out of {args.nvars}")
            hc_bg = _data['hc_bg']
            hc_ss = _data['hc_ss']
            _dsdat = detstats.detect_pspace_model_clbrt_pta(
                fobs_cents, hc_ss, hc_bg, args.npsrs, args.nskies, 
                sigstart=args.sigstart, sigmin=args.sigmin, sigmax=args.sigmax, tol=args.tol, maxbads=args.maxbads,
                thresh=args.thresh, debug=args.debug, red_amp=args.red_amp, red_gamma=args.red_gamma, red2white=args.red2white)
            dsdat.append(_dsdat)
        np.savez(save_dets_to_file+'.npz', dsdat=dsdat, red_amp=args.red_amp, red_gamma=args.red_gamma, npsrs=args.npsrs, red2white=args.red2white) # overwrite
    else:
        print(f"Neither {args.construct=} or {args.detstats} are true. Doing nothing.")

    end_time = datetime.now()
    print("-----------------------------------------")
    print(f"ending at {end_time}")
    print(f"total time: {end_time - start_time}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

