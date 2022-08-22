"""


mpirun -n 14 --use-hwthread-cpus python ./scripts/gen_lib_sams.py output/test_2022-06-27


"""

import argparse
import os
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
# import h5py
from mpi4py import MPI

import holodeck as holo
import holodeck.sam
import holodeck.logger
from holodeck.constants import YR, MSOL, GYR  # noqa


class Parameter_Space:

    def __init__(
        self,
        gsmf_phi0=[-3.35, -2.23, 7],
        # gsmf_alpha0=[-1.56, -0.92, 5],
        mmb_amp=[0.39e9, 0.61e9, 11], mmb_plaw=[1.01, 1.33, 13]
    ):

        self.gsmf_phi0 = np.linspace(*gsmf_phi0)
        # self.gsmf_alpha0 = np.linspace(*gsmf_alpha0)
        self.mmb_amp = np.linspace(*mmb_amp)
        self.mmb_plaw = np.linspace(*mmb_plaw)
        pars = [
            self.gsmf_phi0,
            # self.gsmf_alpha0,
            self.mmb_amp,
            self.mmb_plaw
        ]
        self.names = [
            'gsmf_phi0',
            # 'gsmf_alpha0',
            'mmb_amp',
            'mmb_plaw'
        ]
        self.params = np.meshgrid(*pars, indexing='ij')
        self.shape = self.params[0].shape
        self.size = np.product(self.shape)
        self.params = np.moveaxis(self.params, 0, -1)

        pass

    def number_to_index(self, num):
        idx = np.unravel_index(num, self.shape)
        return idx

    def index_to_number(self, idx):
        num = np.ravel_multi_index(idx, self.shape)
        return num

    def param_dict_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.params[idx]
        rv = {nn: pp for nn, pp in zip(self.names, pars)}
        return rv

    def params_for_number(self, num):
        idx = self.number_to_index(num)
        pars = self.params[idx]
        return pars

    def sam_for_number(self, num):
        params = self.params_for_number(num)

        gsmf_phi0, mmb_amp, mmb_plaw = params

        gsmf = holo.sam.GSMF_Schechter(phi0=gsmf_phi0)
        gpf = holo.sam.GPF_Power_Law()
        gmt = holo.sam.GMT_Power_Law()
        mmbulge = holo.relations.MMBulge_KH2013(mamp=mmb_amp*MSOL, mplaw=mmb_plaw)

        sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge)
        return sam


comm = MPI.COMM_WORLD

BEG = datetime.now()

# DEBUG = False

# ---- Fail on warnings
# # err = 'ignore'
# err = 'raise'
# np.seterr(divide=err, invalid=err, over=err)
# warn_err = 'error'
# # warnings.filterwarnings(warn_err, category=UserWarning)
# warnings.filterwarnings(warn_err)

# ---- Setup ArgParse

parser = argparse.ArgumentParser()
parser.add_argument('output', metavar='output', type=str,
                    help='output path [created if doesnt exist')

# parser.add_argument('-r', '--reals', action='store', dest='reals', type=int,
#                     help='number of realizations', default=10)
# parser.add_argument('-s', '--shape', action='store', dest='shape', type=int,
#                     help='shape of SAM grid', default=50)
# parser.add_argument('-t', '--threshold', action='store', dest='threshold', type=float,
#                     help='sample threshold', default=100.0)
# parser.add_argument('-d', '--dur', action='store', dest='dur', type=float,
#                     help='PTA observing duration [yrs]', default=20.0)
# parser.add_argument('-c', '--cad', action='store', dest='cad', type=float,
#                     help='PTA observing cadence [yrs]', default=0.1)
# parser.add_argument('-d', '--debug', action='store_true', default=False, dest='debug',
#                     help='run in DEBUG mode')
parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                    help='verbose output [INFO]')
# parser.add_argument('--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()
args.NUM_REALS = 30
args.PTA_DUR = 15.0 * YR
args.PTA_CAD = 0.1 * YR


BEG = comm.bcast(BEG, root=0)

this_fname = os.path.abspath(__file__)
head = f"holodeck :: {this_fname} : {str(BEG)} - rank: {comm.rank}/{comm.size}"
head = "\n" + head + "\n" + "=" * len(head) + "\n"
if comm.rank == 0:
    print(head)

log_name = f"holodeck__gen_lib_sams_{BEG.strftime('%Y%m%d-%H%M%S')}"
if comm.rank > 0:
    log_name = f"_{log_name}_r{comm.rank}"

PATH_OUTPUT = Path(args.output).resolve()
if not PATH_OUTPUT.is_absolute:
    PATH_OUTPUT = Path('.').resolve() / PATH_OUTPUT
    PATH_OUTPUT = PATH_OUTPUT.resolve()

if comm.rank == 0:
    PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

comm.barrier()

# ---- Setup Logger ----

fname = f"{PATH_OUTPUT.joinpath(log_name)}.log"
log_lvl = holo.logger.INFO if args.verbose else holo.logger.WARNING
tostr = sys.stdout if comm.rank == 0 else None
LOG = holo.logger.get_logger(name=log_name, level_stream=log_lvl, tofile=fname, tostr=tostr)
LOG.info(head)
LOG.info(f"Output path: {PATH_OUTPUT}")
LOG.info(f"        log: {fname}")

SPACE = Parameter_Space() if comm.rank == 0 else None
SPACE = comm.bcast(SPACE, root=0)

# ------------------------------------------------------------------------------
# ----    Methods
# ------------------------------------------------------------------------------


def main():
    bnum = 0
    LOG.info(f"{SPACE=}, {id(SPACE)=}")
    npars = SPACE.size
    nreals = args.NUM_REALS

    # # -- Load Parameters from Input File
    # params = None
    # if comm.rank == 0:
    #     input_file = os.path.abspath(input_file)
    #     if not os.path.isfile(input_file):
    #         raise ValueError(f"input_file '{input_file}' does not exist!")

    #     if not os.path.isdir(output_path):
    #         raise ValueError(f"output_path '{output_path}' does not exist!")

    #     params = _parse_params_file(input_file)

    #     # Copy input file to output directory
    #     fname_input_copy = os.path.join(output_path, "input_params.txt")
    #     # If file already exists, rename it to backup
    #     fname_backup = zio.modify_exists(fname_input_copy)
    #     if fname_input_copy != fname_backup:
    #         print(f"Moving previous parameters file '{fname_input_copy}' ==> '{fname_backup}'")
    #         shutil.move(fname_input_copy, fname_backup)
    #     print(f"Saving copy of input parameters file to '{fname_input_copy}'")
    #     shutil.copy2(input_file, fname_input_copy)

    # Distribute all parameters to all processes
    # params = comm.bcast(params, root=0)
    bnum = _barrier(bnum)

    # Split and distribute index numbers to all processes
    if comm.rank == 0:
        # indices = range(npars*nreals)
        indices = range(npars)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)
        num_ind_per_proc = [len(ii) for ii in indices]
        # LOG.info(f"{npars=}, {nreals=}, total={npars*nreals} || ave runs per core = {np.mean(num_ind_per_proc)}")
        LOG.info(f"{npars=}, {nreals=} || ave runs per core = {np.mean(num_ind_per_proc)}")
    else:
        indices = None
    indices = comm.scatter(indices, root=0)
    bnum = _barrier(bnum)
    # prog_flag = (comm.rank == 0)
    iterator = holo.utils.tqdm(indices) if comm.rank == 0 else np.atleast_1d(indices)

    for ind in iterator:
        # Convert from 1D index into 2D (param, real) specification
        # param, real = np.unravel_index(ind, (npars, nreals))
        # LOG.info(f"rank:{comm.rank} index:{ind} => {param=} {real=}")
        param = ind

        # # - Check if all output files already exist, if so skip
        # key = pipeline(progress=prog_flag, key_only=True, **pars)
        # if number_output:
        #     digits = int(np.floor(np.log10(999))) + 1
        #     key = f"{ind:0{digits:d}d}" + "__" + key

        # fname_plot_all, fname_plot_gwb = _save_plots_fnames(output_path, key)
        # fname_data = _save_data_fname(output_path, key)
        # fnames = [fname_plot_all, fname_plot_gwb, fname_data]
        # if np.all([os.path.exists(fn) and (os.path.getsize(fn) > 0) for fn in fnames]):
        #     print(f"\tkey: '{key}' already complete")
        #     continue

        try:
            run_sam(param, None, PATH_OUTPUT)
        except Exception as err:
            logging.warning(f"\n\nWARNING: error on rank:{comm.rank}, index:{ind}")
            logging.warning(err)
            import traceback
            traceback.print_exc()
            print("\n\n")

    end = datetime.now()
    print(f"\t{comm.rank} done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}")
    bnum = _barrier(bnum)
    if comm.rank == 0:
        end = datetime.now()
        tail = f"Done at {str(end)} after {str(end-BEG)} = {(end-BEG).total_seconds()}"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n")

    return


def run_sam(pnum, real, path_output):

    iterator = range(args.NUM_REALS)
    if comm.rank == 0:
        iterator = holo.utils.tqdm(iterator, leave=False)

    for real in iterator:

        fname = f"lib_sams__p{pnum:06d}_r{real:03d}.npz"
        fname = os.path.join(path_output, fname)
        if os.path.exists(fname):
            LOG.warning(f"File {fname} already exists.")
            continue

        fobs = holo.utils.nyquist_freqs(args.PTA_DUR, args.PTA_CAD)

        sam = SPACE.sam_for_number(pnum)

        gff, gwf, gwb = holo.sam.sampled_gws_from_sam(
            sam, fobs, sample_threshold=10.0,
            # cut_below_mass=1e7*MSOL, limit_merger_time=4*GYR
        )

        legend = SPACE.param_dict_for_number(pnum)
        np.savez(fname, fobs=fobs, gff=gff, gwb=gwb, gwf=gwf, pnum=pnum, real=real, **legend)
        LOG.info(f"Saved to {fname} after {(datetime.now()-BEG)} (start: {BEG})")

    return


def _barrier(bnum):
    LOG.debug(f"barrier {bnum}")
    comm.barrier()
    bnum += 1
    return bnum


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
    sys.exit(0)
