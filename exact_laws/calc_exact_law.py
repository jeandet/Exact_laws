from exact_laws import logs
from exact_laws.config import load_config
from exact_laws import config
import argparse
import configparser
from exact_laws.running_tools.run_config_wrap import RunConfig
from exact_laws.running_tools.backup_wrap import Backup
from exact_laws.exact_laws_calc import calc_exact_laws_from_config

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='example_input_calc.txt')
parser.add_argument("-q", "--list-laws", help="List available laws", action="store_true")
args = parser.parse_args()

version = "09/07/2022"

if __name__ == "__main__":

    if args.list_laws:
        from exact_laws.exact_laws_calc.laws import LAWS

        print(list(LAWS.keys()))
        exit(0)

    load_config(args.config_file)
    run_config = RunConfig(with_mpi=config.with_mpi.get(), numba_parallel=config.numba_parallel.get(),
                           compat_mode=config.compat_mode.get())
    run_config.configure_log('calc_exact_law')

    # configure the potential parallelisation process (add old way params)
    if run_config.compat_mode:
        run_config.set_nblayer(config.nblayers.get())
        run_config.set_bufnum(config.nbbuff.get())

    # configure the saving process (always valid way)
    backup = Backup(config.restart_checkpoint.get(), run_config.time_deb, run_config.rank)

    logs.getLogger(__name__).info(f"Run of {__file__} version {version}\n")
    calc_exact_laws_from_config(run_config=run_config, backup=backup)
    logs.getLogger(__name__).info(f"Exit")
