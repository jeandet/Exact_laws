from exact_laws.preprocessing.process_on_oca_files import reformat_oca_files
import argparse
from exact_laws import logs
from datetime import datetime
import traceback
from exact_laws.config import load_config

version = "09/07/2022"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='example_input_process.ini')
parser.add_argument("-q", "--list-quantities", help="List available quantities", action="store_true")
args = parser.parse_args()

logs.setup(log_filename=f"reformat_oca_files_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log", log_level=logs.INFO)

if __name__ == "__main__":
    try:
        if args.list_quantities:
            from exact_laws.preprocessing.quantities import QUANTITIES

            print(list(QUANTITIES.keys()))
            exit(0)

        load_config(args.config_file)
        logs.getLogger(__name__).info(f"Run of {__file__} version {version}\n")
        reformat_oca_files()
        logs.getLogger(__name__).info(f"Exit")
    except:
        logs.getLogger(__name__).error(traceback.format_exc())
