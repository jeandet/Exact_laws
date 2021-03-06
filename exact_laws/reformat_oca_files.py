from exact_laws.preprocessing.process_on_oca_files import reformat_oca_files
import argparse
import logging
from datetime import datetime

version = "09/07/2022"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='example_input_process.ini')
parser.add_argument("-q", "--list-quantities", help="List available quantities", action="store_true")
args = parser.parse_args()

logging.basicConfig(filename=f"reformat_oca_files_{datetime.now().strftime('%d%m%Y_%H%M')}.log", 
                    level=logging.DEBUG, 
                    format='%(asctime)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )

if __name__ == "__main__":
    
    if args.list_quantities:
        from exact_laws.preprocessing.quantities import QUANTITIES
        print(list(QUANTITIES.keys()))
        exit(0)
        
    logging.info(f"Run of {__file__} version {version}\n")
    reformat_oca_files(config_file=args.config_file)
    logging.info(f"Exit")
