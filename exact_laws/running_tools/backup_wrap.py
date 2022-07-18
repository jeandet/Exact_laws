import os
import pickle as pkl
from .. import logs


class Backup():
    def __init__(self, restart_checkpoint, time, rank):
        if restart_checkpoint == "":
            if rank == 0:
                os.mkdir(f"./save_{time.strftime('%d%m%Y_%H%M%S')}")  # creation of a recording folder
            self.folder = f"./save_{time.strftime('%d%m%Y_%H%M%S')}/"
            self.already = False
        else:
            self.folder = restart_checkpoint
            self.already = True

    def save(self, object, name, rank='', state=''):
        logs.getLogger(__name__).info(f"Save {name} {state} in folder {self.folder} INIT")
        filename = f"{self.folder}/{name}_rk{rank}.pkl"
        with open(filename, "wb") as f:
            pkl.dump(object, f)
        logs.getLogger(__name__).info(f"Save {name} END")

    def download(self, name, rank=''):
        logs.getLogger(__name__).info(f"Download {name} from folder {self.folder} INIT")
        filename = f"{self.folder}/{name}_rk{rank}.pkl"
        with open(filename, "rb") as f:
            output = pkl.load(f)
        logs.getLogger(__name__).info(f"Download {name} END")
        return output
