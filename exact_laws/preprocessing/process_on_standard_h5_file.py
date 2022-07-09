import os, sys
import h5py as h5
import numpy as np
from datetime import datetime

def verif_file_existence(filename, message):
    if os.path.isfile(filename):
        print(f"The file already exists. {message}")
        print(f"End: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')} \n")
        sys.stdout.flush()
        return True

def check_file(filename):
    """
    Affichage du contenu du fichier .h5 nommé "file".
    Les informations délivrées à propos des cubes de données sont la moyenne et l'écart-type.
    """
    print(f"Check file {filename} : ")
    with h5.File(filename, "r") as g:
        for param in g["param"].keys():
            print(f"\t - {param} : {np.array(g[f'param/{param}'])}")
        sys.stdout.flush()
        for quantity in g.keys():
            if not "param" in quantity:
                tab = np.sort(np.array(g[quantity]).flatten())
                print(f"\t - {quantity} : {np.mean(tab)} $\pm$ {np.std(tab)}")
                del tab
                sys.stdout.flush()

def data_binning(file_process, bin):
    """
    Vérification si le fichier contenant les données réduite existe déjà.
    Si non, enclenche le processus de création.
    Puis affiche le contenu du nouveau fichier.
    """
    print(f"Data binning beginning: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')}")
    sys.stdout.flush()
    output_filename = f"{file_process[:-3]}_bin{str(bin)}.h5"
    if verif_file_existence(output_filename, "Data binning impossible."):
        return output_filename
    else:
        bin_arrays_in_h5(file_process, output_filename, bin)
    print(f"Data binning end: {datetime.today().strftime('%d-%m-%Y %H:%M:%S')} \n")
    sys.stdout.flush()
    print(" ")
    return output_filename


def bin_an_array(tab, binning):
    """
    Input: tab(np.array à réduire), binning(int,facteur de réduction)
    Output: np.array réduit
    """
    return (
        tab.reshape(
            np.shape(tab)[0] // binning,
            binning,
            np.shape(tab)[1] // binning,
            binning,
            np.shape(tab)[2] // binning,
            binning,
        )
        .mean(-1)
        .mean(3)
        .mean(1)
    )


def bin_arrays_in_h5(filename, output_filename, binning):
    """
    Input: file_name(str, name+ext of the original file), binning(int, binning factor)
    Apply a binning on the data contained in the .h5 file named "file_name".
    Record the new dataset.
    Output: name of the new file.
    """
    with h5.File(filename, "r") as g:
        with h5.File(output_filename, "w") as f:
            for k in g.keys():
                if not "param" in k:
                    tab = np.ascontiguousarray(g[k], dtype=np.float64)
                    new_tab = bin_an_array(tab, binning)
                    f.create_dataset(k, data=new_tab)
                else:
                    f.create_group("param")
                    f["param"]["N"] = np.array(g["param"]["N"]) // binning
                    f["param"]["c"] = np.array(g["param"]["c"]) * binning
                    f["param"]["reduction"] = binning
                    keys = list(g["param"].keys())
                    keys.remove("N")
                    keys.remove("c")
                    keys.remove("reduction")
                    for kp in keys:
                        f["param"].create_dataset(kp, data=g["param"][kp])
    return 0