from .. import logs
from .. import config
import numpy as np

from .datasets import load_from_standard_file, load
from .grids import load_incgrid_from_grid, load_listgrid_from_incgrid, div_on_incgrid
from .terms import TERMS
from .laws import LAWS


def initialise_original_dataset(input_filename, run_config):
    logs.getLogger(__name__).info("Initialisation original data INIT")
    original_dataset, laws, terms = load_from_standard_file(input_filename)
    logs.getLogger(__name__).info("Initialisation original data END")
    return original_dataset, laws, terms


def initialise_output_dataset(incremental_grid, original_dataset, laws, terms):
    logs.getLogger(__name__).info(f"Initialisation output data INIT")
    params = {}
    params['laws'] = laws
    params['terms'], params['coeffs'] = list_terms_and_coeffs(laws, terms, original_dataset.params)
    params['state'] = {'index': 0, 'list': 'prim'}
    grid = grid_prim_and_sec(params['terms'], incremental_grid)
    quantities = init_ouput_quantities(grid.coords['listprim'], grid.coords['listsec'], params['terms'])
    output_dataset = load(quantities=quantities, grid=grid, params=params)
    logs.getLogger(__name__).info(f"Initialisation output data END")
    return output_dataset


def list_terms_and_coeffs(laws, terms, physical_params):
    list_terms = terms.copy()
    coeffs = {}
    for law in laws:
        terms_for_law, coeffs[law] = LAWS[law].terms_and_coeffs(physical_params)
        list_terms.append(*terms_for_law)
    return list(set(list_terms)), coeffs


def grid_prim_and_sec(terms, incremental_grid):
    nb_sec_by_dirr = 0
    for term in terms:
        if term.startswith('flux'):
            nb_sec_by_dirr = 1
    return load_listgrid_from_incgrid(coord=incremental_grid.kind.split('_')[0],
                                      incgrid=incremental_grid,
                                      nb_sec_by_dirr=nb_sec_by_dirr)


def init_ouput_quantities(listprim, listsec, terms):
    output_quantities = {}
    Nprim = len(listprim)
    Nsec = len(listsec)
    for term in terms:
        if term.startswith('flux'):
            output_quantities[term] = [np.zeros((Nprim, 3)), np.zeros((Nsec, 3))]
        else:
            output_quantities[term] = [np.zeros((Nprim, 3))]
    return output_quantities


def calc_terms(dataset, output_dataset, run_config, save, backup_setp=10):
    # dataset contient au moins les quantités nécessaires au calcul
    # grid_prim = [Np][3] et grid_sec = [Ns][3] contiennent des vecteurs à calculer sous le format [x,y,z]
    # terms contient les noms des termes à calculer 
    # sortie : {[term_scalaire]:[Np],array([term_flux]):[array([Np,3]),array([Ns,3])]}

    terms = output_dataset.params['terms']
    output_quantities = output_dataset.quantities
    saved_list = output_dataset.params['state']['list']

    logs.getLogger(__name__).info("INIT Calculation of the correlation functions")
    Ndat = dataset.grid.N

    if saved_list == 'prim':
        state_index = output_dataset.params['state']['index']
        vectors = [(index, vector) for index, vector in enumerate(output_dataset.grid.coords['listprim']) if
                   (index >= state_index) and ((index % run_config.size) == run_config.rank)]
        for index, vector in vectors:
            for term in terms:
                output_quantities[term][0][state_index + index] = TERMS[term].calc(vector, Ndat,
                                                                                   **dataset.quantities)
            if (index % (run_config.size * backup_setp)) == 0:
                run_config.barrier()  # TODO check this
                logs.getLogger(__name__).info(f'... End {index} of listprim')
                output_dataset.params['state']['index'] = index + 1
                save(output_dataset, 'data_output', rank=f"{run_config.rank}")

        output_dataset.params['state']['index'] = 0
        saved_list = 'sec'

    if saved_list == 'sec':
        terms_flux = list(filter(lambda e: e.startswith('flux'), terms))
        state_index = output_dataset.params['state']['index']
        vectors = [(index, vector) for index, vector in enumerate(output_dataset.grid.coords['listsec']) if
                   (index >= state_index) and ((index % run_config.size) == run_config.rank)]
        for index, vector in vectors:
            for term in terms_flux:
                output_quantities[term][1][state_index + index] = TERMS[term].calc(vector, Ndat,
                                                                                   **dataset.quantities)
            if (index % (run_config.size * backup_setp)) == 0:
                run_config.barrier()  # TODO check this
                logs.getLogger(__name__).info(f'... End {index} of listsec')
                output_dataset.params['state']['index'] = index + 1
                output_dataset.params['state']['list'] = 'sec'
                save(output_dataset, 'data_output', rank=f"{run_config.rank}")

    run_config.barrier()
    del (output_dataset.params['state'])
    logs.getLogger(__name__).info("END Calculation of a correlation functions")


def reduction_output(quantities, run_config):
    logs.getLogger(__name__).info(f"Reduction output data INIT")
    output = {}
    for k in quantities.keys():
        output[k] = run_config.reduce(quantities[k])
    logs.getLogger(__name__).info(f"Reduction output data END")
    return output


def calc_exact_laws_from_config(run_config, backup):
    '''
    config_file example:
        [INPUT_DATA]
        path = ../../data_process/TestEff_CGL2/
        name = OCA_CGL2_cycle0_TestEff_bin2

        [OUTPUT_DATA]
        path = ./
        name = EL_logcyl40_cls100


        [RUN_PARAMS]
        config = NOP
        numbap = O
        nblayer = 8
        nbbuf = 4
        backup = None

        [GRID_PARAMS]
        Nmax_scale = 40
        Nmax_list = 100
        kind = cls
        coord = logcyl
    '''

    # translate information useful for the computation
    if config.use_reduced_datasets.get():
        input_filename = f"{config.preprocess_output_data_path.get()}/{config.preprocess_output_data_name.get()}_bin{config.preprocess_output_data_reduction.get()}.h5"
    else:
        input_filename = f"{config.preprocess_output_data_path.get()}/{config.preprocess_output_data_name.get()}.h5"
    output_filename = f"{config.computation_output_path.get()}/{config.preprocess_output_data_name.get()}_{config.computation_output_name.get()}.h5"
    input_grid = {}
    input_grid["coord"] = config.grid_coords.get()
    input_grid["kind"] = config.grid_kind.get()
    input_grid["Nmax_scale"] = config.grid_n_max_scale.get()
    input_grid["Nmax_list"] = config.grid_n_max_list.get()

    # Init Original_dataset
    if backup.already:
        original_dataset = backup.download('data_origin')
    else:
        original_dataset, laws, terms = initialise_original_dataset(input_filename, run_config)
        if run_config.rank == 0:
            backup.save(original_dataset, 'data_origin')
    original_dataset.check('original_dataset')

    # Init Incremental_grid
    if backup.already:
        incremental_grid = backup.download('inc_grid')
    else:
        incremental_grid = load_incgrid_from_grid(original_grid=original_dataset.grid, **input_grid)
        if run_config.rank == 0:
            backup.save(incremental_grid, 'inc_grid')

    # Init Output_dataset 
    if backup.already:
        output_dataset = backup.download('data_output', rank=f"{run_config.rank}")
    else:
        output_dataset = initialise_output_dataset(incremental_grid, original_dataset, laws, terms)
        backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
    output_dataset.check('output_dataset')

    # ## CALCUL LOI EXACTE
    if 'state' in output_dataset.params:
        calc_terms(
            dataset=original_dataset,
            output_dataset=output_dataset,
            run_config=run_config,
            save=backup.save
        )

    backup.save(output_dataset, 'data_output', rank=f"{run_config.rank}")
    # ## DIVERGENCE
    div_quantities = div_on_incgrid(incremental_grid, output_dataset)
    for k in div_quantities.keys():
        output_dataset.quantities[k] = [div_quantities[k]]

    # ## RASSEMBLEMENT
    output_dataset.quantities = reduction_output(output_dataset.quantities, run_config)
    backup.save(output_dataset, 'data_output_final', rank=f"{run_config.rank}")

    # # ## Enregistrement données finales
    # logging.getLogger(__name__).info(f"Record final result in {output_filename} INIT")
    # if mpi.rank == 0:
    #     output_dataset.record_to_h5file(output_filename)
    #     with h5.File(output_filename, "a") as file:
    #         file.create_group("param_origin")
    #         for k in original_dataset.params.keys():
    #             file["param_origin"].create_dataset(k, data=original_dataset.params[k])
    # logging.getLogger(__name__).info(f"Record final result END")
