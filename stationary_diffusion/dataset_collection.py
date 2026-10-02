import os
import argparse
import numpy as np

import dataset_generation, conductivity_models, source_models

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
        "-save_path": {
            "required": True,
            "type": str,
            "help": "folder where to store dataset"
        },
        "-N_samples": {
            "default": 1000,
            "type": int,
            "help": "number of samples"
        },
        "-N_grid": {
            "default": 128,
            "type": int,
            "help": "number of discretization points on the grid"
        },
        "-f_model": {
            "default": 'constant',
            "type": str,
            "choices": ["smooth", "rough", "constant"],
            "help": "right-hand side random field"
        },
        "-bc": {
            "default": 'Dirichlet',
            "type": str,
            "choices": ["Dirichlet", "Neumann"],
            "help": "boundary conditions"
        },
        "-k_model": {
            "default": 'isotropic_2',
            "type": str,
            "choices": [f"isotropic_{i+1}" for i in range(4)] + [f"anisotropic_{i+1}" for i in range(4)],
            "help": "conductivity random field"
        },
        "-seed": {
            "default": 11998,
            "type": int,
            "help": "random seed"
        },
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    name = "_".join([str(args[a]) for a in filter(lambda x: x != "save_path", args.keys())])
    rng = np.random.default_rng(args['seed'])
    sources = {
        "smooth": source_models.smooth_source,
        "rough": source_models.rough_source,
        "constant": None
    }
    os.makedirs(args['save_path'], exist_ok=True)
    data = dataset_generation.generate_dataset(args['N_samples'], args['N_grid'], args['bc'], getattr(conductivity_models, args['k_model']), sources[args['f_model']], rng)
    np.savez(f"{args['save_path']}/{name}.npz", **data)
