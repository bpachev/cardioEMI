from mpi4py   import MPI
from typing   import Union
from petsc4py import PETSc
#import scipy.sparse as sparse
#import scipy.io as sio
import numpy        as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import sys
import ufl
import os
import yaml
import pickle 


# Assign intial membrane potential
class Read_input_field:
    def __init__(self, expression: Union[str, float, int]):
        self.expression = expression

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        # If expression is a number, return it directly as an array of the same shape as `x`
        if isinstance(self.expression, (int, float)):
            return self.expression + 0 * x[0] # hack
        # If expression is a string, evaluate it
        elif isinstance(self.expression, str):
            return eval(self.expression, {"np": np, "ufl": ufl, "x": x})
        else:
            raise ValueError("Expression must be a string, int, or float.")                     


# Create input field based on type and value
def read_input_field(expression: Union[str, float, int], mesh=None):

    if isinstance(expression, (int, float)):
        return float(expression)    

    elif isinstance(expression, str):
        # Create a context for eval, including necessary objects
        context = {"np": np, "ufl": ufl}
        
        # If mesh is provided, add x as the spatial coordinate
        if mesh is not None:
            x = ufl.SpatialCoordinate(mesh)
            context["x"] = x

        # Now, evaluate the expression in this context
        return eval(expression, context)
    
    else:
        raise ValueError("Expression must be a string, int, or float.")

# read yml file
def read_input_file(input_yml_file):
        
        # read input yml file
        with open(input_yml_file, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)        
            
        input_parameters = {
                'C_M': 1.0, 'cuda': False, 'sigma_i': 1.0,
                'sigma_e': 1.0, 'R_g': 1.0, 'fem_order': 1,
                'mesh_conversion_factor': 1.0,
                'pc_type': 'hypre', 'ksp_type': 'cg', 'ksp_rtol': 1e-8,
                'save_output': False, 'save_interval': 1, 'verbose': False
        } 

        input_parameters.update(config)
        input_parameters['P'] = input_parameters['fem_order']

        fnames = ['mesh_file', 'tags_dictionary_file']
        required_parameters = ['dt', 'out_name'] + fnames
        for param in required_parameters:
            if param not in config:
                raise ValueError(f"Missing required field '{param}'")

        ######### geometry #########
        for fname in ['mesh_file', 'tags_dictionary_file']:
            check_if_file_exists(config[fname])

        # get ECC tag if specified, otherwise use the minimum
        if 'ECS_TAG' not in config:                  
            
            with open(config["tags_dictionary_file"], "rb") as f:
                membrane_tags = pickle.load(f)

            input_parameters['ECS_TAG'] = min(membrane_tags.keys())          
            
            # Read input file 
            if MPI.COMM_WORLD .rank == 0: print("ECS tag not specified, using minimum one:", input_parameters['ECS_TAG'])  
            
                
        ######### problem #########
        
        if 'time_steps' in config: 
            input_parameters['time_steps'] = config['time_steps']            
        elif 'T' in config:            
            input_parameters['time_steps'] = int(config['T']/config['dt'])        
        else:
            raise SyntaxError(f'INPUT ERROR: provide final time T or time_steps in input .yml file.')
            
        # ionic model 
        if 'ionic_model' in config: 
            input_parameters['ionic_model'] = config['ionic_model']

            if isinstance(input_parameters['ionic_model'], dict):
                if input_parameters['ionic_model'].keys() != {"intra_intra", "intra_extra"}:
                    raise KeyError(f"Use intra_intra and intra_extra input entries!")

        else:
            print('WARNING: setting default passive ionic model')
            input_parameters['ionic_model'] = "Passive"        
            
        return input_parameters
     

def update_status(message):
    sys.stdout.write(f'\r{message}')
    sys.stdout.flush()


def check_if_file_exists(file_path):
    if not os.path.exists(file_path):        
        print(f"The file '{file_path}' does not exist.")
        exit()


def norm_2(vec):
    return sqrt(dot(vec,vec))


def dump(thing, path):
    name = path.split("/")[-1]
    if isinstance(thing, PETSc.Vec):
        assert np.all(np.isfinite(thing.array))
        return np.save(path, thing.array)
    m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
    assert np.all(np.isfinite(m.data))
    return np.save(path, np.c_[m.row, m.col, m.data]), sio.savemat(path, {name: m})


def common_elements(set1, set2):    
    return set1.intersection(set2)

def plot_sparsity_pattern(A):
    ai, aj, av = A.getValuesCSR()
    rows, cols = A.getSize()
    sparse_matrix = sparse.csr_matrix((av, aj, ai), shape=(rows, cols))
    plt.figure(figsize=(8, 8))
    plt.spy(sparse_matrix, markersize=1)
    plt.title("Sparsity Pattern of the Matrix")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()
