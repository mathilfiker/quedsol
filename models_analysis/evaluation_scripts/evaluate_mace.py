# Example script on how to evaluate a trained MACE model on new molecules
from ase import Atoms
from ase.io import write, read
import pandas as pd
from mace.calculators import MACECalculator

# Evaluation set, here is used the example set
test_data = read('eat_delta_example.extxyz', index = ':') 

# Model to evaluate, , here eAT_30 as example
# selectn device = 'cuda' for faster computation
calculator = MACECalculator(model_paths='../../enn/eat_delta_70.model', device ='cpu', default_dtype="float32")

# Get the MACE prediction for the given molecule
def get_pred(molecule):
    molecule.set_calculator(calculator)
    nequip = molecule.get_total_energy()
    return nequip

# Get the real value of the considered property for the given molecule
def get_energy_from_mol(molecule):
    return molecule.info['REF_energy']
    
pred = []
true = []

for i in test_data:
    x = get_pred(i)
    y = get_energy_from_mol(i)
    pred.append(x)
    true.append(y)

# Create dataframe with real values and predictions
df = {
    'molecule': range(len(test_data)),
    'original_value': true,
    'prediction': pred,
}
df =pd.DataFrame(df)
df.to_json('example_mace.json') # Save the dataframe
