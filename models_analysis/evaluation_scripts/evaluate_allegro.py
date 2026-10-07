# Example script on how to evaluate a trained Allegro model on new molecules
from ase import Atoms
from nequip.ase import nequip_calculator
from ase.io import write, read
import pandas as pd

# Evaluation set, here the is used the example set
test_data = read('eat_delta_example.extxyz', index = ':')

# Set up of the Nequip calculator for the model to evaluate, here eAT_70 as example
NN=nequip_calculator.NequIPCalculator
NN=NN.from_deployed_model(model_path="path_to_allegro_models/delta_eat_70_deployed.pth")

# Get the real value of the considered property for the given molecule
def get_energy_from_mol(molecule):
    return molecule.info['REF_energy']

# Get the Allegro prediction for the given molecule
def get_pred(molecule):
    molecule.calc = NN
    nequip = molecule.get_total_energy()
    return nequip

pred = []
true = []
maes = []
rel = []
for i in test_data:
    y = get_energy_from_mol(i)
    x = get_pred(i)
    pred.append(x)
    true.append(y)
    
# Create dataframe with real values, predictions, and errors
df = {
    'molecule': range(len(test_data)),
    'true_value': true,
    'prediction': pred
}
df =pd.DataFrame(df)
df.to_json('example_allegro.json') # Save the dataframe
