from ase import Atoms
from ase.io import write, read
import pandas as pd
from mace.calculators import MACECalculator

# Evaluation set, here the used the example set
test_data = read('dip_delta_example.extxyz', index = ':') 

calculator = MACECalculator(model_paths=f'../../enn/dip_delta_70.model', device ='cpu', default_dtype="float32",model_type='DipoleMACE')

def get_energy_from_mol(molecule):
    return molecule.info['REF_dipole']

true = []
pred = []
for i in test_data:
    true.append(get_energy_from_mol(i))
    i.set_calculator(calculator)
    x = i.get_dipole_moment()
    pred.append(x)


# Create dataframe with real values and predictions
df = {
    'molecule': range(len(test_data)),
    'original_value': true,
    'prediction': pred
}
df =pd.DataFrame(df)
df.to_json('example_mace_dip.json')
