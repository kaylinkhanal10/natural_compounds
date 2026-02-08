try:
    from rdkit.Chem import Inchi
    print("Inchi found")
except ImportError:
    print("Inchi NOT found")

import rdkit.Chem
try:
    print(rdkit.Chem.MolToInchiKey)
    print("MolToInchiKey in Chem")
except AttributeError:
    print("MolToInchiKey NOT in Chem")
