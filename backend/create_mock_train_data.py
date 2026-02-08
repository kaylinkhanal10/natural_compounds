import pandas as pd
import numpy as np

# Mock Data for GNN Verification
data = {
    'smiles': [
        "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "COc1cc(C=O)ccc1O",
        "C1=CC=C(C=C1)C=O", "CC(=O)NO", "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
        "CN", "CCN", "CCCN", "CCCCN", "CCCCCN", "C(=O)O", "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "CCCCC(=O)O"
    ],
    'mw': np.random.rand(20) * 500,
    'logp': np.random.normal(0, 1, 20),
    'tpsa': np.random.rand(20) * 100,
    'hba': np.random.randint(0, 10, 20),
    'hbd': np.random.randint(0, 5, 20),
    'rotb': np.random.randint(0, 10, 20),
    'nring': np.random.randint(0, 5, 20),
    'nrom': np.random.randint(0, 3, 20)
}

df = pd.DataFrame(data)
df.to_excel('raw_data/chembl_mock_train.xlsx', index=False)
print("Created raw_data/chembl_mock_train.xlsx")
