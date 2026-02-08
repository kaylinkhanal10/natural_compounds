try:
    import rdkit
    print("rdkit ok")
except ImportError:
    print("rdkit missing")

try:
    import torch_geometric
    print("pyg ok")
except ImportError:
    print("pyg missing")
