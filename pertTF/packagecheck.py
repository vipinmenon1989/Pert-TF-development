import importlib

packages_to_check = [
    "torch", "anndata", "scanpy", "numpy", "matplotlib",
    "seaborn", "networkx", "pandas", "tqdm", "gseapy",
    "torchtext", "scgpt", "sklearn", "csv", "wandb"
]

print("=== Package Import Check ===")
results = {}
for pkg in packages_to_check:
    try:
        importlib.import_module(pkg)
        results[pkg] = "✅ Installed"
    except ImportError as e:
        results[pkg] = f"❌ Missing ({e})"

# Print a summary table
for pkg, status in results.items():
    print(f"{pkg:<12} : {status}")

# Special check for torchtext internals
try:
    from torchtext.vocab import Vocab
    print("torchtext.vocab.Vocab ✅ available")
except Exception as e:
    print(f"torchtext.vocab.Vocab ❌ not available ({e})")

try:
    from torchtext._torchtext import Vocab as VocabPybind
    print("torchtext._torchtext.Vocab ✅ available")
except Exception as e:
    print(f"torchtext._torchtext.Vocab ❌ not available ({e})")
