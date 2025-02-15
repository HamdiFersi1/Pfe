import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-build-isolation", package])

# List of required packages with versions (transformers downgraded)
packages = [
    "torch==2.6.0",
    "transformers==4.30.0",  # Downgraded to avoid Rust
    "spacy==3.6.0",
    "fastapi==0.100.0",
    "uvicorn==0.22.0",
    "sentence-transformers==2.2.3",
    "pytesseract==0.3.12",
    "pdfminer.six==20240410"
]

# Install each package
for package in packages:
    print(f"Installing {package}...")
    install(package)

# Download spaCy's English model
print("Downloading spaCy English model (en_core_web_sm)...")
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

print("All dependencies have been installed.")
