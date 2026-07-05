import sys
import os

# Ensure the root directory is in the import path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.main import main

if __name__ == "__main__":
    main()
