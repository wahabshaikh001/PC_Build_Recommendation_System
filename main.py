import sys
import os

# Ensure the root directory is the working directory and in path
project_root = os.path.abspath(os.path.dirname(__file__))
os.chdir(project_root)
sys.path.append(project_root)

from src.main import main

if __name__ == "__main__":
    main()
