import os

from src import main

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__)).replace("\\", "/")

    main.main(project_root)
