import sys
from pathlib import Path


def classFactory(iface):
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.main import CustomDEMPlugin

    return CustomDEMPlugin(iface)
