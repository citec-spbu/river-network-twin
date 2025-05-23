import sys
import os

def classFactory(iface):
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.main import CustomDEMPlugin

    return CustomDEMPlugin(iface)
