def classFactory(iface):
    from .src.main import CustomDEMPlugin

    return CustomDEMPlugin(iface)
