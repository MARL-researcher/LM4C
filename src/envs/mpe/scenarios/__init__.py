import os.path as osp


import importlib
def load_source(modname, modpath):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    if not spec:
        raise ValueError("Error loading '%s' module" % modpath)
    module = importlib.util.module_from_spec(spec)
    if not spec.loader:
        raise ValueError("Error loading '%s' module" % modpath)
    spec.loader.exec_module(module)
    return module

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return load_source('', pathname)