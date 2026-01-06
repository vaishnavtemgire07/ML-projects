import importlib, traceback, sys

try:
    importlib.import_module('source.components.model_trainer')
    print('module imported')
except Exception:
    traceback.print_exc()
    sys.exit(1)
