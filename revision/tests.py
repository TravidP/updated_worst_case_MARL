"""Compatibility entrypoint; implementation lives in tests.test_corrections."""
from tests.test_corrections import *

if __name__ == "__main__":
    import runpy
    runpy.run_module('tests.test_corrections', run_name='__main__')
