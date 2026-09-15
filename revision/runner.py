"""Compatibility entrypoint; implementation lives in experiments.runner."""
from experiments.runner import *

if __name__ == "__main__":
    run(parser().parse_args())
