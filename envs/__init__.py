"""Environment package bootstrap helpers."""

import os
import shutil
import sys


def _candidate_sumo_tool_paths():
    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        yield os.path.join(sumo_home, 'tools')

    sumo_bin = shutil.which('sumo')
    if sumo_bin:
        prefix = os.path.dirname(os.path.dirname(os.path.realpath(sumo_bin)))
        yield os.path.join(prefix, 'share', 'sumo', 'tools')
        yield os.path.join(prefix, 'tools')

    yield '/usr/share/sumo/tools'
    yield '/usr/local/share/sumo/tools'


def _ensure_sumo_tools_path():
    for tool_path in _candidate_sumo_tool_paths():
        if os.path.isdir(tool_path) and tool_path not in sys.path:
            sys.path.append(tool_path)
            return


_ensure_sumo_tools_path()
