"""Run browser authorization without exposing OAuth tokens in terminal output."""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rclone', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root-folder-id', required=True)
    p.add_argument('--remote', default='cbwce_archive')
    a = p.parse_args()
    if a.config.exists():
        raise FileExistsError('Refusing to replace an existing credential file')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.remote + a.root_folder_id):
        raise ValueError('Invalid remote name or folder identity')
    cache = Path('/tmp/cbwce-rclone-cache')
    cache.mkdir(mode=0o700, exist_ok=True)
    env = dict(os.environ, XDG_CACHE_HOME=str(cache), RCLONE_CACHE_DIR=str(cache))
    process = subprocess.Popen([a.rclone, '--cache-dir', str(cache), 'authorize', 'drive',
                                '--auth-no-open-browser'], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=env)
    token = None
    for line in process.stdout:
        link = re.search(r'http://127\.0\.0\.1:53682/auth\?state=[A-Za-z0-9_-]+', line)
        if link:
            print('AUTHORIZATION_URL=' + link.group(0), flush=True)
        text = line.strip()
        if text.startswith('{'):
            try:
                candidate = json.loads(text)
                if candidate.get('access_token') and candidate.get('refresh_token'):
                    token = candidate
            except json.JSONDecodeError:
                pass
    code = process.wait()
    if code or not token:
        raise RuntimeError('Authorization did not complete; no credentials were written')
    a.config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    contents = ('[' + a.remote + ']\ntype = drive\nscope = drive\nroot_folder_id = ' +
                a.root_folder_id + '\ntoken = ' + json.dumps(token, separators=(',', ':')) + '\n')
    fd = os.open(a.config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        out.write(contents)
        out.flush()
        os.fsync(out.fileno())
    print('AUTHORIZATION_SAVED=' + str(a.config), flush=True)


if __name__ == '__main__':
    main()
