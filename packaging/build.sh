#!/bin/bash
set -euo pipefail
# Run on a development machine with Python 3.10, pip-tools and Apptainer.
# Require a digest-qualified Python 3.10 image, not an unpinned tag.
: "${CBWCE_BASE_IMAGE:?Set a Python 3.10 image reference containing @sha256:}"
[[ "$CBWCE_BASE_IMAGE" == *@sha256:* ]] || { echo 'Digest-pinned base required' >&2; exit 2; }
python -c 'import sys; assert sys.version_info[:2] == (3,10)'
# The committed lock is the release input. Re-resolving requires an explicit update.
if [[ "${CBWCE_UPDATE_LOCK:-0}" == 1 ]]; then
    pip-compile --allow-unsafe --generate-hashes --resolver=backtracking --index-url=https://pypi.org/simple --no-emit-index-url --no-emit-trusted-host --output-file=packaging/requirements.lock packaging/requirements.in
fi
apptainer build --build-arg "BASE_IMAGE=$CBWCE_BASE_IMAGE" cbwce-v8.1.sif packaging/Apptainer.def
apptainer exec --bind "$PWD" cbwce-v8.1.sif python -c 'import json; from experiments.runtime import identity; print(json.dumps(identity()))' > packaging/runtime.actual.json
python - <<'PY'
import hashlib,json,os
from pathlib import Path
h=hashlib.sha256()
with open('cbwce-v8.1.sif','rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
lock={'base_image':os.environ['CBWCE_BASE_IMAGE'],'image_sha256':h.hexdigest(),'runtime':json.loads(Path('packaging/runtime.actual.json').read_text())}
Path('packaging/runtime.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
PY
echo 'Next: run scripts/verify.py inside this image; publication still requires a frozen validation manifest.'
