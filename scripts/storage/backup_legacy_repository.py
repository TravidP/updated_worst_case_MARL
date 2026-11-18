"""Back up the complete remaining legacy tree, verify a restored copy, then clean exact sources."""
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
ROOT=Path('/home/sdc_joran/Journal/deeprl_signal_control')
CAT=ROOT/'cbwce_archive/catalogs'
STAGE=ROOT/'cbwce_archive/legacy_backup'
KEEP={'cbwce_archive','cbwce_standalone','.agents','.aws','.codex','.vscode'}
REMOTE='cbwce_archive:01_legacy_source/legacy_complete_20261118'
RC=['/home/sdc_joran/.local/bin/cbwce-rclone','--config','/home/sdc_joran/.config/rclone/cbwce-archive-20261116.conf','--cache-dir','/tmp/cbwce-rclone-cache','--stats','60s','--stats-log-level','NOTICE','--timeout','5m']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');os.replace(t,p)
def state(phase):
 write(CAT/'legacy_backup_status.json',dict(phase=phase,time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()));print(phase,flush=True)
def rc(*args):subprocess.run(RC+list(map(str,args)),check=True)
def main():
 import fcntl
 lock=(CAT/'legacy_backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 STAGE.mkdir(exist_ok=True)
 manifest=STAGE/'manifest.json';archive=STAGE/'legacy.tar.gz'
 if manifest.exists() or archive.exists():raise RuntimeError('Existing backup: inspect status before resuming')
 state('inventory_and_hashing')
 files=[];dirs=[]
 for top in sorted(ROOT.iterdir()):
  if top.name in KEEP:continue
  paths=[top]+(sorted(top.rglob('*')) if top.is_dir() else [])
  for p in paths:
   if p.is_symlink():raise RuntimeError('Symlink requires explicit preservation review: '+str(p))
   if p.is_dir():dirs.append(str(p.relative_to(ROOT)));continue
   if not p.is_file():raise RuntimeError('Unsupported file: '+str(p))
   before=p.stat();digest=sha(p);after=p.stat()
   if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during hashing')
   files.append(dict(path=str(p.relative_to(ROOT)),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=digest))
 refs=subprocess.check_output(['git','-C',str(ROOT),'show-ref'],text=True)
 write(STAGE/'source_manifest.json',dict(files=files,dirs=dirs,refs=refs,excluded_preserved=sorted(KEEP)))
 state('packing')
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for entry in files:tar.add(ROOT/entry['path'],arcname=entry['path'],recursive=False)
 write(manifest,dict(files=files,dirs=dirs,refs=refs,excluded_preserved=sorted(KEEP),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size))
 state('uploading')
 rc('copyto',archive,REMOTE+'/legacy.tar.gz','--immutable','--checksum')
 rc('copyto',manifest,REMOTE+'/manifest.json','--immutable','--checksum')
 state('downloading_verification')
 restored=STAGE/'downloaded.tar.gz';checked=STAGE/'downloaded_manifest.json'
 rc('copyto',REMOTE+'/legacy.tar.gz',restored,'--ignore-times')
 rc('copyto',REMOTE+'/manifest.json',checked,'--ignore-times')
 m=json.loads(manifest.read_text())
 if sha(checked)!=sha(manifest) or sha(restored)!=m['archive_sha256']:raise RuntimeError('Cloud roundtrip mismatch')
 state('restoring_and_verifying')
 tree=STAGE/'restored';tree.mkdir()
 expected={x['path']:x for x in files};seen=set()
 with tarfile.open(restored,'r:gz') as tar:
  for member in tar:
   name=member.name
   if name not in expected or name in seen or Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Unexpected member')
   if not member.isfile() and not member.islnk():raise RuntimeError('Unsupported member')
   if member.islnk() and member.linkname not in seen:raise RuntimeError('Unsafe hardlink')
   target=tree/name;target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(member) as src,target.open('xb') as out:
    for b in iter(lambda:src.read(1024*1024),b''):out.write(b)
   os.chmod(target,member.mode&0o777)
   if sha(target)!=expected[name]['sha256']:raise RuntimeError('Restored member mismatch')
   seen.add(name)
 if seen!=set(expected):raise RuntimeError('Incomplete restore')
 restored_refs=subprocess.check_output(['git','-C',str(tree),'show-ref'],text=True)
 if refs!=restored_refs:raise RuntimeError('Git refs differ')
 with (STAGE/'git_fsck.txt').open('w') as log:
  subprocess.run(['git','-C',str(tree),'fsck','--full'],stdout=log,stderr=subprocess.STDOUT,check=True)
 for obj in (tree/'.git/lfs/objects').rglob('*'):
  if obj.is_file() and len(obj.name)==64 and sha(obj)!=obj.name:raise RuntimeError('LFS hash mismatch')
 state('verified_checking_sources_before_cleanup')
 for item in files:
  p=ROOT/item['path'];s=p.stat()
  if s.st_size!=item['bytes'] or s.st_mtime_ns!=item['mtime_ns'] or sha(p)!=item['sha256']:raise RuntimeError('Source changed; cleanup stopped: '+item['path'])
 # Durable evidence remains both locally and remotely before cleanup.
 report=dict(status='restore_verified',archive_sha256=m['archive_sha256'],files=len(files),refs_verified=True,git_fsck_passed=True,lfs_verified=True,remote=REMOTE)
 write(CAT/'legacy_restore_verification.json',report)
 rc('copyto',CAT/'legacy_restore_verification.json',REMOTE+'/verification.json','--immutable','--checksum')
 state('cleaning_verified_legacy_files')
 with (CAT/'legacy_cleanup.jsonl').open('a') as journal:
  for item in files:
   p=ROOT/item['path']
   # Recheck at removal time; never remove a newly changed source.
   if sha(p)!=item['sha256']:raise RuntimeError('Source changed during cleanup')
   journal.write(json.dumps(dict(path=item['path'],sha256=item['sha256'],intent='remove_verified'))+'\n');journal.flush();os.fsync(journal.fileno());p.unlink()
 for d in sorted(dirs,key=lambda x:len(Path(x).parts),reverse=True):
  try:(ROOT/d).rmdir()
  except OSError:pass
 state('legacy_cleaned_backup_retained')
 # Keep local verified backup until the final handover review; do not recursively delete it here.
if __name__=='__main__':main()
