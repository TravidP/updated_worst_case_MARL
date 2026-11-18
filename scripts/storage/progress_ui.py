"""Read-only archive dashboard, bound to localhost."""
import argparse
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTML = '''<!doctype html><meta charset="utf-8"><title>CBWCE 归档进度</title>
<style>body{background:#101827;color:#e7edf7;font:17px system-ui;max-width:1000px;margin:50px auto;padding:20px}h1{font-size:30px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{background:#1d293d;padding:24px;border-radius:14px}.value{font-size:32px;color:#72dfc4;margin-top:12px}small{color:#aab9ce}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:12px;border-bottom:1px solid #344158}#error{color:#ffb39d}</style>
<h1>CBWCE · Google Drive 归档</h1><p id="phase">读取中…</p><p id="error"></p><div class="grid" id="cards"></div>
<p><small id="updated"></small></p><h2>最近批次</h2><table><thead><tr><th>批次</th><th>状态</th><th>原始大小</th><th>文件数</th></tr></thead><tbody id="rows"></tbody></table>
<p><small>每 5 秒刷新。上传进程未开启实时字节统计，因此本页展示批次和阶段，不虚构单文件百分比。只有完整下载校验后的清理才计入释放空间。</small></p>
<script>
const gib=n=>(n/1073741824).toFixed(2)+' GiB';
const names={packing:'打包中',uploading:'上传中',publishing_manifest:'保存归档清单',downloading_verification:'下载核验中',verifying:'校验内容中',cleaning_verified_sources:'清理已核验文件',batch_complete:'批次完成',local_removed:'已归档并清理',local_removed_untracked_only:'已归档，保留受 Git 跟踪文件',inventory_untracked_complete:'清单内未跟踪输出处理完成',download_verified:'下载校验通过',packed:'已打包',stopped_at_batch_boundary:'已在批次边界停止'};
async function refresh(){try{let r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw Error('读取失败');let d=await r.json();document.querySelector('#error').textContent='';document.querySelector('#phase').textContent='第 '+d.worker.batch+' 批 · '+(names[d.worker.phase]||d.worker.phase);let cards=[['已完成批次',d.completed],['已清理文件',d.files.toLocaleString()],['释放磁盘空间',gib(d.freed)],['磁盘可用',gib(d.available)],['已清理原始数据',gib(d.bytes)],['当前归档包',gib(d.current_bytes)]];document.querySelector('#cards').innerHTML=cards.map(x=>'<div class="card">'+x[0]+'<div class="value">'+x[1]+'</div></div>').join('');document.querySelector('#updated').textContent='页面刷新：'+new Date().toLocaleString()+'；任务阶段更新时间：'+new Date(d.worker.updated_at).toLocaleString();document.querySelector('#rows').innerHTML=d.recent.map(x=>'<tr><td>'+x.number+'</td><td>'+(names[x.state]||x.state)+'</td><td>'+gib(x.bytes)+'</td><td>'+x.files+'</td></tr>').join('')}catch(e){document.querySelector('#error').textContent='无法读取最新状态，请检查服务。'}}refresh();setInterval(refresh,5000);
</script>'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8765);a=p.parse_args();cat=a.root.resolve()/'cbwce_archive/catalogs'
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   if self.path=='/': payload=HTML.encode();typ='text/html; charset=utf-8'
   elif self.path=='/status':
    worker=json.loads((cat/'rclone_worker.json').read_text());batches=[]
    for f in sorted(cat.glob('batch_*.json')):
     if f.stem.removeprefix('batch_').isdigit():
      m=json.loads(f.read_text());m['number']=int(f.stem[6:]);batches.append(m)
    done=[m for m in batches if m['state'].startswith('local_removed')];disk=os.statvfs(a.root)
    payload=json.dumps(dict(worker=worker,completed=len(done),files=sum(m.get('removed_files',0) for m in done),freed=sum(m.get('removed_allocated_bytes',0) for m in done),bytes=sum(m.get('removed_bytes',0) for m in done),available=disk.f_bavail*disk.f_frsize,current_bytes=next((m['archive_bytes'] for m in batches if m['number']==worker['batch']),0),recent=[dict(number=m['number'],state=m['state'],bytes=m['source_bytes'],files=m['files']) for m in batches[-8:][::-1]])).encode();typ='application/json'
   else:self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
  def log_message(self,*args):pass
 print('Dashboard: http://127.0.0.1:%d'%a.port,flush=True)
 ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
