'use strict';
const $=id=>document.getElementById(id);
const names={parent:'Parent',wce:'离线 WCE',baseline:'Baseline',random_group:'Random Group',domain_randomization:'Domain Randomization',fixed_wce:'Fixed WCE',online_wce:'Online WCE'};
const states={complete:'完成',recent:'近期有进展',monitoring:'正在测试（日志证据）',fresh:'未开始',failed:'失败',stale:'长时间无更新',unknown:'状态未知'};
const fmt=n=>n==null?'—':Number(n).toLocaleString('zh-CN');
const when=n=>n?new Date(n*1000).toLocaleString('zh-CN',{hour12:false}):'—';
const duration=n=>{if(n==null)return '—';const s=Math.floor(n);return `${Math.floor(s/3600)}小时 ${Math.floor(s%3600/60)}分 ${s%60}秒`};
const wall=t=>`${t.wall_lower_bound?'至少 ':''}${duration(t.wall_seconds)}`;
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let snapshot={},selected=null,branchPath=null,etag=null;
function render(){
 const d=snapshot,tasks=d.tasks||[];
 $('error').hidden=!d.error;if(d.error)$('error').textContent=`采集失败 ${when(d.error.time)}：${d.error.message}。保留上次有效数据。`;
 $('freshness').textContent=`最近更新 ${when(d.updated_at)} · 下次更新 ${when(d.next_refresh)}${d.refreshing?' · 正在采集…':''}`;
 $('refresh').disabled=!!d.refreshing;
 const stats=[['已完成任务',`${d.completed||0} / ${d.total||56}`,'Parent / WCE / 续训'],['剩余阶段 steps',fmt(d.remaining),`已完成 ${d.goal?((d.steps/d.goal)*100).toFixed(1):0}%`],['磁盘可用',d.disk_free?`${(d.disk_free/2**30).toFixed(1)} GiB`:'—','训练所在磁盘'],['内存可用',d.memory_available?`${(d.memory_available/2**30).toFixed(1)} GiB`:'—',`总内存 ${d.memory_total?(d.memory_total/2**30).toFixed(1):'—'} GiB`]];
 $('stats').innerHTML=stats.map(([a,b,c])=>`<div class="stat"><span>${a}</span><strong>${b}</strong><small>${c}</small></div>`).join('');
 $('wall-total').textContent=`累计任务 wall-clock：${wall(d)} · ${(Number(d.wall_seconds||0)/3600).toFixed(2)}任务小时`;
 $('groups').innerHTML=Object.entries(names).map(([k,n])=>{const ts=tasks.filter(t=>t.group===k),goal=ts.reduce((a,t)=>a+t.goal,0),done=ts.reduce((a,t)=>a+Math.min(t.step,t.goal),0);return `<div class="group"><strong>${n}</strong><span>${ts.filter(t=>t.status==='complete').length}/${ts.length} 完成 · ${goal?(100*done/goal).toFixed(1):0}%</span><progress max="${goal||1}" value="${done}" aria-label="${n}进度"></progress></div>`}).join('');
 Object.keys(names).forEach((k,i)=>{const ts=tasks.filter(t=>t.group===k),p=document.createElement('p');p.textContent=wall({wall_seconds:ts.reduce((s,t)=>s+(t.wall_seconds||0),0),wall_lower_bound:ts.some(t=>t.wall_lower_bound)});$('groups').children[i].appendChild(p)});
 const shown=tasks.filter(t=>['group','network','controller'].every(k=>!$(k).value||t[k]===$(k).value));
 $('count').textContent=`显示 ${shown.length} / ${tasks.length} 项`;
 $('rows').innerHTML=shown.map(t=>`<tr><td><strong>${esc(names[t.group])}</strong><small>${esc(t.network)} · ${esc(t.controller.toUpperCase())}</small></td><td><span class="badge ${esc(t.status)}">${esc(states[t.status])}</span><small>进程：${esc(t.process==='unknown'?'未知':t.process)}</small>${t.branches.length>1?`<small>${t.branches.length}个独立分支，未相加</small>`:''}</td><td><strong>${t.percent}%</strong><small>${fmt(t.step)} / ${fmt(t.goal)}</small><progress max="100" value="${t.percent}" aria-label="阶段进度"></progress></td><td>${fmt(t.episode)} / ${fmt(t.completed_episodes)}${t.partial_episodes.length?`<small>另有部分回合 ${t.partial_episodes.join(', ')}</small>`:''}</td><td>${fmt(t.learning_steps)}<small>WCE updates: ${fmt(t.wce_updates)}</small></td><td>${fmt(t.checkpoint_step)}</td><td>${when(t.last_log)}<small>测试 ${when(t.last_monitor)}</small></td><td><button data-task="${esc(t.id)}">曲线</button></td></tr>`).join('');
 $('rows').querySelectorAll('button').forEach(b=>b.onclick=()=>{selected=b.dataset.task;branchPath=null;detail();$('detail').scrollIntoView({behavior:'smooth',block:'start'});});
 [...$('rows').children].forEach((row,i)=>{const td=document.createElement('td');td.textContent=wall(shown[i]);row.insertBefore(td,row.children[6])});
 $('checkpoint-note').textContent=d.checkpoint_note||'';
 if(selected)detail();
}
function chart(id,rows,key,sdKey){
 const axis=$('axis').value,points=rows.map(r=>({x:r[axis],y:r[key],sd:sdKey?r[sdKey]||0:0}));
 const valid=points.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
 if(!valid.length){$(id).innerHTML='<div class="chart-empty">尚无完整记录 / 指标缺失</div>';return;}
 const W=620,H=260,L=78,R=20,T=18,B=46;
 let xmin=Math.min(...valid.map(p=>p.x)),xmax=Math.max(...valid.map(p=>p.x)),ymin=Math.min(...valid.map(p=>p.y-p.sd)),ymax=Math.max(...valid.map(p=>p.y+p.sd));
 if(xmin===xmax){xmin=Math.max(0,xmin-1);xmax+=1}if(ymin===ymax){ymin-=Math.max(1,Math.abs(ymin)*.05);ymax+=Math.max(1,Math.abs(ymax)*.05)}
 const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
 const x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),y=v=>T+(ymax-v)/(ymax-ymin)*(H-T-B);
 const num=v=>Math.abs(v)>=10000?v.toExponential(1):Number(v.toPrecision(4)).toLocaleString();
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(key)} 随 ${axis} 变化的曲线">`;
 for(let i=0;i<5;i++){const v=ymin+(ymax-ymin)*i/4;svg+=`<line class="chart-grid" x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}"/><text class="chart-label" text-anchor="end" x="${L-8}" y="${y(v)+4}">${num(v)}</text>`;}
 for(let i=0;i<4;i++){const v=xmin+(xmax-xmin)*i/3;svg+=`<text class="chart-label" text-anchor="middle" x="${x(v)}" y="${H-22}">${num(v)}</text>`;}
 // Missing points break the line rather than implying measurements exist.
 let segments=[],segment=[];for(const p of points){if(Number.isFinite(p.y)&&Number.isFinite(p.x)){segment.push(p)}else if(segment.length){segments.push(segment);segment=[]}}if(segment.length)segments.push(segment);
 for(const s of segments){if(sdKey)svg+=`<polygon class="chart-band" points="${s.map(p=>`${x(p.x)},${y(p.y+p.sd)}`).concat([...s].reverse().map(p=>`${x(p.x)},${y(p.y-p.sd)}`)).join(' ')}"/>`;svg+=`<polyline class="chart-line" points="${s.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')}"/>`;}
 for(const p of valid)svg+=`<circle class="chart-point" cx="${x(p.x)}" cy="${y(p.y)}" r="3"><title>${axis}: ${p.x}; ${key}: ${p.y}${sdKey?`; SD: ${p.sd}`:''}</title></circle>`;
 svg+=`<text class="chart-label" text-anchor="middle" x="${(W+L)/2}" y="${H-2}">${axis==='step'?'阶段 steps':'Episode'}</text></svg>`;$(id).innerHTML=svg;
}
function detail(){
 const t=(snapshot.tasks||[]).find(t=>t.id===selected);if(!t)return;
 $('detail').hidden=false;$('detail-title').textContent=`${names[t.group]} / ${t.network} / ${t.controller.toUpperCase()}`;
 $('branch').innerHTML=t.branches.map(b=>`<option value="${esc(b.path)}">${esc(b.path.split('/').pop())} · ${fmt(b.step)} steps</option>`).join('');
 if(branchPath&&t.branches.some(b=>b.path===branchPath))$('branch').value=branchPath;
 const b=t.branches.find(b=>b.path===$('branch').value)||t;branchPath=b.path;
 $('detail-info').textContent=`${states[b.status]} · ${fmt(b.step)} / ${fmt(b.goal)} steps · ${b.monitor.length}轮完整测试`;
 $('detail-info').textContent+=` · 当前分支累计 ${wall(b)} · 该任务全部分支累计 ${wall(t)}`;
 $('wall-attempts').textContent=(b.wall_attempts||[]).map(r=>`${r.path.split('/').pop()}：${r.lower_bound?'至少 ':''}${duration(r.seconds)} (${r.source})`).join('\n');
 $('runpath').textContent=b.path||'尚未创建训练目录';$('warnings').textContent=[...new Set([...(t.warnings||[]),...(b.warnings||[])])].join('\n');
 $('monitor-note').textContent=t.stage==='wce'?'离线 WCE 未配置定期 monitor。下方训练 reward 来自冻结 controller，不等于 WCE 自身优化奖励。':'训练前、每50个完整 episode及结束时执行3次600秒 Uniform测试。曲线取各智能体、动作和rollout的均值；阴影为rollout间样本SD。缺失指标不绘制。';
 const reward=$('reward').value;
 chart('monitor-chart',b.monitor,reward,reward+'_sd');chart('queue-chart',b.monitor,'queue');chart('waiting-chart',b.monitor,'waiting');
 chart('training-chart',b.training.map(r=>({step:r.stage_simulation_steps,episode:r.episode,value:r['mean_'+(reward==='raw'?'raw':'learner')+'_reward']})),'value');
}
async function load(){try{const response=await fetch('/api/snapshot',{headers:etag?{'If-None-Match':etag}:{}});if(response.status===304)return;if(!response.ok)throw Error(`HTTP ${response.status}`);etag=response.headers.get('ETag');snapshot=await response.json();render();}catch(e){$('error').hidden=false;$('error').textContent='无法连接 dashboard 服务，保留已显示数据：'+e.message;}}
['group','network','controller'].forEach(k=>$(k).onchange=render);['axis','reward'].forEach(k=>$(k).onchange=detail);
$('branch').onchange=()=>{branchPath=$('branch').value;detail()};$('close-detail').onclick=()=>{selected=null;$('detail').hidden=true};
$('refresh').onclick=async()=>{try{$('refresh').disabled=true;const r=await fetch('/api/refresh',{method:'POST',headers:{'X-Dashboard-Refresh':'1'}});if(!r.ok)throw Error(`HTTP ${r.status}`);setTimeout(load,1200)}catch(e){$('error').hidden=false;$('error').textContent=e.message;$('refresh').disabled=false}};
load();setInterval(load,10000);
