'use strict';
const $ = id => document.getElementById(id);
const el = (tag,text,cls) => {const node=document.createElement(tag); if(text!=null) node.textContent=text; if(cls) node.className=cls; return node;};
const labels={inventory:'Model inventory',tuning:'Development tuning',freeze:'Freeze winners',verify:'Verify winners',parent:'Baseline',wce:'Offline WCE',baseline:'Baseline continuation',random_group:'Random group',domain_randomization:'Domain randomization',fixed_wce:'Fixed WCE',online_wce:'Online WCE',evaluate:'Evaluation',formal:'Training & evaluation',report:'Report'};
const stages=['tuning','parent','wce','baseline','random_group','domain_randomization','fixed_wce','online_wce','evaluate'];
for(const value of [...stages,'inventory','freeze','verify','report']){
 const option=el('option',labels[value]);option.value=value;$('stage').append(option);
}
const selected=new Set();let data,detailId='',resultsLoaded=false,busy=false;
const fmt=x=>x==null?'—':Number(x).toLocaleString('en',{maximumFractionDigits:2});
const gib=x=>fmt(x/1073741824);
const statusText=s=>s.replaceAll('_',' ');
async function json(path,options){const r=await fetch(path,options);const v=await r.json();if(!r.ok)throw Error(v.error||r.statusText);return v;}
async function action(name,ids=[],stage=null){
 try{if(!data)return;await json('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Workflow-Token':data.token},body:JSON.stringify({action:name,ids,stage})});$('message').textContent='Request saved.';await refresh();}
 catch(e){$('message').textContent=e.message;}
}
$('auto').onclick=()=>action('auto');$('pause').onclick=()=>action('pause');$('resume').onclick=()=>action('resume');$('release').onclick=()=>action('stage',[],$('stage').value);
for(const [id,cmd] of Object.entries({runTasks:'run',holdTasks:'pause',resumeTasks:'resume',retryTasks:'retry'}))$(id).onclick=()=>action(cmd,[...selected]);
function drawTable(){
 if(!data)return;
 const term=$('filter').value.trim().toLowerCase(), frag=document.createDocumentFragment();
 const rows=data.tasks.filter(t=>[t.number,t.id,t.status,statusText(t.status),labels[t.stage]].join(' ').toLowerCase().includes(term));
 for(const t of rows){
  const tr=el('tr');const add=(v)=>{const td=el('td');if(typeof v==='string')td.textContent=v;else td.append(v);tr.append(td);return td;};
  const checkbox=el('input');checkbox.type='checkbox';checkbox.checked=selected.has(t.id);checkbox.setAttribute('aria-label','Select '+t.id);
  checkbox.onchange=()=>{if(checkbox.checked)selected.add(t.id);else selected.delete(t.id);updateSelection();};add(checkbox);
  const b=el('button',(t.number?t.number+' · ':'')+(labels[t.stage]||t.stage),'link');b.onclick=()=>{detailId=t.id;loadDetail(true);};
  const title=add(b);if(!t.number)title.append(el('small',[t.selection,t.id.split('/').pop()].filter(Boolean).join(' · ')));
  add(t.network+' / '+t.algorithm.toUpperCase());const st=add(el('span',statusText(t.status),'status '+t.status));if(t.reason)st.append(el('small',t.reason));
  add(fmt(t.quantity)+' / '+fmt(t.limit));const score=add(fmt(t.best));if(t.best_episode!=null)score.append(el('small','Episode '+t.best_episode));
  add(t.cpu||t.rss?fmt(t.cpu)+' cores / '+gib(t.rss)+' GiB':'—');add(t.elapsed?fmt(t.elapsed/60)+' min':'—');frag.append(tr);
 }
 $('tasks').replaceChildren(frag);$('count').textContent=rows.length+' tasks shown';updateSelection();
}
function updateSelection(){for(const id of ['runTasks','holdTasks','resumeTasks','retryTasks'])$(id).disabled=selected.size===0;}
$('filter').oninput=drawTable;
function render(){
 const {pool:p,tasks,control:c}=data, count=s=>tasks.filter(t=>s.includes(t.status)).length;
 $('clock').textContent=new Date(data.now*1000).toLocaleTimeString();$('mode').textContent=(c.paused?'Paused':c.auto?'Automatic dispatch':'Manual dispatch')+' · '+data.workflow;
 const metrics=[['CPU capacity',fmt(p.effective)+' cores',p.logical+' hardware threads'],['Worker CPU budget',fmt(p.cpu_budget)+' cores','Current queue limit '+p.max_parallel+' jobs'],['Available RAM',gib(p.available_ram)+' GiB','Pool budget '+gib(p.ram_budget)+' GiB'],['Running',count(['running','pausing']),'Independent processes'],['Ready / resource wait',count(['ready'])+' / '+count(['waiting_resources']),'Released jobs'],['Completed',count(['complete']),tasks.length+' tasks prepared']];
 $('metrics').replaceChildren(...metrics.map(([k,v,n])=>{const x=el('div',null,'tile');x.append(el('small',k),el('strong',v),el('small',n));return x;}));
 const planned={tuning:24,parent:6,wce:6,baseline:6,random_group:6,domain_randomization:6,fixed_wce:6,online_wce:6,evaluate:1380};
 $('phases').replaceChildren(...stages.map(name=>{
  const jobs=tasks.filter(t=>t.stage===name), total=jobs.length||planned[name];
  const running=jobs.filter(t=>['running','pausing'].includes(t.status)).length;
  const done=jobs.filter(t=>t.status==='complete').length;
  const issues=jobs.filter(t=>['failed','review_required','needs_attention','blocked'].includes(t.status)).length;
  const tile=el('div',null,'tile phase');
  tile.append(el('strong',labels[name]),el('small',running+' running · '+done+' / '+total+' complete'));
  if(issues)tile.append(el('small',issues+' need attention','needs_attention'));
  return tile;
 }));
 const checks=data.phases.filter(p=>['inventory','freeze','verify','report'].includes(p.name)||p.status==='needs_attention'||p.cutoff);
 $('checks').replaceChildren(...checks.map(p=>{
  const tile=el('div',null,'tile phase');tile.append(el('strong',labels[p.name]),el('small',statusText(p.status),'status '+p.status));
  if(p.reason)tile.append(el('small',p.reason));
  if(p.status==='needs_attention'){const b=el('button','Retry phase');b.onclick=()=>action('retry-phase',[],p.name);tile.append(b);}
  return tile;
 }));
 if(checks.some(p=>p.status==='needs_attention'))$('workflowChecks').open=true;
 drawTable();
 if(!data.reports.length){$('reports').textContent='Paired reports appear after the complete evaluation matrix. Select a task to inspect its current results.';return;}
 $('reports').replaceChildren(...data.reports.map(file=>{const a=el('a',file);a.href='/reports/'+file;a.download=file;return a;}));
 if(!resultsLoaded){resultsLoaded=true;json('/api/results').then(rows=>{const table=el('table');if(rows.length){const cols=['network','controller','method','selection','scope','mean_queue'].filter(k=>k in rows[0]);const head=el('tr');cols.forEach(k=>head.append(el('th',k)));table.append(head);rows.forEach(row=>{const tr=el('tr');cols.forEach(k=>tr.append(el('td',row[k])));table.append(tr);});}$('resultTable').replaceChildren(table);}).catch(e=>{resultsLoaded=false;$('message').textContent=e.message;});}
}
const trainColor='#1e7168', validationColor='#ad5022';
function chart(title,units,series,xTitle,best){
 const panel=el('section',null,'chart');panel.append(el('h3',title));
 const legend=el('div',null,'legend');for(const line of series)legend.append(el('span',line.label,line.validation?'validKey':'trainKey'));
 panel.append(legend);
 const points=series.flatMap(line=>line.points);
 if(!points.length){panel.append(el('p','Waiting for the first recorded point.','muted'));return panel;}
 const c=el('canvas');c.width=760;c.height=185;c.setAttribute('role','img');c.setAttribute('aria-label',title+' · '+units);
 const g=c.getContext('2d'), xs=points.map(p=>p.x),ys=points.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs),lo=Math.min(...ys),hi=Math.max(...ys);
 const range=hi===lo?1:hi-lo, ymin=lo-range*.08,ymax=hi+range*.08;
 const x=v=>65+665*(v-xmin)/Math.max(1,xmax-xmin),y=v=>136-111*(v-ymin)/(ymax-ymin);
 g.font='14px sans-serif';g.fillStyle='#496267';g.fillText(units,5,14);
 for(const v of new Set([lo,hi])){g.fillText(fmt(v),4,y(v)+5);g.strokeStyle='#dde7e5';g.beginPath();g.moveTo(64,y(v));g.lineTo(735,y(v));g.stroke();}
 for(const line of series){
  g.strokeStyle=g.fillStyle=line.validation?validationColor:trainColor;g.lineWidth=2;g.beginPath();
  for(const [i,p] of line.points.entries()){
   if(!line.validation){if(i)g.lineTo(x(p.x),y(p.y));else g.moveTo(x(p.x),y(p.y));}
   if(line.validation||line.points.length===1){g.fillRect(x(p.x)-3,y(p.y)-3,6,6);}
  }
  if(!line.validation)g.stroke();
 }
 if(best){g.strokeStyle='#8b400f';g.beginPath();g.arc(x(best.x),y(best.y),7,0,Math.PI*2);g.stroke();}
 g.fillStyle='#496267';g.fillText(fmt(xmin),60,159);if(xmax!==xmin)g.fillText(fmt(xmax),704,159);
 g.fillText(xTitle,317,178);panel.append(c);
 if(best)panel.append(el('small','Selected best: '+fmt(best.y)+' vehicles at episode '+best.x));
 return panel;
}
function drawCurves(curves,task){
 $('charts').replaceChildren();$('curveControls').hidden=task.stage==='evaluate';if(task.stage==='evaluate')return;
 const {training,validation,live}=curves,scale=$('rewardScale').value;
 const series=(rows,fn,label,valid=false)=>({label,validation:valid,points:rows.filter(r=>Number.isFinite(fn(r))).map(r=>({x:r.episode,y:fn(r)}))});
 const queueTrain=series(training,r=>r.queue,'Training'),queueVal=series(validation,r=>r.score,'Fixed validation',true);
 const wce=task.stage==='wce', best=queueVal.points.reduce((prev,p)=>!prev||(wce?p.y>prev.y:p.y<prev.y)?p:prev,null);
 $('charts').append(chart('Mean network queue · '+(wce?'maximize':'minimize')+' on validation','Vehicles',[queueTrain,queueVal],'Episode',best));
 const roles=wce?['wce']:task.stage==='online_wce'?['controller','wce']:['controller'];
 for(const role of roles){
  const read=r=>r.rewards && r.rewards[role] && r.rewards[role][scale], tr=series(training,read,'Training'),lines=[tr];
  if(role==='controller'||wce)lines.push(series(validation,read,'Fixed validation',true));
  const title=(role==='controller'?'Controller':'WCE')+' mean immediate reward';
  $('charts').append(chart(title,scale==='learner'?'Learner units':(role==='controller'?'Raw reward / junction / step':'Mean vehicles / demand block'),lines,'Episode'));
 }
 const blocks=live.block_queues||[];
 $('charts').append(chart('Block queue'+(live.live_episode==null?'':' · episode '+live.live_episode),'Vehicles',[{label:'Block mean',points:blocks.map(b=>({x:b.end_seconds,y:b.mean_queue}))}],'Simulated seconds'));
}
$('rewardScale').onchange=()=>loadDetail();
async function loadDetail(scroll=false){
 const id=detailId;if(!id)return;
 try{const info=await json('/api/task?id='+encodeURIComponent(id));if(id!==detailId)return;
 const task=data.tasks.find(t=>t.id===id);$('detail').hidden=false;$('taskTitle').textContent=id;
 $('identity').textContent='Dependencies: '+(task.deps.join(', ')||'none')+' · Last validation: '+fmt(task.latest)+' · Output: '+info.output;
 drawCurves(info.curves,task);
 const scores=info.rollouts.map(r=>'Realization '+r.realization+' · mean queue '+fmt(r.mean_queue)).join('\n');
 $('log').textContent=[scores,JSON.stringify(info.result,null,2),info.log].filter(Boolean).join('\n');if(scroll)$('detail').scrollIntoView({behavior:'smooth'});
 }catch(e){$('message').textContent=e.message;}
}
$('close').onclick=()=>{detailId='';$('detail').hidden=true;};
async function refresh(){if(busy)return;busy=true;try{data=await json('/api/state');render();await loadDetail();}catch(e){$('message').textContent='Connection: '+e.message;}finally{busy=false;}}
refresh();setInterval(refresh,5000);
