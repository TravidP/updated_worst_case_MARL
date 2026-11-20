"""Paired evaluation statistics, with training-seed and rollout uncertainty separated."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.core import digest
from experiments.protocol import settings, training_seeds
from experiments.storage import atomic_json
from experiments.evaluation import read_evaluation
from experiments.scenarios import definitions


def sample_sd(values):
    return float(np.std(values,ddof=1)) if len(values)>1 else None


def bootstrap_mean_ci(differences,key):
    values=np.asarray(differences,dtype=float)
    rng=np.random.RandomState(int(digest(['paired-bootstrap-v1',key])[:8],16))
    samples=values[rng.randint(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return list(map(float,np.percentile(samples,[2.5,97.5])))


def scope_values(rows,scenarios):
    """Equal scenario weighting within each paired realization, never pool training seeds."""
    return np.array([np.mean([rows[s,i]['mean_queue'] for s in scenarios]) for i in range(10)])


def scopes(network):
    scenes=definitions(network,'seen')+definitions(network,'test')
    result={s['id']:[s['id']] for s in scenes}
    result.update({'family:'+family:[s['id'] for s in scenes if s['family']==family]
                   for family in sorted({s['family'] for s in scenes if s['family']!='seen'})})
    result.update({split:[s['id'] for s in scenes if s['split']==split] for split in ('seen','test')})
    return result


def report(inputs,output,seeds=None,networks=None,controllers=None,selections=None,campaign=None):
    p=settings();networks=networks or p['networks'];controllers=controllers or p['controllers']
    contract=None
    if campaign:
        from experiments.campaign import load_campaign
        contract=load_campaign(campaign)
        if seeds is not None and seeds!=contract['training_seeds']:raise ValueError('Seed list differs from campaign')
        seeds=contract['training_seeds']
    selections=selections or (contract['evaluation_selections'] if contract else ['best'])
    seeds=training_seeds(seeds)
    if not set(networks)<=set(p['networks']) or not set(controllers)<=set(p['controllers']):raise ValueError('Unknown report scope')
    if not set(selections)<={'best','final'}:raise ValueError('Unknown report selection')
    for values in (networks,controllers,selections):
        if len(set(values))!=len(values):raise ValueError('Duplicate report scope')
    cells={};pairing={};identities={};test_contracts={};metadata={}
    for source in inputs:
        m,rows=read_evaluation(source)
        if m['pilot']:raise ValueError('Pilot results cannot enter publication reports')
        key=(m['network'],m['controller'],m['method'],m['selection'],m['seed'])
        if key[0] not in networks or key[1] not in controllers or key[3] not in selections or key[4] not in seeds:
            raise ValueError('Input outside the explicitly requested report scope')
        if m['method'] not in p['methods'] or m['rollouts']!=10:raise ValueError('Invalid method/realization count')
        if contract and (m['test_hash']!=contract['tests'][key[0]]['hash'] or
                         m['validation_hash']!=contract['validation'][key[0]]['hash'] or m['source']!=contract['source']):
            raise ValueError('Input does not match campaign')
        identity=(m['checkpoint_hash'],m['test_hash'],m['validation_hash'],m['source'],digest(m['runtime']),m['metrics_schema'])
        if key in identities and identities[key]!=identity:raise ValueError('Mixed model/experiment identities')
        identities[key]=identity;metadata[key]=m
        comparison_contract=(m['test_hash'],m['validation_hash'],m['source'],digest(m['runtime']),m['metrics_schema'])
        if key[0] in test_contracts and test_contracts[key[0]]!=comparison_contract:raise ValueError('Incompatible evaluation contracts')
        test_contracts[key[0]]=comparison_contract
        cell=cells.setdefault(key,{})
        for r in rows:
            rkey=(r['scenario'],r['realization'])
            if rkey in cell:raise ValueError('Duplicate rollout comparison cell')
            if r['horizon']!=3600 or r['sample_count']!=3600:raise ValueError('Incomplete evaluation horizon')
            if not np.isfinite(r['mean_queue']) or r['mean_queue']<0:raise ValueError('Invalid queue')
            if not np.isclose(r['integrated_queue'],3600*r['mean_queue'],rtol=1e-12):raise ValueError('Queue integral mismatch')
            if len(r.get('minutes',[]))!=60 or r.get('window_counts')!=[1200]*3:raise ValueError('Incomplete detailed metrics')
            pairkey=(key[0],)+rkey
            pair=(r['demand_hash'],r['sumo_seed'],r['arrival_seed'],r['policy_seed'])
            if pairkey in pairing and pairing[pairkey]!=pair:raise ValueError('Unpaired traffic/SUMO/policy seeds')
            pairing[pairkey]=pair;cell[rkey]=r
    expected={(n,a,m,sel,s) for n in networks for a in controllers for m in p['methods'] for sel in selections for s in seeds}
    if set(cells)!=expected:raise ValueError('Incomplete requested model/seed matrix; missing '+str(sorted(expected-set(cells))))
    for key,rows in cells.items():
        scenarios={s['id'] for s in definitions(key[0],'seen')+definitions(key[0],'test')}
        if set(rows)!={(s,i) for s in scenarios for i in range(10)}:raise ValueError('Incomplete prescribed 23 x 10 evaluation suite')
    seed_rows=[];paired=[];by_seed={};auxiliary=[]
    for key,rows in sorted(cells.items()):
        n,a,m,sel,s=key
        for scope,scenes in scopes(n).items():
            values=scope_values(rows,scenes);mean=float(np.mean(values))
            by_seed[key[:4]+(scope,s)]=mean
            seed_rows.append(dict(network=n,controller=a,method=m,selection=sel,scope=scope,seed=s,
                                  selected_episode=metadata[key].get('selected_episode'),
                                  best_episode=metadata[key].get('best_episode'),completed_episodes=metadata[key].get('completed_episodes'),
                                  mean_queue=mean,rollout_sd=sample_sd(values),n_realizations=10,n_scenarios=len(scenes)))
            chosen=[rows[scene,i] for scene in scenes for i in range(10)]
            aux=dict(network=n,controller=a,method=m,selection=sel,scope=scope,seed=s)
            for metric in ('peak_queue','p95_queue','last_600_s_slope','inserted','arrived','completed','removed','remaining','pending',
                           'teleports','collisions','mean_speed','completion_rate'):
                valid=[r[metric] for r in chosen if r.get(metric) is not None]
                aux[metric]=float(np.mean(valid)) if valid else None
            trip_count=sum(r['trips']['count'] for r in chosen)
            aux['completed_trip_samples']=trip_count
            for metric in ('travel_time','waiting_time','time_loss','departure_delay'):
                aux[metric]=sum(r['trips']['sums'][metric] for r in chosen)/trip_count if trip_count else None
            auxiliary.append(aux)
        comparisons=[]
        if m!='baseline':comparisons.append('baseline')
        if m=='online_wce':comparisons.extend(x for x in p['methods'] if x not in ('baseline','online_wce'))
        for comparator in comparisons:
            baseline=cells[n,a,comparator,sel,s]
            for scope,scenes in scopes(n).items():
                values=scope_values(rows,scenes);other=scope_values(baseline,scenes);difference=values-other
                ci=bootstrap_mean_ci(difference,[n,a,m,comparator,sel,s,scope])
                paired.append(dict(network=n,controller=a,method=m,comparator=comparator,selection=sel,scope=scope,seed=s,
                    method_minus_comparator=float(difference.mean()),paired_rollout_sd=sample_sd(difference),
                    improvement_percent=float(100*(other.mean()-values.mean())/other.mean()) if other.mean()!=0 else None,
                    wins=int(np.sum(difference<0)),ties=int(np.sum(difference==0)),n_pairs=10,
                    ci_lower=ci[0],ci_upper=ci[1],uncertainty='evaluation_realizations_given_trained_models'))
    summary=[]
    for key in sorted({k[:-1] for k in by_seed}):
        values=[by_seed[key+(s,)] for s in seeds]
        summary.append(dict(zip(['network','controller','method','selection','scope'],key),
            mean_queue=float(np.mean(values)),training_seed_sd=sample_sd(values),n_training_seeds=len(seeds)))
    paired_seeds=[]
    grouped={}
    for row in paired:
        key=tuple(row[k] for k in ('network','controller','method','comparator','selection','scope'))
        grouped.setdefault(key,[]).append(row['method_minus_comparator'])
    for key,values in sorted(grouped.items()):
        paired_seeds.append(dict(zip(['network','controller','method','comparator','selection','scope'],key),
            method_minus_comparator=float(np.mean(values)),training_seed_sd=sample_sd(values),n_training_seeds=len(values)))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name,records in [('seeds',seed_rows),('comparison',summary),('paired_differences',paired),
                         ('paired_training_seeds',paired_seeds),('auxiliary_metrics',auxiliary)]:
        with (out/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    atomic_json(out/'summary.json',dict(comparisons=summary,paired_training_seeds=paired_seeds))
    atomic_json(out/'manifest.json',dict(status='complete',inputs=list(map(str,inputs)),training_seeds=seeds,
        n_training_seeds=len(seeds),networks=networks,controllers=controllers,selections=selections,
        note='Rollout CIs describe evaluation randomness conditional on trained models; single-seed training SD is null.'))
    return dict(status='complete',n_training_seeds=len(seeds),model_cells=len(cells),output=str(out))
