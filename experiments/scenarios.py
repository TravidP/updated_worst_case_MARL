"""Frozen seen, validation and test demand scenarios."""
import json
import numpy as np
from experiments.core import ROOT, digest, write_json
from experiments.demand import original_profiles, profiles, mixture, materialize, save_artifact, check_artifact
from experiments.protocol import settings, dataset_root


def definitions(network, split):
    names = [g['name'] for g in original_profiles(network)]
    def scenario(name, family, seed, blocks):
        return {'id': name, 'network': network, 'split': split, 'family': family,
                'generation_seed': seed, 'horizon': 3600, 'blocks': blocks}
    def block(start, duration, profile='Uniform', **extra):
        return dict(start=start, duration=duration, profile=profile, **extra)
    if split == 'seen':
        return [scenario('seen_' + n, 'seen', None, [block(0,3600,n)]) for n in names]
    validation = split == 'validation'
    if split not in ('test','validation'):
        raise ValueError('Unknown split')
    sigmas = settings()['validation_redistribution_sigma'] if validation else [.25,.5,.75]
    seeds = iter(settings()['validation_generation_seeds'] if validation else settings()['test_generation_seeds'])
    result = []
    for sigma in sigmas:
        seed = next(seeds)
        result.append(scenario('redistribution_' + str(sigma), 'redistribution', seed,
                               [block(0,3600,sigma=sigma)]))
    for index in range(2 if validation else 3):
        seed = next(seeds)
        weights = np.random.RandomState(seed).dirichlet(np.ones(11)).tolist()
        result.append(scenario('mixture_' + str(index+1), 'mixture', seed,
                               [block(0,3600,weights=weights)]))
    for interval in ([settings()['validation_switch_seconds']] if validation else [300,900,1200]):
        seed = next(seeds)
        blocks = [block(start,min(interval,3600-start),'N_to_S' if i%2==0 else 'W_to_E')
                  for i,start in enumerate(range(0,3600,interval))]
        result.append(scenario('switch_' + str(interval), 'temporal', seed, blocks))
    for peak in ([settings()['validation_peak']] if validation else [1.1,1.25,1.5]):
        seed = next(seeds)
        result.append(scenario('peak_' + str(peak), 'peak', seed,
                     [block(0,1200),block(1200,1200,multiplier=peak),block(2400,1200)]))
    return result


def block_rows(groups, definition, block):
    if 'weights' in block:
        rows = mixture(groups, block['weights'])
    else:
        rows = list(next(g for g in groups if g['name'] == block['profile'])['rows'])
    rates = np.array([r[2] for r in rows])
    if 'sigma' in block:
        rates *= np.random.RandomState(definition['generation_seed']).lognormal(0,block['sigma'],len(rates))
        rates *= sum(r[2] for r in rows) / rates.sum()
    rates *= block.get('multiplier',1.)
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('Invalid generated OD rates')
    return [(r[0],r[1],float(rate)) for r,rate in zip(rows,rates) if rate > 0]


def artifact_for(network, definition, arrival_seed, env):
    directory = dataset_root(network) / ('test' if definition['split']=='seen' else definition['split']) / 'artifacts'
    directory.mkdir(parents=True,exist_ok=True)
    groups = profiles(network)
    metadata = dict(network=network, network_hash=env.asset_hashes['network'], horizon=3600,
                    arrival_seed=arrival_seed, scenario=definition,
                    profiles_hash=digest(groups), scenario_hash=digest(definition))
    path = directory / ('{}_{}.json'.format(definition['id'],arrival_seed))
    if path.exists():
        artifact = json.loads(path.read_text()); check_artifact(artifact)
        if any(artifact.get(k)!=v for k,v in metadata.items()):
            raise ValueError('Artifact inputs changed: '+str(path))
        return path
    rng = np.random.RandomState(arrival_seed)
    vehicles=[]; cursor=0
    for i,b in enumerate(definition['blocks']):
        if b['start']!=cursor or b['duration']<=0:
            raise ValueError('Noncontiguous scenario')
        vehicles += materialize(block_rows(groups,definition,b),cursor,b['duration'],rng,env.resolve_route,'block'+str(i))
        cursor+=b['duration']
    if cursor!=3600: raise ValueError('Wrong scenario horizon')
    save_artifact(path,vehicles,metadata)
    return path


def materialize_suite(network, controller, output, splits=('seen','test','validation'), rollouts=10):
    from envs.experiment_env import make_environment
    output.mkdir(parents=True,exist_ok=False)
    env=make_environment(network,controller,output/'runtime',9001)
    try:
        env.reset_episode(61001,evaluation=True,horizon=3600)
        env.prepare_routes()
        artifacts=[str(artifact_for(network,d,seed,env).relative_to(ROOT)) for split in splits
                   for d in definitions(network,split) for seed in settings()['arrival_seeds'][:rollouts]]
        write_json(output/'manifest.json',{'network':network,'artifacts':artifacts,'status':'complete'})
        return {'artifacts':len(artifacts),'manifest':str(output/'manifest.json')}
    finally:
        env.terminate()
