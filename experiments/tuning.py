"""Generate the prescribed 24 development candidates without running training."""
import configparser
import itertools
from pathlib import Path
from experiments.protocol import config_path
from experiments.storage import atomic_json

def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
    for network,family in itertools.product(['grid','monaco'],['ma2c','ppo','dql']):
        rates={'ma2c':['5e-4','2.5e-4'],'ppo':['3e-4','1.5e-4'],'dql':['1e-4','3e-4']}[family]
        key='buffer_size' if family=='dql' else 'entropy_coef_init'
        values=['50000','20000'] if family=='dql' else ['0.01','0.001']
        for rank,(lr,value) in enumerate(itertools.product(rates,values)):
            c=configparser.ConfigParser();c.read(config_path(network,family));c['MODEL_CONFIG']['lr_init']=lr;c['MODEL_CONFIG'][key]=value
            name='%s_%s_%d.ini'%(network,family,rank)
            with (output/name).open('w') as f:c.write(f)
            rows.append(dict(network=network,controller=family,config=name,tie_rank=rank,seed=9001,episodes=1000))
    atomic_json(output/'candidates.json',rows);return len(rows)
