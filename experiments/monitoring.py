"""Isolated frozen-model 600-second Uniform monitoring, separate from final tests."""
import csv
import fcntl
import random
import json
import time
from pathlib import Path
import numpy as np
from experiments.core import write_json, digest
from experiments.demand import materialize, save_artifact, check_artifact
from experiments.checkpoint import inspect_checkpoint, load_checkpoint


def csv_rows(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_monitor(args, run_path, checkpoint, episode, stage_steps, telemetry):
    from envs.experiment_env import make_environment
    from agents.controller import Controller
    from experiments.protocol import output_root
    from tf_compat import tf
    from experiments.telemetry import Telemetry
    numpy_state, python_state = np.random.get_state(), random.getstate()
    started = time.monotonic()
    root = Path(run_path)/'monitoring'
    root.mkdir(exist_ok=True)
    if not (root/'manifest.json').exists():
        write_json(root/'manifest.json', dict(network=args.network, controller=args.controller, horizon=600,
            profile='Uniform', repetitions=args.monitor_rollouts, every_episodes=args.monitor_every,
            arrival_seeds=list(range(53001,53001+args.monitor_rollouts)),
            sumo_seeds=list(range(63001,63001+args.monitor_rollouts)),
            policy_seeds=list(range(73001,73001+args.monitor_rollouts)),
            protocol='learner_boundary_scaled_v3'))
    target = root/('round_%06d' % episode)
    target.mkdir(exist_ok=False)
    identity = inspect_checkpoint(checkpoint)
    write_json(target/'checkpoint_reference.json', dict(path=str(checkpoint), hash=identity['hash'],
                                                       stage_simulation_steps=stage_steps))
    env = ctrl = None
    results, arrays = [], []
    try:
        env = make_environment(args.network, args.controller, target/'runtime', args.seed,
                               config_path=getattr(args, 'config', None))
        env.capture_waiting = True
        first_obs = env.reset_episode(63001, evaluation=True, horizon=600)
        ctrl = Controller(env, args.controller, args.seed)
        cache = output_root('evaluate')/'monitoring_inputs'/args.network/env.asset_hashes['network'][:16]
        cache.mkdir(parents=True, exist_ok=True)
        group = next(g for g in env.groups if g['name']=='Uniform')
        demand_id = digest(group['rows'])[:16]
        for index in range(args.monitor_rollouts):
            child = target/('rollout_%02d' % (index+1));child.mkdir()
            # The first reset above is already the exact first paired rollout.
            # Reuse it instead of paying for a second identical SUMO startup.
            obs = first_obs if index == 0 else env.reset_episode(
                63001+index, evaluation=True, horizon=600)
            load_checkpoint(checkpoint, {'controller':ctrl});ctrl.reset()
            ctrl.rng.rng['policy'] = np.random.RandomState(73001+index)
            before = ctrl.learning_steps
            weights = ctrl.sess.run(ctrl.variables)
            artifact_path = cache/('uniform_%s_%d.json' % (demand_id, 53001+index))
            with (cache/'artifact.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if artifact_path.exists():
                    artifact = json.loads(artifact_path.read_text());check_artifact(artifact)
                else:
                    vehicles = materialize(group['rows'], 0, 600, np.random.RandomState(53001+index), env.resolve_route, 'monitor')
                    artifact = save_artifact(artifact_path, vehicles, dict(network=args.network,
                        network_hash=env.asset_hashes['network'], profile_hash=digest(group['rows']),
                        arrival_seed=53001+index, horizon=600))
            if artifact['profile_hash'] != digest(group['rows']) or artifact['network_hash'] != env.asset_hashes['network']:
                raise ValueError('Monitoring artifact identity mismatch')
            write_json(child/'traffic_reference.json', dict(path=str(artifact_path), hash=artifact['hash']))
            env.inject(artifact['vehicles'])
            for _ in range(120):
                action = ctrl.act(obs,env,False)
                nxt,rewards,done=env.step(action[0])
                learner,clipped=ctrl.observe(obs,action,rewards,nxt,done,False)
                env.controller_rows[-1].update(
                    learner_rewards=learner.tolist(),
                    reward_clipped=clipped.astype(int).tolist())
                obs=nxt
            assert done and [r['time'] for r in env.rows] == list(range(1,601))
            assert ctrl.learning_steps == before
            assert all(np.array_equal(a,b) for a,b in zip(weights,ctrl.sess.run(ctrl.variables)))
            env.export_episode(child/'rollout')
            np.savez_compressed(str(child/'waiting.npz'), time=np.arange(1,601), lanes=np.array(env.metric.lanes),
                                current_wait=np.asarray(env.wait_rows))
            csv_rows(child/'timeseries.csv', env.rows)
            summary = dict(mean_total_queue=float(np.mean([r['queue'] for r in env.rows])),
                mean_current_wait_seconds=float(np.mean([r['current_wait_mean_seconds'] for r in env.rows])),
                stopped_vehicle_seconds=float(sum(r['queue'] for r in env.rows)),
                scheduled=env.scheduled, inserted=sum(r['inserted'] for r in env.rows),
                completed=sum(r['completed'] for r in env.rows), pending=env.rows[-1]['pending'],
                remaining=env.rows[-1]['active'], teleports=sum(r['teleports'] for r in env.rows),
                collisions=sum(r['collisions'] for r in env.rows), demand_hash=artifact['hash'],
                effective_sumo_seed=env.effective_seed)
            write_json(child/'summary.json',summary);results.append(summary)
            arrays.append(np.array([[r['queue'],r['current_wait_mean_seconds']] for r in env.rows]))
        means=np.mean(arrays,axis=0)
        sd=np.std(arrays,axis=0,ddof=1) if len(arrays)>1 else np.zeros_like(means)
        scalars={}
        for key in ['mean_total_queue','mean_current_wait_seconds','stopped_vehicle_seconds','completed','pending','teleports']:
            values=[r[key] for r in results]
            scalars[key]=float(np.mean(values))
            if len(values)>1:scalars[key+'_sd']=float(np.std(values,ddof=1))
        telemetry.scalars({'monitor/'+k:v for k,v in scalars.items()},ctrl.learning_steps)
        detail=Telemetry(target)
        try:
            for second,row in enumerate(means,1):
                detail.scalars({'monitor/queue_by_second':row[0], 'monitor/current_wait_by_second':row[1]},second)
            plot(target,means,sd,len(results),episode,ctrl.learning_steps)
            image=tf.Summary.Image(encoded_image_string=(target/'queue_waiting.png').read_bytes(), colorspace=3)
            telemetry.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='monitor/queue_waiting',image=image)]),ctrl.learning_steps)
            telemetry.writer.flush()
        finally:detail.close()
        write_json(target/'summary.json',dict(status='complete',episode=episode,learning_steps=ctrl.learning_steps,
            stage_simulation_steps=stage_steps,rollouts=results,metrics=scalars,elapsed_seconds=time.monotonic()-started))
        assert inspect_checkpoint(checkpoint)['hash']==identity['hash']
        print('MONITOR {} {} episode={} learning_steps={} mean_queue={:.3f}'.format(
            args.network,args.controller,episode,ctrl.learning_steps,scalars['mean_total_queue']),flush=True)
        return dict(status='complete', episode=episode,
                    learning_steps=ctrl.learning_steps,
                    stage_simulation_steps=stage_steps, rollouts=results,
                    metrics=scalars, elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        write_json(target/'failure.json',dict(status='failed',error=repr(exc)))
        if env is not None and env.rows:
            env.export_episode(target/'incomplete_rollout')
        raise
    finally:
        if ctrl is not None:ctrl.close()
        if env is not None:env.terminate()
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def plot(target,mean,sd,count,episode,steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for index,(ax,label) in enumerate(zip(axes,['Total queue (vehicles)','Mean current wait (seconds / present vehicle)'])):
        ax.plot(np.arange(1,601),mean[:,index],color='#2479a5')
        if count>1:ax.fill_between(np.arange(1,601),np.maximum(0,mean[:,index]-sd[:,index]),mean[:,index]+sd[:,index],alpha=.2)
        ax.set(xlabel='Simulation seconds',ylabel=label,xlim=(0,600),ylim=(0,None));ax.grid(alpha=.2)
    fig.suptitle('Uniform monitoring | episode {} | {:,} learning steps'.format(episode,steps))
    fig.text(.5,.01,'{} paired realizations; mean ± sample SD; empty start, no warm-up'.format(count),ha='center')
    fig.tight_layout(rect=[0,.06,1,.94])
    for suffix in ['png','svg']:fig.savefig(str(target/('queue_waiting.'+suffix)),dpi=140)
    plt.close(fig)
