"""Single controller loop for parent, frozen-WCE and all continuation methods.

Run with: python -m revision.runner --help
Publication runs require a current passing verification gate; pilots never use publication seeds.
"""
import argparse
import json
import time
import signal
import pickle
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from experiments.core import ROOT, METHODS, Streams, RunRecord, digest, file_hash, validate_rollout, write_json
from experiments.demand import profiles, mixture, materialize, save_artifact, check_artifact
from experiments.checkpoint import inspect_checkpoint, load_checkpoint, save_checkpoint


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['parent', 'wce', 'continue', 'evaluate', 'demand'], required=True)
    p.add_argument('--network', choices=['grid', 'monaco'], required=True)
    p.add_argument('--controller', choices=['ia2c', 'ma2c', 'iqll', 'ppo'], required=True)
    p.add_argument('--output', required=True, help='New exclusive run/attempt directory')
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--method', choices=METHODS, default='baseline')
    p.add_argument('--parent', help='Exact revision parent checkpoint directory')
    p.add_argument('--wce', help='Exact pretrained WCE checkpoint directory')
    p.add_argument('--resume', help='Exact same-stage revision checkpoint; new attempt directory required')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--config', help='Pilot-only isolated controller INI override')
    display = p.add_mutually_exclusive_group()
    display.add_argument('--visualization', action='store_true',
                         help='Show a local SUMO GUI window during each episode (default: off)')
    display.add_argument('--no-visualization', dest='visualization', action='store_false',
                         help='Run SUMO without a GUI window')
    p.set_defaults(visualization=False)
    p.add_argument('--steps', type=int, help='Pilot parent/continuation learning budget')
    p.add_argument('--episodes', type=int, help='Pilot offline WCE episode count')
    p.add_argument('--checkpoint-every', type=int, default=10)
    p.add_argument('--monitor-every', type=int, default=50)
    p.add_argument('--monitor-rollouts', type=int, default=3)
    p.add_argument('--gate', help='Passing verification gate for publication training')
    p.add_argument('--artifact', help='Materialized evaluation traffic JSON')
    p.add_argument('--sumo-seed', type=int, default=61001)
    p.add_argument('--policy-seed', type=int, default=71001)
    p.add_argument('--arrival-seed', type=int, default=51001)
    p.add_argument('--schedule', help='JSON blocks: start, duration, eleven weights; total 3600 seconds')
    p.add_argument('--fail-after-steps', type=int, help='Pilot-only evaluation fault injection')
    return p


def require_gate(path):
    if not path:
        raise ValueError('Publication runs require --gate from revision.verify')
    gate = json.loads(Path(path).read_text())
    if gate.get('status') != 'passed' or gate.get('corrections') != ['C{:02}'.format(i) for i in range(1, 11)]:
        raise ValueError('Verification gate has not passed all corrections')
    for name, expected in gate['source_hashes'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Source changed after verification: ' + name)
    for name, expected in gate.get('input_hashes', {}).items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Input changed after verification: ' + name)
    for name, expected in gate['evidence'].items():
        if file_hash(Path(path).parent / name) != expected:
            raise ValueError('Verification evidence changed')


def run(args):
    stage_started = time.monotonic()
    from envs.experiment_env import make_environment, validate_visualization
    args.visualization = getattr(args, 'visualization', False)
    validate_visualization(args.visualization)
    from agents.controller import Controller
    from agents.wce import WCE
    from experiments.protocol import settings
    protocol = settings()
    from experiments.checkpoint import checkpoint_identity
    for selected in (args.parent, args.wce, args.resume):
        if selected:
            checkpoint_identity(selected, args.network, args.controller, args.seed)
    if args.stage == 'evaluate' and args.parent:
        parent_manifest = Path(args.parent).parent / 'manifest.json'
        if parent_manifest.exists():
            pm = json.loads(parent_manifest.read_text())
            if pm.get('stage') == 'continue':
                if (pm['network'], pm['controller'], pm['seed']) != (args.network, args.controller, args.seed):
                    raise ValueError('Evaluation identity does not match trained controller')
                args.method = pm['method']
    if args.monitor_every < 1 or not 1 <= args.monitor_rollouts <= 3:
        raise ValueError('Monitoring interval must be positive; rollouts must be 1..3')
    if not args.pilot and (args.monitor_every, args.monitor_rollouts) != (50, 3):
        raise ValueError('Publication monitoring requires every 50 episodes and three rollouts')
    if args.checkpoint_every < 1:
        raise ValueError('checkpoint-every must be positive')
    if args.fail_after_steps is not None and (not args.pilot or args.stage != 'evaluate'):
        raise ValueError('Fault injection is only available for pilot evaluation')
    if args.config and not args.pilot:
        raise ValueError('--config is pilot-only; publication runs require tracked revised INIs')
    if args.pilot:
        if args.seed in protocol['training_seeds']:
            raise ValueError('Pilot seeds must not overlap publication training seeds')
    elif args.stage in ('parent', 'wce', 'continue'):
        require_gate(args.gate)
        if args.seed not in protocol['training_seeds']:
            raise ValueError('Use the prescribed publication training seed: 101')
        if args.steps is not None or args.episodes is not None:
            raise ValueError('Publication budgets cannot be overridden')
    goal = (args.steps if args.steps is not None else (protocol['parent_steps'] if args.stage == 'parent' else protocol['continuation_steps']))
    if args.stage == 'wce':
        goal = 1320 * (args.episodes if args.episodes is not None else protocol['offline_episodes'])
    if goal <= 0 or (args.stage in ('wce', 'continue') and goal % 1320):
        raise ValueError('WCE/continuation budgets must contain full 1320-step episodes')
    parents = {name: {'path': str(Path(path).resolve()), 'hash': inspect_checkpoint(path)['hash']}
               for name, path in [('controller', args.parent), ('wce', args.wce), ('resume', args.resume)] if path}
    from experiments.configuration import (effective_dict, load_controller_config,
                                           load_wce_config)
    controller_config, config_path = load_controller_config(
        args.network, args.controller, args.config, args.seed)
    wce_config, selected_wce_path = load_wce_config(args.network)
    inputs = dict(vars(args), parents=parents, training_profiles=profiles(args.network),
                  controller_config_hash=file_hash(config_path),
                  controller_config_path=str(config_path),
                  controller_effective_config=effective_dict(controller_config),
                  wce_config_path=str(selected_wce_path),
                  wce_config_hash=file_hash(selected_wce_path),
                  wce_effective_config=effective_dict(wce_config),
                  revision_overrides={'objective': 'queue', 'controller_reward': 'learner_boundary_scaled_v3',
                                      'episode_seconds': 3600 if args.stage in ('demand', 'evaluate') else 6600,
                                      'control_seconds': 5})
    if args.artifact:
        inputs['artifact_file_hash'] = file_hash(args.artifact)
    if args.schedule:
        inputs['schedule_file_hash'] = file_hash(args.schedule)
    record = RunRecord(args.output, inputs)
    record.started = stage_started
    env = controller = wce = telemetry = None
    models = {}
    stage_steps, episode = 0, 0
    streams = Streams(args.seed)
    try:
        env = make_environment(args.network, args.controller, record.path / 'runtime', args.seed,
                               visualization=args.visualization, config_path=args.config)
        obs = env.reset_episode(args.sumo_seed)
        env.prepare_routes()
        write_json(record.path / 'environment.json', {'assets': env.asset_hashes, 'lanes': env.metric.lanes,
                   'nodes': env.node_names, 'node_lanes': {n: env.nodes[n].ilds_in for n in env.node_names},
                   'neighbors': {n: env.nodes[n].neighbor for n in env.node_names},
                   'configuration': {s: dict(env.config[s]) for s in env.config.sections()}})
        if args.stage == 'demand':
            schedule = json.loads(Path(args.schedule).read_text()) if args.schedule else [
                {'start': 0, 'duration': 3600, 'weights': [int(g['name'] == 'Uniform') for g in env.groups]}]
            vehicles, cursor = [], 0
            rng = np.random.RandomState(args.arrival_seed)
            for index, block in enumerate(schedule):
                if block['start'] != cursor or block['duration'] <= 0:
                    raise ValueError('Schedule must cover 3600 seconds without gaps/overlap')
                vehicles += materialize(mixture(env.groups, block['weights']), cursor, block['duration'],
                                        rng, env.resolve_route, 'block{}'.format(index))
                cursor += block['duration']
            if cursor != 3600:
                raise ValueError('Evaluation demand horizon must be 3600 seconds')
            artifact = save_artifact(record.path / 'demand.json', vehicles,
                {'network': args.network, 'network_hash': env.asset_hashes['network'], 'horizon': 3600,
                 'arrival_seed': args.arrival_seed, 'schedule': schedule})
            record.finish('complete', demand_hash=artifact['hash'], scheduled=len(vehicles))
            return str(record.path / 'demand.json')
        controller = Controller(env, args.controller, args.seed)
        models['controller'] = controller
        if args.stage != 'parent' and not args.parent and not args.resume:
            raise ValueError('An explicit parent/final controller checkpoint is required')
        if args.parent:
            load_checkpoint(args.parent, {'controller': controller})
        if args.stage in ('continue', 'wce') and not args.pilot and controller.learning_steps != 1000000:
            raise ValueError('Training must branch from an exact one-million-step parent')
        use_wce = args.stage == 'wce' or (args.stage == 'continue' and args.method in ('fixed_wce', 'online_wce'))
        if use_wce:
            wce = WCE(env, args.seed)
            models['wce'] = wce
            if args.stage == 'continue' and not args.wce and not args.resume:
                raise ValueError('Fixed/online methods require the same explicit pretrained WCE')
            if args.wce:
                wm = inspect_checkpoint(args.wce)
                if wm['parents'].get('controller', {}).get('hash') != parents['controller']['hash']:
                    raise ValueError('WCE was trained against a different controller parent')
                load_checkpoint(args.wce, {'wce': wce})
        initial_steps = controller.learning_steps
        if args.resume:
            state = load_checkpoint(args.resume, models)
            if (state['stage'], state['method'], state['goal']) != (args.stage, args.method, goal):
                raise ValueError('Resume stage/method/budget mismatch')
            streams.restore(state['streams'])
            stage_steps, episode, initial_steps = state['stage_steps'], state['episode'], state['initial_steps']
        if args.stage == 'evaluate':
            if not args.artifact:
                raise ValueError('Evaluation requires a complete --artifact')
            if not args.pilot and controller.learning_steps != 2320000:
                raise ValueError('Evaluate the exact final-budget checkpoint')
            artifact = json.loads(Path(args.artifact).read_text())
            check_artifact(artifact)
            if artifact['network_hash'] != env.asset_hashes['network'] or artifact['horizon'] != 3600:
                raise ValueError('Demand artifact network/horizon mismatch')
            obs = env.reset_episode(args.sumo_seed, evaluation=True, horizon=3600)
            controller.reset()
            controller.rng.rng['policy'] = np.random.RandomState(args.policy_seed)
            env.inject(artifact['vehicles'])
            for index in range(720):
                decision = controller.act(obs, env, False)
                nxt, rewards, done = env.step(decision[0])
                learner_rewards, clipped = controller.observe(
                    obs, decision, rewards, nxt, done, False)
                env.controller_rows[-1].update(
                    learner_rewards=learner_rewards.tolist(),
                    reward_clipped=clipped.astype(int).tolist())
                obs = nxt
                if (index + 1) % 120 == 0:
                    with (record.path / 'progress.jsonl').open('a') as f:
                        f.write(json.dumps({'stage': 'evaluate', 'episode': 1, 'simulation_steps': index + 1,
                            'learning_steps': controller.learning_steps, 'goal': 720,
                            'mean_queue': float(np.mean([r['queue'] for r in env.rows[-600:]])),
                            'wce_updates': 0, 'wall_seconds': time.monotonic() - stage_started}) + '\n')
                if args.fail_after_steps == index + 1:
                    raise RuntimeError('Deliberate pilot evaluation interruption')
            summary = validate_rollout(env.rows)
            env.export_episode(record.path / 'rollout')
            env.terminate()
            trips = [t for t in ET.parse(str(env.trip_file)).getroot().findall('tripinfo')
                     if float(t.get('arrival', '-1')) >= 0]
            summary['completed_trip_denominator'] = len(trips)
            for label, attribute in [('travel_time', 'duration'), ('waiting_time', 'waitingTime'),
                                     ('time_loss', 'timeLoss'), ('departure_delay', 'departDelay')]:
                summary['mean_completed_' + label] = (sum(float(t.get(attribute, '0')) for t in trips) / len(trips)
                                                       if trips else None)
            summary.update(scheduled=len(artifact['vehicles']), demand_hash=artifact['hash'],
                           effective_sumo_seed=env.effective_seed, checkpoint=parents['controller'])
            write_json(record.path / 'rollout_summary.json', summary)
            record.finish('complete', **{k: v for k, v in summary.items() if k != 'status'})
            return str(record.path / 'rollout_summary.json')
        from experiments.telemetry import Telemetry
        from experiments.monitoring import run_monitor
        telemetry = Telemetry(record.path)
        checkpoint = Path(args.resume) if args.resume else None
        monitored_steps = set(state.get('monitored_steps', [])) if args.resume else set()

        def save_current():
            controller.assert_finite()
            target = record.path / ('checkpoint_{:09}'.format(stage_steps))
            if not target.exists():
                snapshot = {'stage':args.stage, 'method':args.method, 'goal':goal, 'stage_steps':stage_steps,
                    'episode':episode, 'initial_steps':initial_steps, 'streams':streams.state(),
                    'monitored_steps':sorted(monitored_steps), 'monitor_every':args.monitor_every,
                    'monitor_rollouts':args.monitor_rollouts}
                save_checkpoint(target, models, snapshot, parents)
            return target

        def monitor():
            nonlocal checkpoint
            if stage_steps in monitored_steps:
                return
            checkpoint = save_current()
            # A completed round in a previous attempt can be reused; failures are retried in this attempt.
            previous = Path(args.resume).parent/'monitoring'/('round_%06d' % episode)/'summary.json' if args.resume else None
            if previous and previous.exists():
                result = json.loads(previous.read_text())
                if result.get('status') == 'complete' and result.get('stage_simulation_steps') == stage_steps:
                    reuse = record.path/'monitoring'/('round_%06d' % episode)
                    reuse.mkdir(parents=True, exist_ok=True)
                    write_json(reuse/'summary.json',dict(result,reused_from=str(previous)))
                    monitored_steps.add(stage_steps)
                    return result
            t = time.monotonic()
            before_monitor = hashlib.sha256(pickle.dumps((controller.state(), streams.state()), protocol=4)).hexdigest()
            result = run_monitor(args,record.path,checkpoint,episode,stage_steps,telemetry)
            after_monitor = hashlib.sha256(pickle.dumps((controller.state(), streams.state()), protocol=4)).hexdigest()
            if before_monitor != after_monitor:
                raise AssertionError('Monitoring changed training state')
            monitored_steps.add(stage_steps)
            record.add_time('monitoring', time.monotonic()-t)
            return result

        if args.resume and (state.get('monitor_every'),state.get('monitor_rollouts')) != (args.monitor_every,args.monitor_rollouts):
            raise ValueError('Resume monitoring settings mismatch')
        if args.stage in ('parent','continue') and (not args.resume or stage_steps == goal or episode % args.monitor_every == 0):
            monitor()
        while stage_steps < goal:
            episode_started = time.monotonic()
            updates_before = controller.backward_calls
            reset_start = time.monotonic()
            obs = env.reset_episode(streams['sumo'].randint(1, 2147483647))
            controller.reset()
            record.add_time('reset', time.monotonic() - reset_start)
            for block in range(11):
                t = time.monotonic()
                wobs = logits = value = None
                if use_wce:
                    wobs = env.wce_observation()
                    weights, logits, value = wce.act(wobs)
                elif args.stage == 'parent' or args.method == 'baseline':
                    weights = np.eye(11)[block]
                elif args.method == 'random_group':
                    weights = np.eye(11)[streams['selection'].randint(11)]
                else:
                    weights = streams['selection'].dirichlet(np.ones(11))
                record.add_time('demand_selection', time.monotonic() - t)
                t = time.monotonic()
                vehicles = materialize(mixture(env.groups, weights), block * 600, 600,
                    streams['demand'], env.resolve_route, 'e{}_b{}'.format(episode, block))
                env.inject(vehicles)
                record.add_time('demand_generation_insertion', time.monotonic() - t)
                block_start = len(env.lane_rows)
                for _ in range(min(120, goal - stage_steps)):
                    learning = args.stage != 'wce'
                    t = time.monotonic()
                    decision = controller.act(obs, env, learning)
                    record.add_time('controller_inference', time.monotonic() - t)
                    t = time.monotonic()
                    nxt, rewards, done = env.step(decision[0])
                    record.add_time('simulation_measurement', time.monotonic() - t)
                    t = time.monotonic()
                    learner_rewards, reward_clipped = controller.observe(
                        obs, decision, rewards, nxt, done, learning)
                    record.add_time('controller_learning', time.monotonic() - t)
                    obs = nxt
                    env.controller_rows[-1].update(learning_steps=controller.learning_steps,
                        stage_simulation_steps=stage_steps+1, learning=learning,
                        learner_rewards=learner_rewards.tolist(),
                        reward_clipped=reward_clipped.astype(int).tolist())
                    if learning:
                        telemetry.scalars({
                            'train/raw_reward_by_learning_step': float(np.mean(rewards)),
                            'train/learner_reward_by_learning_step': float(np.mean(learner_rewards)),
                            'train/reward_by_learning_step': float(np.mean(learner_rewards)),
                            'train/reward_clip_fraction_by_learning_step': float(np.mean(reward_clipped)),
                        }, controller.learning_steps)
                    telemetry.updates(controller)
                    stage_steps += 1
                    if stage_steps % 120 == 0:
                        progress = {'stage': args.stage, 'episode': episode + 1, 'simulation_steps': stage_steps,
                            'learning_steps': controller.learning_steps, 'goal': goal,
                            'mean_queue': float(np.mean([r['queue'] for r in env.rows[-600:]])),
                            'wce_updates': wce.updates if wce else 0, 'wall_seconds': time.monotonic() - stage_started}
                        telemetry.scalars({'block/mean_total_queue_recent_600_seconds':progress['mean_queue']}, controller.learning_steps if learning else stage_steps//120)
                        with (record.path / 'progress.jsonl').open('a') as f:
                            f.write(json.dumps(progress) + '\n')
                block_samples = env.lane_rows[block_start:]
                block_reward = env.metric.wce(block_samples) if len(block_samples) == 600 else None
                with (record.path / 'demand_decisions.jsonl').open('a') as stream:
                    stream.write(json.dumps({'episode': episode, 'block': block, 'weights': weights.tolist(),
                        'wce_raw_reward': block_reward, 'completed_seconds': len(block_samples),
                        'scheduled_vehicles': len(vehicles), 'traffic_hash': digest(vehicles)}) + '\n')
                if use_wce and (args.stage == 'wce' or args.method == 'online_wce'):
                    t = time.monotonic()
                    wce_learner_reward, wce_clipped = wce.observe(
                        wobs, logits, value, block_reward)
                    with (record.path / 'wce_rewards.jsonl').open('a') as stream:
                        stream.write(json.dumps({
                            'episode': episode, 'block': block,
                            'raw_reward': block_reward,
                            'learner_reward': wce_learner_reward,
                            'clipped': bool(wce_clipped),
                        }) + '\n')
                    telemetry.updates(wce, 'wce')
                    record.add_time('wce_learning', time.monotonic() - t)
                if stage_steps == goal:
                    break
            controller.flush(obs, env.cur_sec == 6600)
            telemetry.updates(controller)
            controller.assert_finite()
            episode += 1
            telemetry.episode(env,controller,args.stage,episode,stage_steps,time.monotonic()-episode_started,updates_before)
            env.export_episode(record.path / ('episode_{:04}'.format(episode)))
            if episode % args.checkpoint_every == 0 or stage_steps == goal:
                t = time.monotonic()
                checkpoint = save_current()
                record.add_time('checkpoint',time.monotonic()-t)
            if args.stage in ('parent','continue') and (episode % args.monitor_every == 0 or stage_steps == goal):
                monitor()
            print('{} {} {} episode={} simulation_steps={} learning_steps={}'.format(
                args.network, args.controller, args.stage, episode, stage_steps, controller.learning_steps), flush=True)
        expected = initial_steps + (0 if args.stage == 'wce' else goal)
        if controller.learning_steps != expected:
            raise AssertionError('Incorrect controller-learning budget')
        record.finish('complete', checkpoint=str(checkpoint), learning_steps=controller.learning_steps,
                      stage_simulation_steps=stage_steps, backward_calls=controller.backward_calls,
                      minibatch_updates_per_agent=controller.minibatch_updates,
                      wce_updates=wce.updates if wce else 0)
        return str(checkpoint)
    except BaseException as exc:
        if env is not None and env.rows:
            env.export_episode(record.path / 'incomplete_attempt')
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record.finish(status, error=repr(exc), completed_stage_steps=stage_steps,
                      completed_episodes=episode)
        raise
    finally:
        if telemetry is not None:
            telemetry.close()
        for model in models.values():
            model.close()
        if env is not None:
            env.terminate()


if __name__ == '__main__':
    run(parser().parse_args())
