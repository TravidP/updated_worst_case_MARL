"""Strict configuration loading for the revised experiment workflow.

The revised INI files describe trainable models and the traffic environment.  Run
identity, seeds and budgets remain protocol/CLI concerns and are injected only
after the on-disk file has passed validation.
"""
import configparser
import math
from pathlib import Path

from experiments.core import ROOT


FAMILIES = ('ia2c', 'ma2c', 'iqll', 'ppo')
NETWORK_SUFFIX = {'grid': 'large', 'monaco': 'real'}

COMMON_MODEL = {
    'max_grad_norm', 'gamma', 'lr_init', 'lr_decay', 'lr_min',
    'lr_decay_steps', 'batch_size', 'reward_norm', 'reward_clip',
    'tf_operation_timeout_ms', 'tf_intra_op_threads', 'tf_inter_op_threads',
}
ACTOR_CRITIC_MODEL = {
    'entropy_coef_init', 'entropy_coef_min', 'entropy_decay',
    'entropy_decay_steps', 'value_coef', 'num_lstm', 'num_fw', 'num_ft',
    'adv_norm',
}
MODEL_KEYS = {
    'ia2c': COMMON_MODEL | ACTOR_CRITIC_MODEL | {'rmsp_alpha', 'rmsp_epsilon'},
    'ma2c': COMMON_MODEL | ACTOR_CRITIC_MODEL | {'rmsp_alpha', 'rmsp_epsilon', 'num_fp'},
    'ppo': COMMON_MODEL | (ACTOR_CRITIC_MODEL - {'adv_norm'}) | {
        'adam_epsilon', 'ppo_clip_ratio', 'ppo_n_epoch', 'ppo_adv_norm',
    },
    'iqll': COMMON_MODEL | {
        'adam_epsilon', 'buffer_size', 'epsilon_init', 'epsilon_min',
        'epsilon_decay', 'epsilon_decay_steps', 'updates_per_trigger',
        'update_interval',
    },
}
ENV_COMMON = {
    'clip_wave', 'clip_wait', 'control_interval_sec', 'agent', 'coop_gamma',
    'data_path', 'episode_length_sec', 'norm_wave', 'norm_wait', 'coef_wait',
    'objective', 'scenario', 'yellow_interval_sec', 'fast_wait_metric',
    'sim_progress_log_sec', 'step_stage_warn_sec', 'trainer_stage_warn_sec',
    'stall_watchdog_sec', 'stall_watchdog_poll_sec',
}
ENV_NETWORK = {
    'grid': {'peak_flow1', 'peak_flow2', 'init_density'},
    'monaco': {'flow_rate'},
}
ENV_REQUIRED = {
    'clip_wave', 'clip_wait', 'control_interval_sec', 'agent', 'coop_gamma',
    'data_path', 'episode_length_sec', 'norm_wave', 'norm_wait', 'coef_wait',
    'objective', 'scenario', 'yellow_interval_sec',
}
WCE_KEYS = {
    'gamma', 'lr_init', 'lr_decay', 'lr_min', 'lr_decay_steps',
    'entropy_coef_init', 'entropy_coef_min', 'entropy_decay',
    'entropy_decay_steps', 'value_coef', 'max_grad_norm', 'rmsp_alpha',
    'rmsp_epsilon', 'reward_norm', 'reward_clip',
    'tf_operation_timeout_ms', 'tf_intra_op_threads', 'tf_inter_op_threads',
}


def controller_config_path(network, family):
    return ROOT / 'config' / 'revised' / ('config_{}_{}.ini'.format(family, NETWORK_SUFFIX[network]))


def wce_config_path(network):
    return ROOT / 'config' / 'revised' / ('config_wce_{}.ini'.format(NETWORK_SUFFIX[network]))


def _read(path):
    path = Path(path).resolve()
    parser = configparser.ConfigParser()
    if parser.read(str(path)) != [str(path)]:
        raise ValueError('Unable to read revised configuration: ' + str(path))
    return parser, path


def _exact_sections(parser, expected, path):
    observed = set(parser.sections())
    if observed != set(expected):
        raise ValueError('{} requires sections {}; found {}'.format(path, sorted(expected), sorted(observed)))


def _exact_keys(section, allowed, required, label):
    keys = set(section)
    unknown = keys - allowed
    missing = required - keys
    if unknown or missing:
        raise ValueError('{} configuration keys invalid; unknown={}, missing={}'.format(
            label, sorted(unknown), sorted(missing)))


def _positive(section, keys, label, allow_zero=()):
    for key in keys:
        if key not in section:
            continue
        value = section.getfloat(key)
        if not math.isfinite(value) or value < 0 or (value == 0 and key not in allow_zero):
            raise ValueError('{} {} must be {}'.format(label, key, 'nonnegative' if key in allow_zero else 'positive'))


def _integers(section, keys, label):
    for key in keys & set(section):
        value = section.getint(key)
        if value <= 0:
            raise ValueError('{} {} must be a positive integer'.format(label, key))


def _booleans(section, keys, label):
    for key in keys & set(section):
        try:
            section.getboolean(key)
        except ValueError as exc:
            raise ValueError('{} {} must be boolean'.format(label, key)) from exc


def _validate_decay(section, prefix, label):
    mode = section.get(prefix + '_decay')
    if mode not in ('constant', 'linear'):
        raise ValueError('{} {}_decay must be constant or linear'.format(label, prefix))
    value_prefix = prefix + '_coef' if prefix == 'entropy' else prefix
    conditional = {value_prefix + '_min', prefix + '_decay_steps'}
    present = conditional & set(section)
    if mode == 'linear' and present != conditional:
        raise ValueError('{} linear {} decay requires {}'.format(label, prefix, sorted(conditional)))
    if mode == 'constant' and present:
        raise ValueError('{} constant {} decay must not declare {}'.format(label, prefix, sorted(present)))


def load_controller_config(network, family, path=None, seed=1):
    if network not in NETWORK_SUFFIX or family not in FAMILIES:
        raise ValueError('Unknown revised network/controller: {}/{}'.format(network, family))
    parser, selected = _read(path or controller_config_path(network, family))
    _exact_sections(parser, ('MODEL_CONFIG', 'ENV_CONFIG'), selected)
    model, env = parser['MODEL_CONFIG'], parser['ENV_CONFIG']
    required = set(MODEL_KEYS[family]) - {
        'lr_min', 'lr_decay_steps', 'entropy_coef_min', 'entropy_decay_steps',
        'epsilon_decay_steps', 'tf_operation_timeout_ms', 'tf_intra_op_threads',
        'tf_inter_op_threads',
    }
    _exact_keys(model, MODEL_KEYS[family], required, family)
    _exact_keys(env, ENV_COMMON | ENV_NETWORK[network], ENV_REQUIRED | ENV_NETWORK[network], network)
    expected_scenario = 'large_grid' if network == 'grid' else 'real_net'
    if env.get('agent') != family or env.get('scenario') != expected_scenario:
        raise ValueError('Configuration identity does not match {}/{}'.format(network, family))
    if env.get('objective') != 'queue' or env.getfloat('coef_wait') != 0:
        raise ValueError('Revised controller objective must be queue with coef_wait=0')
    from experiments.protocol import settings
    protocol = settings()
    if env.getint('episode_length_sec') != protocol['training_seconds']:
        raise ValueError('INI episode_length_sec disagrees with protocol')
    if env.getint('control_interval_sec') != protocol['control_seconds']:
        raise ValueError('INI control_interval_sec disagrees with protocol')
    if not 0 <= env.getint('yellow_interval_sec') < env.getint('control_interval_sec'):
        raise ValueError('yellow_interval_sec must be within the control interval')
    _positive(model, {'gamma', 'lr_init', 'lr_min', 'reward_norm', 'reward_clip',
                      'max_grad_norm', 'entropy_coef_init', 'entropy_coef_min',
                      'value_coef', 'rmsp_alpha', 'rmsp_epsilon', 'adam_epsilon',
                      'epsilon_init', 'epsilon_min'}, family,
              allow_zero={'entropy_coef_init', 'entropy_coef_min', 'epsilon_min'})
    _integers(model, {'batch_size', 'num_lstm', 'num_fw', 'num_ft', 'num_fp',
                      'buffer_size', 'update_interval', 'updates_per_trigger',
                      'ppo_n_epoch', 'lr_decay_steps', 'entropy_decay_steps',
                      'epsilon_decay_steps', 'tf_operation_timeout_ms',
                      'tf_intra_op_threads', 'tf_inter_op_threads'}, family)
    _booleans(model, {'adv_norm', 'ppo_adv_norm'}, family)
    if not 0 < model.getfloat('gamma') <= 1:
        raise ValueError('gamma must be in (0, 1]')
    _validate_decay(model, 'lr', family)
    if family in ('ia2c', 'ma2c', 'ppo'):
        _validate_decay(model, 'entropy', family)
    if family == 'iqll':
        _validate_decay(model, 'epsilon', family)
        if not 0 <= model.getfloat('epsilon_min') <= model.getfloat('epsilon_init') <= 1:
            raise ValueError('epsilon values must satisfy 0 <= min <= init <= 1')
    if family == 'ppo':
        if not 0 < model.getfloat('ppo_clip_ratio') < 1 or model.getint('ppo_n_epoch') < 1:
            raise ValueError('Invalid PPO clip ratio or epoch count')
    _positive(env, {'control_interval_sec', 'episode_length_sec', 'norm_wave',
                    'norm_wait', 'clip_wave', 'clip_wait', 'coop_gamma',
                    'peak_flow1', 'peak_flow2', 'flow_rate'}, network,
              allow_zero={'coop_gamma'})
    _booleans(env, {'fast_wait_metric'}, network)
    # Runtime identity comes from CLI/protocol, not from the tracked INI.
    env['seed'] = str(int(seed))
    env['test_seeds'] = ','.join(map(str, protocol['sumo_seeds'][:3]))
    return parser, selected


def load_wce_config(network, path=None):
    if network not in NETWORK_SUFFIX:
        raise ValueError('Unknown revised network: ' + str(network))
    parser, selected = _read(path or wce_config_path(network))
    _exact_sections(parser, ('WCE_CONFIG',), selected)
    cfg = parser['WCE_CONFIG']
    required = WCE_KEYS - {
        'lr_min', 'lr_decay_steps', 'entropy_coef_min', 'entropy_decay_steps',
        'tf_operation_timeout_ms', 'tf_intra_op_threads', 'tf_inter_op_threads',
    }
    _exact_keys(cfg, WCE_KEYS, required, 'wce/' + network)
    _positive(cfg, {'gamma', 'lr_init', 'lr_min', 'max_grad_norm', 'reward_norm',
                    'reward_clip', 'entropy_coef_init', 'entropy_coef_min',
                    'value_coef', 'rmsp_alpha', 'rmsp_epsilon'}, 'wce/' + network,
              allow_zero={'entropy_coef_init', 'entropy_coef_min'})
    _integers(cfg, {'lr_decay_steps', 'entropy_decay_steps',
                    'tf_operation_timeout_ms', 'tf_intra_op_threads',
                    'tf_inter_op_threads'}, 'wce/' + network)
    if not 0 < cfg.getfloat('gamma') <= 1:
        raise ValueError('WCE gamma must be in (0, 1]')
    _validate_decay(cfg, 'lr', 'wce/' + network)
    _validate_decay(cfg, 'entropy', 'wce/' + network)
    return parser, selected


def effective_dict(parser):
    return {section: dict(parser[section]) for section in parser.sections()}
