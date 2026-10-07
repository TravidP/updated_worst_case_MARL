"""Verify, summarize and document the explicitly partial Monaco legacy replay."""
import argparse
import csv
import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.core import file_hash, write_json
from scripts.validate_external_results import validate
from scripts.analyze_external_results import analyze


def finalize(campaign, wait):
    directory = campaign.parent
    started = time.monotonic()
    while True:
        paths = list((directory/'monaco').glob('*/*/external/*/rollout_*/attempt_*/result.json'))
        completed = sum(json.loads(p.read_text()).get('status') == 'complete' for p in paths)
        if completed == 200:
            break
        if not wait or time.monotonic()-started > 10800:
            raise RuntimeError('Incomplete Monaco legacy replay: %d/200' % completed)
        time.sleep(10)
    metadata = json.loads(campaign.read_text())
    assert file_hash(ROOT/'scripts/monaco_legacy_campaign.py') == metadata['scheduler_hash']
    assert file_hash(ROOT/'scripts/legacy_monaco_evaluate.py') == metadata['evaluator_hash']
    accepted = validate(campaign, 'monaco')
    curves = {}
    for summary in accepted['summaries']:
        path = Path(summary)
        manifest = json.loads(path.with_name('manifest.json').read_text())
        artifact = json.loads(Path(manifest['artifact']).read_text())
        injections = json.loads(path.with_name('legacy_injections.json').read_text())
        assert [r['time'] for r in injections] == list(range(0,3600,600))
        assert [r['scheduled'] for r in injections] == [sum(v['legacy_block']==i for v in artifact['vehicles']) for i in range(6)]
        env = json.loads(path.with_name('environment.json').read_text())
        old_lanes = [lane for lanes in env['node_lanes'].values() for lane in lanes]
        assert len(old_lanes) == len(set(old_lanes)) == 116
        assert set(old_lanes) == set(env['lanes'])
        with np.load(path.with_name('rollout.npz'), allow_pickle=False) as data:
            series = data['queue'].sum(axis=1)
        curves.setdefault((manifest['controller'],manifest['method']),[]).append(series)
    accepted.update(protocol='legacy_partial_demand_replay', legacy_block_injections_verified=200,
                    legacy_monitored_lane_equivalence_verified=200)
    acceptance = directory/'acceptance'
    acceptance.mkdir(exist_ok=False)
    write_json(acceptance/'monaco.json', accepted)
    analysis = directory/'analysis/monaco'
    analyze(campaign,'monaco',analysis)
    subprocess.run([sys.executable, str(ROOT/'docs/evaluation_workbook/grid_results_site/export_network_data.py'),
                    '--network','monaco','--campaign',str(campaign),
                    '--evaluation-set','monaco_legacy_replay'],check=True)
    before = json.loads((ROOT/'reports/wce_analysis_20261006/main_site_before_external.json').read_text())
    changed = [p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed, 'Main results changed: '+str(changed)
    statistics = json.loads((analysis/'paired_statistics.json').read_text())
    with (analysis/'queue_timeseries_mean_sd.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        names = sorted(curves)
        writer.writerow(['time_sec'] + [c+'__'+m+'__'+s for c,m in names for s in ('mean','sd')])
        stacked = {key: np.stack(curves[key]) for key in names}
        for t in range(3600):
            writer.writerow([t+1] + [value for key in names for value in
                                      (float(stacked[key][:,t].mean()),float(stacked[key][:,t].std(ddof=1)))])
    audits = [json.loads(p.read_text()) for p in sorted((directory/'artifacts/monaco').glob('legacy_generation_*.json'))]
    lines = ['日期：2026-10-06','Monaco 旧需求流程复测结果','',
             '完成：200/200 正式评估 + 4 次完整预检查。4 controllers × 5 methods × 10 rollouts，每次3600秒。',
             '结论：旧代码可以运行该 CSV，因为不可达 OD 被静默跳过；它没有运行完整14个OD。',
             '位置：envs/real_net_env.py 的 _inject_scenario_traffic；先抽 Poisson 数量，findRoute失败/空路径时continue。',
             '原始评估入口：eval_signal_controllers_real.py；同一需求CSV重复6次，每块600秒，默认10次评估。',
             '本次直接调用原需求生成函数并记录每次抽样；实际SUMO按600秒块注入，未删除CSV行、改路网或补路。',
             '不可达 OD：-10051#2 → 10043，108.333333 veh/h；源需求2383.333332，路由可达需求2274.999999 veh/h（少4.545455%）。',
             '历史模型的8个指定目录只有.meta，均缺.index与.data-*，无法恢复原模型；详见legacy_model_inventory.json。',
             '使用当前seed101的20个最终模型，2320000学习步；完整checkpoint哈希、网络/输入哈希、3600连续样本均验收。',
             '部分历史parent来源缺失仍单独记录；本次验证现有最终模型，未声称补齐历史血缘证明。',
             '与历史精确复现的差异：当前模型/执行器、空网络解析并缓存路线、共享配对需求和独立策略RNG。',
             '历史生成器首次解析OD时使用实时网络，需求与策略共享全局numpy RNG；旧文件没有给全局numpy RNG显式设定评估种子。',
             'SUMO种子61001–61010，需求种子51001–51010；未假称使用历史10000–100000种子及其随机轨迹。',
             '旧策略默认A2C/PPO采样，IQL贪心；本次使用当前冻结策略评估规则。',
             '排队口径：116条受控入口lane逐秒halting总和；已验证与旧_measure_traffic_step的lane集合相同且没有重复。',
             '保留旧出发时间抖动，不对块尾/3600秒做上界裁剪，3份输入各有1辆计划在3600秒或之后出发。',
             '此结果是部分需求的旧流程复测，不能标为完整14OD外部需求已通过。',
             '网站独立新增Monaco旧流程复测集合；原主实验929个数据文件哈希未变，严格Group12的Monaco状态仍待验证。',
             '网站：http://127.0.0.1:8878/?evaluationSet=monaco_legacy_replay&network=monaco&split=external',
             '未更改原Grid结果、主实验或论文。','',
             '需求种子 | 抽样总数 | 不可达OD跳过 | 实际计划车辆 | 3600秒及之后出发']
    for audit in audits:
        lines.append('%d | %d | %d | %d | %d' % (audit['seed'],audit['sampled'],audit['skipped'],audit['scheduled'],audit['depart_at_or_after_horizon']))
    lines += ['', '平均排队（辆，越低越好）：controller | method | mean | sd | completed | pending | teleports']
    for row in statistics['summary']:
        lines.append('%s | %s | %.3f | %.3f | %.1f | %.1f | %.1f' %
                     (row['controller'],row['method'],row['mean_queue']['mean'],row['mean_queue']['sd'],
                      row['completed']['mean'],row['pending']['mean'],row['teleports']['mean']))
    lines += ['', 'Online WCE 相对 baseline：controller | 平均逐次收益% | wins/10 | Holm p']
    for row in statistics['comparisons']:
        if row['method']=='online_wce' and row['reference']=='baseline':
            lines.append('%s | %.3f%% | %d/10 | %.6f' %
                         (row['controller'],row['mean_percent_reduction'],row['wins'],row['holm_adjusted_p']))
    lines += ['', '统计：50000次配对bootstrap探索性95%区间；双侧精确sign-flip检验，对20项比较做Holm校正。',
              '推断仅针对单个训练seed101下的评估随机性。所有冻结学习步保持2320000，WCE更新为0。',
              'Campaign：'+str(campaign.resolve()), '验收：'+str((acceptance/'monaco.json').resolve()),
              '详细统计：'+str((analysis/'paired_statistics.json').resolve()),
              '曲线CSV：'+str((analysis/'queue_timeseries_mean_sd.csv').resolve())]
    report = ROOT/'reports/wce_analysis_20261006/monaco_legacy_execution_report.txt'
    with report.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',rollouts=200,report=str(report)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    finalize(a.campaign.resolve(),a.wait)
