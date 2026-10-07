"""Verify, summarize and document the full-demand repaired Monaco frozen-policy evaluation."""
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
            raise RuntimeError('Incomplete repaired Monaco campaign: %d/200' % completed)
        time.sleep(10)
    metadata = json.loads(campaign.read_text())
    assert file_hash(ROOT/'scripts/repaired_monaco_campaign.py') == metadata['scheduler_hash']
    assert file_hash(ROOT/'scripts/repaired_monaco_evaluate.py') == metadata['evaluator_hash']
    accepted = validate(campaign, 'monaco')
    assert accepted['repaired_transfer_verified'] == 200
    curves = {}
    for name in accepted['summaries']:
        path = Path(name)
        manifest = json.loads(path.with_name('manifest.json').read_text())
        with np.load(path.with_name('rollout.npz'), allow_pickle=False) as data:
            series = data['queue'].sum(axis=1)
        curves.setdefault((manifest['controller'],manifest['method']),[]).append(series)
    acceptance = directory/'acceptance'
    acceptance.mkdir(exist_ok=False)
    write_json(acceptance/'monaco.json', accepted)
    analysis = directory/'analysis/monaco'
    analyze(campaign,'monaco',analysis)
    subprocess.run([sys.executable, str(ROOT/'docs/evaluation_workbook/grid_results_site/export_network_data.py'),
                    '--network','monaco','--campaign',str(campaign),
                    '--evaluation-set','monaco_repaired_full14'],check=True)
    before = json.loads((directory/'site_before.json').read_text())
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
    audits = [json.loads(Path(a['path']).read_text())['generation_audit'] for a in metadata['networks']['monaco']['artifacts']]
    lines = ['日期：2026-10-06', 'Monaco 修复地图完整需求冻结策略评估', '',
             '完成：200/200 正式评估 + 4 次3600秒完整预检查。4 controllers × 5 methods × 10 rollouts。',
             '完整14/14 OD，原始2383.333332 veh/h需求；六个600秒块直接调用原需求生成函数；没有跳过OD或抽样车辆。',
             '恢复原MoST地图遗漏道路10180#0、10180#1及优先路口9601；28个受控路口、116条入口车道与264个信号连接编号保留。',
             'netconvert重算几何，43个检测器位置/覆盖调整；这属于修复地图迁移测试，不是原地图测试或新训练结果。',
             '先在真实原地图严格加载20个当前最终checkpoint，再核验观测/动作/车道顺序/邻接关系，冻结推理于实际修复地图。',
             '逐次记录原训练签名与实际目标签名、权重前后SHA256；200次均权重不变，学习步2320000，WCE更新0。',
             '训练seed101；需求seed51001–51010，SUMO seed61001–61010；方法共享配对需求与SUMO随机种子，策略使用独立RNG。',
             '保留旧生成器出发时间抖动，包括3600秒及以后出发；未声称精确重现历史共享随机数轨迹。',
             '仅验证当前20个最终模型；部分历史训练父模型缺失，上游原始地图来源版本仍未完全核实。',
             '排队：116条受控入口lane逐秒halting总和。速度：车辆秒加权平均，与旧速度等时间平均口径不同。',
             '网站独立集合新增；已有主实验、Grid外部评估和旧Monaco部分需求复测数据全部哈希未变。',
             '网站：http://127.0.0.1:8878/?evaluationSet=monaco_repaired_full14&network=monaco&split=external', '',
             '需求种子 | 抽样总数 | 跳过 | 计划车辆 | 3600秒及之后出发']
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
    report = ROOT/'reports/wce_analysis_20261006/monaco_repaired_execution_report.txt'
    with report.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    subprocess.run(['/home/sdc_joran/miniconda3/envs/deeprlsc/bin/python',str(ROOT/'scripts/export_monaco_legacy_metrics.py'),str(campaign)],check=True)
    print(json.dumps(dict(status='complete',rollouts=200,report=str(report)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    finalize(a.campaign.resolve(),a.wait)
