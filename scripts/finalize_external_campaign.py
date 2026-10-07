#!/usr/bin/env python3
"""Accept, analyze and export a complete supplementary campaign exclusively."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_external_results import validate


def finalize(campaign, network, wait):
    campaign = campaign.resolve()
    started = time.monotonic()
    result_root = campaign.parent / network
    while True:
        paths = list(result_root.glob('*/*/external/*/rollout_*/attempt_*/result.json'))
        complete = sum(json.loads(p.read_text()).get('status') == 'complete' for p in paths)
        if complete >= 200:
            break
        if not wait or time.monotonic() - started > 3600:
            raise RuntimeError('Matrix incomplete: %d/200' % complete)
        print(json.dumps({'status':'waiting','completed':complete,'expected':200}), flush=True)
        time.sleep(30)
    acceptance = validate(campaign, network)
    report_root = campaign.parent / 'acceptance'
    report_root.mkdir(exist_ok=True)
    acceptance['tools'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT/'scripts/validate_external_results.py',
                                     ROOT/'scripts/analyze_external_results.py',
                                     ROOT/'docs/evaluation_workbook/grid_results_site/export_network_data.py']}
    with (report_root / (network + '.json')).open('x') as f:
        json.dump(acceptance, f, indent=2)
    analysis = campaign.parent / 'analysis' / network
    subprocess.check_call([sys.executable, str(ROOT/'scripts/analyze_external_results.py'), '--campaign', str(campaign),
                           '--network', network, '--output', str(analysis)], cwd=str(ROOT))
    subprocess.check_call([sys.executable, str(ROOT/'docs/evaluation_workbook/grid_results_site/export_network_data.py'),
                           '--campaign', str(campaign), '--network', network], cwd=str(ROOT))
    before = json.loads((ROOT/'reports/wce_analysis_20261006/main_site_before_external.json').read_text())
    changed = [p for p, h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    if changed:
        raise RuntimeError('Main data changed: ' + str(changed))
    statistics = json.loads((analysis/'paired_statistics.json').read_text())
    lines = ['日期：2026-10-06', '外部需求测试执行报告', '',
             'Grid：200/200 正式评估完成，4 controllers × 5 methods × 10 rollouts。',
             '另有 4 次独立的完整 controller 冒烟评估。所有正式结果通过 3600 秒连续样本、NPZ 维度、需求/SUMO/policy seed 配对、来源/网络/checkpoint 哈希与冻结学习步数验收。',
             '输入：本地 Grid sparse external candidate；原生 2983 veh/h，140 个正需求 OD。十个共享 artifact 已通过实际 SUMO 路线加载。上游 Hangzhou 来源及映射未验证。',
             'Monaco：未启动正式 campaign。-10051#2 → 10043 的 108.333333 veh/h 不可达，涉及四条源 flow，当前路网缺少 10180#0 和 10180#1。两段路径均包含受控路口，不能删掉前段声称等价。另有历史来源 manifest 缺失。',
             '主网站原有 929 个数据文件 SHA-256 均未改变。主实验仍为每网络 4600 条，共 9200 条；Grid 补充结果 200 条独立展示。',
             '仅 seed 101；置信区间针对评估随机性，不能推断跨训练种子稳定性。未启动新训练或 WCE 更新。', '',
             '平均排队（mean_queue，越低越好）：', 'controller | method | mean | sd | completed | pending | teleports | collisions']
    for row in statistics['summary']:
        def mean(metric):
            return 'NA' if row[metric] is None else '%.3f' % row[metric]['mean']
        lines.append('%s | %s | %s | %.3f | %s | %s | %s | %s' %
                     (row['controller'],row['method'],mean('mean_queue'),row['mean_queue']['sd'],
                      mean('completed'),mean('pending'),mean('teleports'),mean('collisions')))
    lines += ['', 'Online WCE 相对 baseline：controller | 平均逐次收益% | wins/10 | Holm p']
    for row in statistics['comparisons']:
        if row['method']=='online_wce' and row['reference']=='baseline':
            lines.append('%s | %.3f%% | %d/10 | %.6f' %
                         (row['controller'],row['mean_percent_reduction'],row['wins'],row['holm_adjusted_p']))
    lines += ['', '配对置信区间使用 50,000 次配对 bootstrap，95% percentile，未做同时覆盖校正，作为探索性区间。',
              '正式 p 值使用双侧精确配对 sign-flip 检验（1024 个符号组合），对 20 个预设比较做 Holm 校正；要求零假设下差异对称。',
              '详细置信区间与所有指标：'+str(analysis/'paired_statistics.json'),
              '验收：'+str(report_root/(network+'.json')), 'Campaign：'+str(campaign),
              '网站：http://127.0.0.1:8878/?evaluationSet=external_group12&network=grid']
    report = ROOT / 'reports/wce_analysis_20261006/external_execution_report.txt'
    with report.open('x') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','report':str(report),'main_data_changed':changed}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--network',choices=['grid','monaco'],required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args();finalize(a.campaign,a.network,a.wait)
