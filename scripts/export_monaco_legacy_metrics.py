"""Use the original evaluator's raw-CSV and summary functions on complete replays."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval_signal_controllers_real import RolloutResult, summarize_group, save_raw_timeseries
from experiments.core import file_hash, write_json


def export(campaign):
    root=campaign.parent
    accepted=json.loads((root/'acceptance/monaco.json').read_text())
    assert accepted['rollouts']==200 and accepted['status']=='passed'
    groups={}
    for name in accepted['summaries']:
        path=Path(name)
        manifest=json.loads(path.with_name('manifest.json').read_text())
        summary=json.loads(path.read_text())
        rows=[json.loads(s) for s in path.with_name('rollout.jsonl').read_text().splitlines()]
        assert [r['time'] for r in rows]==list(range(1,3601))
        queue=[r['queue'] for r in rows]
        # Exactly the old _measure_traffic_step speed convention, including zero
        # for an empty network, then the old unweighted time average.
        speed=[r['speed_sum']/r['active'] if r['active'] else 0. for r in rows]
        assert np.isclose(np.mean(queue),summary['mean_queue'])
        groups.setdefault((manifest['controller'],manifest['method']),[]).append(
            (manifest['sumo_seed'],queue,speed,summary['mean_speed']))
    out=root/'analysis/monaco/legacy_metrics'
    out.mkdir(exist_ok=False)
    summaries=[]
    for (controller,method),records in sorted(groups.items()):
        records=sorted(records)
        assert len(records)==10
        result=RolloutResult(queue_ts=[r[1] for r in records],speed_ts=[r[2] for r in records])
        stats=summarize_group(result)
        save_raw_timeseries(str(out),controller+'__'+method,12,'Real_Life_Monaco',result)
        summaries.append(dict(controller=controller,method=method,n=10,
                              legacy_queue_overall_mean=stats.queue_overall_mean,
                              legacy_speed_overall_mean=stats.speed_overall_mean,
                              current_vehicle_weighted_speed_mean=float(np.mean([r[3] for r in records]))))
    with (out/'legacy_summary_averages.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    write_json(out/'protocol.json',dict(status='passed',groups=20,rollouts=200,seconds=3600,
               original_evaluator_hash=file_hash(ROOT/'eval_signal_controllers_real.py'),
               original_functions=['summarize_group','save_raw_timeseries'],
               speed_metric='Old per-second mean vehicle speed, zero when no vehicles, equally averaged over 3600 seconds',
               current_site_speed_metric='Vehicle-seconds weighted speed; separate metric, not equal to the legacy average',
               queue_metric='Same 116-lane total halting count and same per-second time average',
               incomplete_padding='Not used; every series required to contain 3600 samples'))
    print(json.dumps(dict(status='complete',raw_csv_files=40,summary_csv=str(out/'legacy_summary_averages.csv'))))


if __name__=='__main__':
    export(Path(sys.argv[1]).resolve())
