"""Restore omitted original-map roads using supported SUMO PlainXML patches.

Creates a new version; never replaces checkpoint-bound training assets.
Requires the TF1 environment's sumolib package (but does not load any model).
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.core import file_hash, write_json

BASE = ROOT/'real_net_subnet/data/in/most.net.xml'
SOURCE = ROOT/'real_net/data/in/most.net.xml'
RESTORE = {'10180#0','10180#1'}
NODES = {'9963','9601','8746'}


def run(command, log):
    with log.open('x') as stream:
        subprocess.check_call(command, stdout=stream, stderr=subprocess.STDOUT)


def road_elements(root):
    return {e.get('id'):e for e in root.findall('edge') if e.get('function')!='internal'}


def tls(root):
    return {t.get('id'):(dict(t.attrib), [dict(p.attrib) for p in t]) for t in root.findall('tlLogic')}


def controlled_connections(root):
    keys=['from','to','fromLane','toLane','tl','linkIndex']
    return sorted(tuple(c.get(k) for k in keys) for c in root.findall('connection') if c.get('tl'))


def repair(out):
    import sumolib
    out.mkdir(parents=True, exist_ok=False)
    data=out/'data/in';data.mkdir(parents=True)
    patches=out/'patches';patches.mkdir()
    export=out/'source_export';export.mkdir()
    source=ET.parse(str(SOURCE)).getroot();base=ET.parse(str(BASE)).getroot()
    original=road_elements(source);existing=road_elements(base)
    assert not RESTORE.intersection(existing) and RESTORE.issubset(original)
    source_nodes={n.get('id'):n for n in source.findall('junction')}
    prefix=export/'original'
    commands=[]
    command=['netconvert','--sumo-net-file',str(SOURCE),'--plain-output-prefix',str(prefix),
             '--plain-output.lanes','true','--offset.disable-normalization','true',
             '--output-file',str(export/'roundtrip.net.xml')]
    commands.append(command);run(command,export/'conversion.log')
    for tag,extension,ids in [('nodes','nod',NODES),('edges','edg',RESTORE)]:
        raw=ET.parse(str(prefix)+'.'+extension+'.xml').getroot()
        patch=ET.Element(tag)
        for item in raw:
            if item.get('id') in ids:patch.append(copy.deepcopy(item))
        assert len(patch)==len(ids)
        ET.ElementTree(patch).write(str(patches/('restore.'+extension+'.xml')))
    keep=set(existing)|RESTORE
    raw=ET.parse(str(prefix)+'.con.xml').getroot();patch=ET.Element('connections')
    for connection in raw:
        edge=original.get(connection.get('from'))
        if edge is not None and edge.get('to') in NODES and connection.get('from') in keep and connection.get('to') in keep:
            assert not connection.get('tl'), 'Repair must not introduce signal-control changes'
            patch.append(copy.deepcopy(connection))
    assert len(patch)==3
    ET.ElementTree(patch).write(str(patches/'restore.con.xml'))
    network=data/'most.net.xml'
    command=['netconvert','--sumo-net-file',str(BASE),'--node-files',str(patches/'restore.nod.xml'),
             '--edge-files',str(patches/'restore.edg.xml'),'--connection-files',str(patches/'restore.con.xml'),
             '--offset.disable-normalization','true','--geometry.min-radius.fix.railways','false',
             '--geometry.max-grade.fix','false','--output-file',str(network)]
    commands.append(command);run(command,out/'repair.log')
    fixed=ET.parse(str(network)).getroot();fixed_roads=road_elements(fixed)
    assert set(fixed_roads)==keep and len(keep)==272
    assert tls(base)==tls(fixed) and len(tls(fixed))==28
    assert controlled_connections(base)==controlled_connections(fixed)
    additional=ROOT/'real_net_subnet/data/in/most.add.xml'
    monitored={d.get('lane') for d in ET.parse(str(additional)).getroot() if d.get('lane')}
    def lanes(root):
        return {l.get('id'):dict(l.attrib) for e in road_elements(root).values() for l in e.findall('lane')}
    before_lanes,after_lanes=lanes(base),lanes(fixed)
    assert len(monitored)==116 and monitored.issubset(after_lanes)
    detector_tree=ET.parse(str(additional));detector_changes=[]
    for detector in detector_tree.getroot():
        lane=detector.get('lane')
        if not lane:continue
        old_length=float(before_lanes[lane]['length']);new_length=float(after_lanes[lane]['length'])
        old_start=float(detector.get('pos'));old_end=float(detector.get('endPos'))
        setback=old_length-old_end
        new_end=new_length-setback
        if new_end<=0:new_end=new_length*0.9
        new_start=max(0.,new_end-(old_end-old_start))
        assert 0<=new_start<new_end<=new_length
        if abs(new_start-old_start)>1e-6 or abs(new_end-old_end)>1e-6:
            detector_changes.append(dict(id=detector.get('id'),lane=lane,before=dict(pos=old_start,endPos=old_end),
                                         after=dict(pos=new_start,endPos=new_end)))
        detector.set('pos','%.4f'%new_start);detector.set('endPos','%.4f'%new_end)
        detector.set('file','/dev/null')
    detector_tree.write(str(data/'most.add.xml'))
    write_json(out/'detector_adjustments.json',detector_changes)
    changes=[dict(lane=k,changed_attributes={a:dict(before=v.get(a),after=after_lanes[k].get(a))
                 for a in set(v)|set(after_lanes[k]) if v.get(a)!=after_lanes[k].get(a)})
             for k,v in sorted(before_lanes.items()) if v!=after_lanes[k]]
    write_json(out/'geometry_changes.json',changes)
    demand=ROOT/'real_net_subnet/demand_groups/Real_Life_Monaco.csv'
    (out/'Real_Life_Monaco.csv').write_bytes(demand.read_bytes())
    (data/'keep_edges.txt').write_text('\n'.join(sorted(keep))+'\n')
    net=sumolib.net.readNet(str(network))
    route_xml=ET.Element('routes')
    ET.SubElement(route_xml,'vType',id='type1',vClass='passenger',length='5',accel='5',decel='10',speedDev='0')
    od_routes=[]
    def probe_vehicle(vid,edges,depart):
        vehicle=ET.SubElement(route_xml,'vehicle',id=vid,type='type1',depart=str(depart))
        ET.SubElement(vehicle,'route',edges=' '.join(edges))
    with demand.open() as stream:
        rows=list(csv.DictReader(stream))
    for index,row in enumerate(rows):
        o,d=row['origin_edge'],row['dest_edge']
        path,cost=net.getShortestPath(net.getEdge(o),net.getEdge(d),vClass='passenger')
        assert path,'Unreachable repaired OD: '+o+' -> '+d
        edges=[e.getID() for e in path]
        assert edges[0]==o and edges[-1]==d
        od_routes.append(dict(origin=o,destination=d,rate=float(row['veh_per_hour']),edges=edges))
        probe_vehicle('od_%02d'%index,edges,index)
    assert len(rows)==14
    reconstructed=[]
    audit=ROOT/'real_net_subnet/demand_groups/Real_Life_Monaco_route_audit.csv'
    with audit.open() as stream:
        for index,row in enumerate(csv.DictReader(stream)):
            full=row['full_route'].split();blocks=[];block=[]
            for edge in full:
                if edge in keep:block.append(edge)
                elif block:blocks.append(block);block=[]
            if block:blocks.append(block)
            assert len(blocks)==1,'Source flow leaves and re-enters: '+row['flow_id']
            edges=blocks[0]
            assert edges[0]==row['origin_edge'] and edges[-1]==row['dest_edge']
            for a,b in zip(edges,edges[1:]):
                assert net.getEdge(b) in net.getEdge(a).getOutgoing(), 'Disconnected original flow'
            projected=dict(row,projected_subnet_route=' '.join(edges),block_count=1,status='repaired_topology_verified')
            reconstructed.append(projected)
            probe_vehicle('flow_'+row['flow_id'],edges,len(rows)+index)
    assert len(reconstructed)==88
    with (out/'Real_Life_Monaco_route_audit.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,list(reconstructed[0]));writer.writeheader();writer.writerows(reconstructed)
    ET.ElementTree(route_xml).write(str(data/'connectivity_probe.rou.xml'))
    cfg=ET.Element('configuration');inp=ET.SubElement(cfg,'input')
    ET.SubElement(inp,'net-file',value='in/most.net.xml')
    ET.SubElement(inp,'additional-files',value='in/most.add.xml')
    ET.SubElement(inp,'route-files',value='in/connectivity_probe.rou.xml')
    ET.SubElement(ET.SubElement(cfg,'time'),'end',value='3600')
    ET.ElementTree(cfg).write(str(out/'data/connectivity_probe.sumocfg'))
    command=['sumo','-c',str(out/'data/connectivity_probe.sumocfg'),'--seed','71001',
             '--no-step-log','true','--duration-log.disable','true','--time-to-teleport','300',
             '--tripinfo-output',str(out/'probe_tripinfo.xml'),
             '--error-log',str(out/'probe_errors.log')]
    commands.append(command);run(command,out/'probe.log')
    errors=(out/'probe_errors.log').read_text()
    assert 'Error:' not in errors
    trips=ET.parse(str(out/'probe_tripinfo.xml')).getroot().findall('tripinfo')
    completed={t.get('id') for t in trips if float(t.get('arrival','-1'))>=0}
    expected={'od_%02d'%i for i in range(14)}|{'flow_'+r['flow_id'] for r in reconstructed}
    assert completed==expected,'Incomplete physical route probe: '+str(expected-completed)
    # Reconstruct the OD CSV from all original flow contributions, not by editing a failed row.
    totals={}
    for row in reconstructed:
        key=(row['origin_edge'],row['dest_edge'])
        totals[key]=totals.get(key,0)+float(row['contribution_veh_per_hour'])
    assert set(totals)=={(r['origin_edge'],r['dest_edge']) for r in rows}
    assert all(abs(totals[(r['origin_edge'],r['dest_edge'])]-float(r['veh_per_hour']))<1e-6 for r in rows)
    node_changes=[]
    base_nodes={n.get('id'):n for n in base.findall('junction')}
    for name in sorted(NODES):
        node_changes.append(dict(id=name,old_type=base_nodes[name].get('type') if name in base_nodes else None,
                                 original_type=source_nodes[name].get('type'),
                                 repaired_type=next(n.get('type') for n in fixed.findall('junction') if n.get('id')==name)))
    manifest=dict(status='passed',repair_version=out.name,restored_edges=sorted(RESTORE),junction_changes=node_changes,
                  source_map=str(SOURCE),source_map_hash=file_hash(SOURCE),base_network_hash=file_hash(BASE),
                  repaired_network_hash=file_hash(network),original_demand_hash=file_hash(demand),
                  original_additional_hash=file_hash(additional),repaired_additional_hash=file_hash(data/'most.add.xml'),
                  adjusted_detector_positions=len(detector_changes),
                  demand_unchanged=file_hash(demand)==file_hash(out/'Real_Life_Monaco.csv'),
                  source_flows=88,positive_od=14,total_veh_per_hour=sum(float(r['veh_per_hour']) for r in rows),
                  passenger_routes=od_routes,physical_probe_scheduled=102,physical_probe_completed=len(completed),
                  tls_programs_unchanged=28,controlled_link_mappings_unchanged=264,monitored_lanes_unchanged=116,
                  existing_lane_attribute_changes=len(changes),
                  monitored_lane_length_changes=sum(before_lanes[l]['length']!=after_lanes[l]['length'] for l in monitored),
                  tools=dict(netconvert=subprocess.check_output(['netconvert','--version']).decode().splitlines()[0]),
                  commands=commands,upstream_reference='https://github.com/lcodeca/MoSTScenario',
                  provenance_scope='Restored from the exact local 2018 source map; upstream byte-for-byte equivalence unverified.',
                  model_compatibility='New network hash. Existing final checkpoints remain bound to the old network; no hash bypass or policy evaluation performed.',
                  geometry_note='Current netconvert recomputes some existing lane and junction geometry. See geometry_changes.json. This is a new topology version, not an equivalent original-network evaluation.')
    write_json(out/'repair_manifest.json',manifest)
    print(json.dumps(dict(status='passed',output=str(out),restored_edges=sorted(RESTORE),positive_od=14,
                         original_flows=88,completed_probe_vehicles=len(completed)),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repair(a.output.resolve())
