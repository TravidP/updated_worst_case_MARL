"""Bounded-memory completed-trip statistics; unfinished/removed trips are separate."""
import math
import xml.etree.ElementTree as ET

FIELDS = dict(travel_time='duration', waiting_time='waitingTime',
              time_loss='timeLoss', departure_delay='departDelay')


def summarize_trips(path):
    sums = {name: 0. for name in FIELDS}
    count = removed = unfinished = arrived_records = 0
    context = ET.iterparse(path, events=('start', 'end'))
    _, root = next(context)
    for event, element in context:
        if event != 'end' or element.tag != 'tripinfo':continue
        arrival = float(element.attrib['arrival'])
        if not math.isfinite(arrival):raise ValueError('Invalid tripinfo arrival')
        arrived_records += int(arrival >= 0)
        if element.get('vaporized', '').lower() not in ('', 'false', '0'):
            removed += 1
        elif arrival < 0:
            unfinished += 1
        else:
            for name, attribute in FIELDS.items():
                value = float(element.attrib[attribute])
                if not math.isfinite(value) or value < 0:raise ValueError('Invalid tripinfo '+attribute)
                sums[name] += value
            count += 1
        root.clear()
    return dict(count=count, excluded=removed+unfinished, removed=removed,
                unfinished=unfinished, arrived_records=arrived_records, sums=sums,
                means={name: value/count if count else None for name, value in sums.items()})


def attach_trip_metrics(result, trips):
    """SUMO arrivals may include teleport removals; keep physical completions separate."""
    if trips['arrived_records'] != result['completed']:
        raise ValueError('Tripinfo arrival records differ from SUMO arrived vehicles')
    result['arrived'] = result['completed']
    result['completed'] = trips['count']
    result['removed'] = trips['removed']
    result['completion_rate'] = trips['count']/result['inserted'] if result['inserted'] else None
    result['trips'] = trips
