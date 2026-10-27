import contextlib
import tempfile
import unittest
from pathlib import Path
from experiments.checkpoint import Checkpoints,inspect_checkpoint

class Saver:
    def save(self,sess,path,**kwargs):
        Path(path+'.index').write_text('fixture')
        Path(path+'.data').write_bytes(b'parameters')

class Graph:
    def as_default(self):return contextlib.nullcontext()

class Model:
    graph=Graph();sess=None;saver=Saver();inference_saver=Saver()
    signature={'kind':'fixture'};training=True
    def state(self):return {'counter':1}

class StorageTests(unittest.TestCase):
    def test_retention_protects_roles_pins_and_recovery_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            store=Checkpoints(d)
            for i in range(8):
                name='cp%d'%i
                store.publish(name,{'controller':Model()},{},'full')
                store.update(name=name,latest=True)
                if i==0:store.pin(name)
                if i==1:store.update('best',name)
                if i==2:store.update('best_at_common_budget',name)
            removed=store.collect()
            self.assertEqual(set(removed),{'cp3','cp4','cp5'})
            for name in ['cp0','cp1','cp2','cp6','cp7']:inspect_checkpoint(store.path/name)
            # Recovery refers to an older best snapshot even after current best changes.
            store.publish('snap',{'controller':Model()},{},'inference')
            store.publish('new',{'controller':Model()},{'selection':{'history':[], 'best':{'checkpoint':'snap'}}})
            store.update(name='new',latest=True);store.collect()
            self.assertTrue((store.path/'snap').exists())

    def test_failed_publish_preserves_previous_recovery(self):
        class Broken(Saver):
            def save(self,*args,**kwargs):raise OSError('simulated transfer failure')
        with tempfile.TemporaryDirectory() as d:
            store=Checkpoints(d);store.publish('valid',{'controller':Model()},{})
            store.update(name='valid',latest=True)
            broken=Model();broken.saver=Broken()
            with self.assertRaises(OSError):store.publish('failed',{'controller':broken},{})
            self.assertEqual(store.role('latest').name,'valid');inspect_checkpoint(store.role('latest'))
            self.assertFalse((store.path/'failed').exists())

if __name__=='__main__':unittest.main()
