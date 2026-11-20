import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.configuration import load_controller_config
from experiments.core import QueueMetric
from agents.controller import Controller
from agents.dql import Replay
from experiments.checkpoint import save_checkpoint,load_checkpoint,inspect_checkpoint

class FakeEnv:
    def __init__(self,family):
        self.network='grid';self.config=load_controller_config('grid',family)[0]
        self.n_s_ls=[4,4];self.n_a_ls=[2,3];self.n_w_ls=[0,0];self.n_f_ls=[0,0]
        self.asset_hashes={'network':'fixture'};self.metric=QueueMetric({'a':['x'],'b':['y']},{'a':['b'],'b':['a']})
        c=self.config['MODEL_CONFIG']
        if family=='dql':
            c['buffer_size']='32';c['learning_starts']='4';c['batch_size']='4';c['update_interval']='1';c['target_interval']='10'
        else:c['batch_size']='4'
    def update_fingerprint(self,probabilities):pass

class LearningTests(unittest.TestCase):
    def test_continuation_inherits_weights_and_fresh_optimizer(self):
        from experiments.runner import initialize_continuation
        for family in ('ma2c','ppo','dql'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as folder:
                env=FakeEnv(family)
                parent=Controller(env,family,101)
                child=Controller(env,family,101)
                try:
                    obs=[np.ones(4),np.ones(4)]
                    for step in range(6):
                        action=parent.act(obs,env,True)
                        parent.observe(obs,action,[-2,-3],obs,step==5,True)
                    path=Path(folder)/'parent'
                    save_checkpoint(path,{'controller':parent},{'episode':1})
                    weights=child.online if family=='dql' else child.trainable
                    learned=parent.online if family=='dql' else parent.trainable
                    inherited={v.name for v in weights}
                    if family=='dql':
                        inherited.update(v.name for v in child.target)
                    other=[v for v in child.variables if v.name not in inherited]
                    before=child.sess.run(other)
                    initialize_continuation(child,path)
                    for x,y in zip(parent.sess.run(learned),child.sess.run(weights)):
                        np.testing.assert_array_equal(x,y)
                    for x,y in zip(before,child.sess.run(other)):
                        np.testing.assert_array_equal(x,y)
                    self.assertEqual((child.learning_steps,child.scheduler_steps),(0,0))
                    if family=='dql':
                        self.assertEqual((child.replay.size,child.epsilon),(0,1.0))
                        for x,y in zip(child.sess.run(child.online),child.sess.run(child.target)):
                            np.testing.assert_array_equal(x,y)
                finally:
                    parent.close()
                    child.close()

    def test_replay_bound_and_chunk_roundtrip(self):
        r=Replay(8,[3,2]);size=r.nbytes
        for i in range(30):r.add([[i]*3,[i]*2],[0,1],[1,2],[[i+1]*3,[i+1]*2],False)
        self.assertEqual(r.size,8);self.assertEqual(r.nbytes,size)
        with tempfile.TemporaryDirectory() as d:
            r.save(Path(d)/'replay');other=Replay(8,[3,2]);other.load(Path(d)/'replay')
            self.assertEqual(other.cursor,r.cursor)
            for a,b in zip(r.arrays,other.arrays):
                for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_target_and_exact_next_update_recovery(self):
        env=FakeEnv('dql');m=Controller(env,'dql',9001);obs=[np.ones(4),np.ones(4)]
        try:
            initial=m.sess.run(m.target)
            for _ in range(6):
                decision=m.act(obs,env,True);m.observe(obs,decision,[-2,-3],obs,False,True)
            for a,b in zip(initial,m.sess.run(m.target)):np.testing.assert_array_equal(a,b)
            self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(initial,m.sess.run(m.online))))
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'full';save_checkpoint(path,{'controller':m},{'episode':1})
                nxt=Controller(env,'dql',12)
                try:
                    load_checkpoint(path,{'controller':nxt})
                    for _ in range(4):
                        a=m.act(obs,env,True);b=nxt.act(obs,env,True);self.assertEqual(a,b)
                        m.observe(obs,a,[-2,-3],obs,False,True);nxt.observe(obs,b,[-2,-3],obs,False,True)
                    for a,b in zip(m.sess.run(m.variables),nxt.sess.run(nxt.variables)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
                    for a,b in zip(m.sess.run(m.online),m.sess.run(m.target)):np.testing.assert_array_equal(a,b)
                finally:nxt.close()
                save_checkpoint(Path(d)/'light',{'controller':m},{},'inference')
                frozen=Controller(env,'dql',9001,training=False)
                try:
                    load_checkpoint(Path(d)/'light',{'controller':frozen},inference=True)
                    self.assertIsNone(frozen.replay);self.assertEqual(frozen.target,[])
                    self.assertFalse(any('Adam' in v.name for v in frozen.variables))
                    with self.assertRaises(ValueError):load_checkpoint(Path(d)/'light',{'controller':m})
                finally:frozen.close()
                (path/'runner.pkl').write_bytes(b'broken')
                with self.assertRaises(ValueError):inspect_checkpoint(path)
        finally:m.close()

    def test_actor_critic_masked_update_and_frozen_graph(self):
        for family in ['ma2c','ppo']:
            env=FakeEnv(family);m=Controller(env,family,9001);obs=[np.ones(4),np.ones(4)]
            try:
                for t in range(3):
                    decision=m.act(obs,env,True);m.observe(obs,decision,[-2,-3],obs,t==2,True)
                self.assertEqual(m.batch_histogram,{3:1});self.assertEqual(m.learning_steps,3)
                self.assertTrue(all(np.isfinite(row['advantage_std']) for row in m.diagnostics))
                with tempfile.TemporaryDirectory() as d:
                    save_checkpoint(Path(d)/'light',{'controller':m},{},'inference')
                    frozen=Controller(env,family,9001,training=False)
                    try:
                        load_checkpoint(Path(d)/'light',{'controller':frozen},inference=True)
                        self.assertFalse(any('Adam' in v.name or 'RMSProp' in v.name for v in frozen.variables))
                    finally:frozen.close()
            finally:m.close()

if __name__=='__main__':unittest.main()
