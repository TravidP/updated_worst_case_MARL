"""Independent DQN controllers with fixed-array replay and separate targets."""
import json
from pathlib import Path
import numpy as np
from tf_compat import tf
from experiments.core import Streams
from experiments.protocol import scientific_config

class Replay:
    def __init__(self, capacity, dimensions):
        self.capacity=int(capacity); self.size=0; self.cursor=0
        self.arrays=[]
        for width in dimensions:
            self.arrays.append(dict(obs=np.empty((capacity,width),np.float32),
                next_obs=np.empty((capacity,width),np.float32), action=np.empty(capacity,np.int32),
                reward=np.empty(capacity,np.float32), terminal=np.empty(capacity,np.bool_)))

    def add(self, obs, actions, rewards, next_obs, terminal):
        k=self.cursor
        for i,a in enumerate(self.arrays):
            for name,value in [('obs',obs[i]),('next_obs',next_obs[i]),('action',actions[i]),
                               ('reward',rewards[i]),('terminal',terminal)]: a[name][k]=value
        self.size=min(self.size+1,self.capacity); self.cursor=(k+1)%self.capacity

    @property
    def nbytes(self): return sum(a.nbytes for agent in self.arrays for a in agent.values())

    def save(self, directory):
        directory=Path(directory);directory.mkdir()
        (directory/'index.json').write_text(json.dumps(dict(size=self.size,cursor=self.cursor,capacity=self.capacity)))
        for i,a in enumerate(self.arrays):
            for start in range(0,self.size,4096):
                np.savez_compressed(str(directory/('%d_%d.npz'%(i,start))),
                                    **{k:v[start:min(start+4096,self.size)] for k,v in a.items()})

    def load(self, directory):
        directory=Path(directory);index=json.loads((directory/'index.json').read_text())
        if index['capacity']!=self.capacity or not 0<=index['size']<=self.capacity or not 0<=index['cursor']<self.capacity:
            raise ValueError('Replay identity mismatch')
        for i,a in enumerate(self.arrays):
            for start in range(0,index['size'],4096):
                with np.load(str(directory/('%d_%d.npz'%(i,start))),allow_pickle=False) as data:
                    for k,v in a.items(): v[start:min(start+4096,index['size'])]=data[k]
        self.size=index['size'];self.cursor=index['cursor']

class DQLController:
    def __init__(self, env, family, seed, training=True):
        from agents.controller import initialization_stream
        self.family=family; self.training=training;self.config=env.config['MODEL_CONFIG'];c=self.config
        self.rng=Streams(seed);self.learning_steps=0;self.scheduler_steps=0
        self.backward_calls=0;self.minibatch_updates=0;self.batch_histogram={};self.diagnostics=[]
        self.exploration_origin=0;self.continuation=False;self.previous_done=True
        self.batch=c.getint('batch_size');self.gamma=c.getfloat('gamma')
        self.reward_norm=c.getfloat('reward_norm');self.reward_clip=c.getfloat('reward_clip')
        self.signature=dict(kind='controller',network=env.network,family=family,n_s=list(map(int,env.n_s_ls)),
            n_a=list(map(int,env.n_a_ls)),config=scientific_config(env.config),assets=env.asset_hashes,
            lanes=env.metric.lanes,reward='learner_boundary_scaled_v3',architecture='dqn_128_64_v1')
        self.replay=Replay(c.getint('buffer_size'),env.n_s_ls) if training else None
        self.graph=tf.Graph();self.policies=[];self.online=[];self.target=[];self.updates=[]
        with self.graph.as_default(), initialization_stream(self.rng['initialization']):
            tf.set_random_seed(seed)
            for i,(width,actions) in enumerate(zip(env.n_s_ls,env.n_a_ls)):
                with tf.variable_scope('agent%d'%i):
                    obs=tf.placeholder(tf.float32,[None,int(width)],name='obs')
                    def net(x, scope):
                        with tf.variable_scope(scope):
                            h=tf.layers.dense(x,128,activation=tf.nn.relu,name='fc1')
                            h=tf.layers.dense(h,64,activation=tf.nn.relu,name='fc2')
                            return tf.layers.dense(h,int(actions),name='q')
                    q=net(obs,'online')
                    online=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='agent%d/online'%i)
                    self.online.extend(online)
                    entry=dict(obs=obs,q=q)
                    if training:
                        nxt=tf.placeholder(tf.float32,[None,int(width)],name='next_obs')
                        tq=net(nxt,'target')
                        target=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='agent%d/target'%i)
                        self.target.extend(target)
                        action=tf.placeholder(tf.int32,[None]);reward=tf.placeholder(tf.float32,[None])
                        terminal=tf.placeholder(tf.bool,[None]);lr=tf.placeholder(tf.float32,[])
                        selected=tf.reduce_sum(q*tf.one_hot(action,int(actions)),axis=1)
                        y=tf.stop_gradient(reward+self.gamma*(1.-tf.cast(terminal,tf.float32))*tf.reduce_max(tq,axis=1))
                        loss=tf.losses.huber_loss(y,selected)
                        grads=tf.gradients(loss,online);norm=tf.global_norm(grads)
                        clipped,_=tf.clip_by_global_norm(grads,c.getfloat('max_grad_norm'))
                        optimizer=tf.train.AdamOptimizer(lr,epsilon=c.getfloat('adam_epsilon'))
                        metrics=dict(td_loss=loss,predicted_q=tf.reduce_mean(selected),target_q=tf.reduce_mean(y),grad_norm=norm)
                        with tf.control_dependencies(list(metrics.values())):
                            update=optimizer.apply_gradients(zip(clipped,online))
                        entry.update(next_obs=nxt,action=action,reward=reward,terminal=terminal,lr=lr,
                                     update=update,metrics=metrics,target=y)
                self.policies.append(entry)
            self.sync=[tf.assign(t,o) for o,t in zip(self.online,self.target)]
            self.variables=tf.global_variables();self.trainable=tf.trainable_variables()
            self.saver=tf.train.Saver(self.variables,max_to_keep=0)
            self.inference_saver=tf.train.Saver(self.online,max_to_keep=0)
            self.sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1,device_count={'GPU':0}))
            self.sess.run(tf.global_variables_initializer())
            if training:self.sess.run(self.sync)

    @property
    def lr(self):
        from agents.controller import scheduled
        return scheduled(self.config,'lr',self.scheduler_steps)

    @property
    def epsilon(self):
        if self.continuation:
            return .1+(.01-.1)*min(1.,max(0,self.learning_steps-self.exploration_origin)/132000.)
        return 1.+(.01-1.)*min(1.,self.scheduler_steps/self.config.getint('epsilon_decay_steps'))

    def branch(self): self.continuation=True;self.exploration_origin=self.learning_steps
    def reset(self): self.previous_done=True
    def flush(self,*args,**kwargs): pass
    def transform_rewards(self,rewards):
        scaled=np.asarray(rewards,dtype=np.float32)/self.reward_norm
        learner=np.clip(scaled,-self.reward_clip,self.reward_clip) if self.reward_clip>0 else scaled
        return learner,learner!=scaled

    def act(self,obs,env,learning):
        if learning and not self.training:raise ValueError('Inference-only controller')
        q=self.sess.run([p['q'] for p in self.policies],{p['obs']:[o] for p,o in zip(self.policies,obs)})
        actions=[]
        for values in q:
            actions.append(int(self.rng['policy'].randint(values.shape[1])) if learning and self.rng['policy'].rand()<self.epsilon else int(np.argmax(values[0])))
        return actions,[],[]

    def observe(self,obs,decision,rewards,next_obs,done,learning):
        reward,clipped=self.transform_rewards(rewards)
        if learning:
            if not self.training:raise ValueError('Inference-only controller')
            self.replay.add(obs,decision[0],reward,next_obs,done)
            self.learning_steps+=1;self.scheduler_steps+=1
            if self.learning_steps>=self.config.getint('learning_starts') and self.replay.size>=self.batch and self.learning_steps%self.config.getint('update_interval')==0:
                for i,p in enumerate(self.policies):
                    index=self.rng['replay'].choice(self.replay.size,self.batch,replace=False)
                    data=self.replay.arrays[i]
                    feed={p[k]:v[index] for k,v in data.items()};feed[p['lr']]=self.lr
                    metrics,_=self.sess.run([p['metrics'],p['update']],feed)
                    self.diagnostics.append(dict(metrics,agent=i,epsilon=self.epsilon,learning_steps=self.learning_steps))
                self.backward_calls+=1;self.minibatch_updates+=1
                self.batch_histogram[self.batch]=self.batch_histogram.get(self.batch,0)+1
            if self.learning_steps%self.config.getint('target_interval')==0:self.sess.run(self.sync)
        self.previous_done=bool(done)
        return reward,clipped

    def state(self):
        return dict(rng=self.rng.state(),**{k:getattr(self,k) for k in ('learning_steps','scheduler_steps',
            'backward_calls','minibatch_updates','batch_histogram','exploration_origin','continuation','previous_done')})
    def restore_state(self,state):
        self.rng.restore(state['rng'])
        for k,v in state.items():
            if k!='rng':setattr(self,k,v)
    def assert_finite(self):
        for value in self.sess.run(self.variables):
            if not np.isfinite(value).all():raise FloatingPointError('Nonfinite DQL parameters')
    def close(self):self.sess.close()
