import copy
from dataclasses import replace
import json
import time
import numpy as np
import pytest
import torch

from curriculum.jackal_curriculum import JackalCurriculum, stages, validate_scenario
from navigation.jackal_curriculum_train import CurriculumEnv, load_config, main, run
from test_jackal_ppo_diagnostic import ScriptedJackalEnv


def outcomes(level, success=True):
    pool=stages()[level]
    return [dict(scenario_id=pool[i%len(pool)].scenario_id,status='completed',success=success,collision=False) for i in range(12)]


def test_stage_order_and_endpoints():
    p=stages()
    assert [len(x) for x in p]==[3,4,4,4]
    assert all(validate_scenario(s)==s for pool in p for s in pool)
    for start,goal in [((1.5,0,0),(3,0)),((float('nan'),0,0),(3,0)),((0,-1.5,0),(0,-1.5)),((20,0,0),(3,0))]:
        with pytest.raises(ValueError):
            validate_scenario(replace(p[0][0],start=start,goal=goal))


def test_promotion_retention_and_fallback():
    m=JackalCurriculum()
    m.charge(2048)
    assert m.observe(outcomes(0))=='retain'
    assert m.observe(outcomes(0))=='retain'
    m.charge(2048)
    assert m.observe(outcomes(0))=='promote'
    assert m.level==1 and m.level_steps==0
    m.charge(2048); assert m.observe(outcomes(1,False))=='retain'
    m.charge(2048); assert m.observe(outcomes(1,False))=='fallback'
    assert m.level==0


def test_insufficient_and_failed_evaluation_cannot_promote():
    m=JackalCurriculum(); m.charge(2048)
    assert m.observe(outcomes(0)[:1])=='insufficient'
    e=outcomes(0); e[0]['status']='failed'
    assert m.observe(e)=='insufficient'
    assert m.passes==0
    with pytest.raises(ValueError): m.observe(outcomes(0),source='fixed')
    with pytest.raises(ValueError): m.observe(outcomes(0),source='training')
    assert m.history==[]


def test_sampling_and_serialization():
    m=JackalCurriculum(); a=np.random.default_rng(642); b=np.random.default_rng(642)
    assert [m.sample(a) for _ in range(50)]==[m.sample(b) for _ in range(50)]
    m.charge(2048); m.observe(outcomes(0)); m.charge(2048); m.observe(outcomes(0))
    restored=JackalCurriculum(**json.loads(json.dumps(m.state_dict())))
    assert restored==m
    state=copy.deepcopy(a.bit_generator.state)
    expected=[m.sample(a) for _ in range(20)]
    b.bit_generator.state=state
    assert expected==[restored.sample(b) for _ in range(20)]


def test_transition_reset_and_finite(tmp_path):
    env=ScriptedJackalEnv(); m=JackalCurriculum()
    w=CurriculumEnv(env,m,np.random.default_rng(1),tmp_path,time.monotonic()+30)
    w.reset(); w.step(np.zeros(2)); m.level=1; w.reset()
    assert env.resets==2 and w.open_episode['length']==0
    assert env.reset_options[-1]['start'] in [s.start for s in stages()[1]]
    assert json.loads((tmp_path/'training_episodes.jsonl').read_text())['status']=='partial'
    with pytest.raises(ValueError): w.step(np.array([np.nan,0]))
    assert env.robot.stops>0
    env.nan_at=env.step_calls+1
    with pytest.raises(ValueError): w.step(np.zeros(2))
    with pytest.raises(ValueError): w._finite(np.full(40,np.nan))


def test_preview_never_runs(monkeypatch):
    monkeypatch.setattr('navigation.jackal_curriculum_train.run',lambda *a,**k: pytest.fail('preview executes training'))
    assert main(['preview'])==0
    with pytest.raises(ValueError): run(load_config('configs/jackal_curriculum_6h.yaml'),'unused',mode='dry-run')


def test_bounded_pilot_checkpoint_resume_and_cleanup(tmp_path):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        c=load_config('configs/jackal_curriculum_6h.yaml'); c['output_root']=str(tmp_path)
        c['pilot_timesteps']=128
        env=ScriptedJackalEnv(episode_length=8)
        first=run(c,tmp_path/'pilot',environment_factory=lambda:env)
        assert first['status']=='completed' and first['trained_steps']==128
        assert first['shutdown']['stopped'] and first['shutdown']['closed']
        meta=json.loads((tmp_path/'pilot/latest.json').read_text())
        assert meta['curriculum']['steps']==128 and meta['timesteps']==128 and len(meta['sha256'])==64
        resumed=run(c,tmp_path/'pilot',resume=True,environment_factory=lambda:ScriptedJackalEnv())
        assert resumed['status']=='completed' and resumed['trained_steps']==128
        assert resumed['curriculum']['steps']==128
        bad=copy.deepcopy(c); bad['seed']+=1
        result=run(bad,tmp_path/'pilot',resume=True,environment_factory=lambda:pytest.fail('must reject before env'))
        assert result['status']=='failed' and 'mismatch' in result['error']['message']
    finally:
        torch.set_num_threads(previous)


def test_failure_preserves_committed_checkpoint(tmp_path):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        c=load_config('configs/jackal_curriculum_6h.yaml'); c['output_root']=str(tmp_path)
        env=ScriptedJackalEnv(fail_at=5,fail_with=TimeoutError('scripted sensor failure'))
        result=run(c,tmp_path/'failure',environment_factory=lambda:env)
        assert result['status']=='failed' and result['trained_steps']==0
        assert result['shutdown']['closed'] and env.robot.stops>0
        assert (tmp_path/'failure/checkpoint_00000000.zip').exists()
        assert (tmp_path/'failure/failures.jsonl').exists()
    finally:
        torch.set_num_threads(previous)
