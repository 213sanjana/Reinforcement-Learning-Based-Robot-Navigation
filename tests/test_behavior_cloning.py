"""BC data, optimization, checkpoint, determinism and closed-loop checks."""
import copy
import csv
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from imitation.bc_dataset import BCDataset, load_training_pair
from imitation.bc_network import BCNetwork, load_bc_checkpoint, resolve_device
from imitation.dataset import load_dataset, save_dataset
from imitation.expert_generator import generate_dataset
from imitation.train_bc import EarlyStopping, run_epoch, seed_everything, train
from imitation.evaluate_bc import evaluate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def datasets(tmp_path):
    first, second = tmp_path/'train.npz', tmp_path/'validation.npz'
    generate_dataset(first, episodes=10, seed=1000000, split='train')
    a, _ = load_dataset(first)
    generate_dataset(second, episodes=5, seed=3000000, split='validation', excluded_ids=set(a['map_ids']))
    return first, second


def config(datasets, output):
    return {'data':{'train':str(datasets[0]),'validation':str(datasets[1])},
            'model':{'hidden_size':16}, 'training':{
                'batch_size':16, 'learning_rate':.0003, 'epochs':3, 'patience':8, 'min_delta':0.,
                'seed':2027, 'device':'cpu','num_workers':0,'deterministic':True,'gradient_clip':1.,
                'output_directory':str(output)}}


def test_dataset_contract_and_loader(datasets):
    a,b = load_training_pair(*datasets)
    x,y = a[0]
    assert x.shape == (3,10,10) and x.dtype == torch.float32 and y.dtype == torch.int64
    assert a.episodes == 10 and b.episodes == 5
    assert next(iter(DataLoader(a,batch_size=7)))[0].shape == (7,3,10,10)
    assert set(a.map_ids).isdisjoint(b.map_ids)


def test_rejects_test_split_and_leakage(datasets,tmp_path):
    a,meta = load_dataset(datasets[0]); meta['split']='test'
    path=tmp_path/'test.npz'; save_dataset(path,a,meta)
    with pytest.raises(ValueError,match='Expected split'):
        BCDataset(path,'train')
    with pytest.raises(ValueError,match='only train/validation'):
        BCDataset(path,'test')
    meta['split']='validation'
    leaked=tmp_path/'leaked.npz'; save_dataset(leaked,a,meta)
    with pytest.raises(ValueError,match='leakage'):
        load_training_pair(datasets[0],leaked)


def test_invalid_actions_and_channels(datasets,tmp_path):
    a,meta=load_dataset(datasets[0]); a['actions'][0]=4
    path=tmp_path/'bad.npz'; save_dataset(path,a,meta)
    with pytest.raises(ValueError,match='actions'):
        BCDataset(path,'train')
    a,meta=load_dataset(datasets[0]); meta['channels'].reverse()
    path=tmp_path/'channels.npz'; save_dataset(path,a,meta)
    with pytest.raises(ValueError,match='channel'):
        BCDataset(path,'train')


def test_reuses_cnn_and_output_shape():
    from models.cnn_dqn_network import CNNDQNNetwork
    seed_everything(7); original=CNNDQNNetwork(10,10,16)
    seed_everything(7); actor=BCNetwork(10,10,16)
    for a,b in zip(original.features.parameters(),actor.features.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(original.head[0].weight,actor.shared[0].weight,rtol=0,atol=0)
    assert actor(torch.zeros(2,3,10,10)).shape == (2,4)
    with pytest.raises(ValueError,match='Expected'):
        actor(torch.zeros(2,10,10,3))


def test_learning_and_validation_no_updates():
    seed_everything(3); model=BCNetwork(3,3,16)
    x=torch.zeros(8,3,3,3); x[:,1,0,0]=1; x[:,2,2,2]=1
    loader=DataLoader(TensorDataset(x,torch.full((8,),3,dtype=torch.int64)),batch_size=8)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    before,_=run_epoch(model,loader,torch.device('cpu'))
    for _ in range(15):
        run_epoch(model,loader,torch.device('cpu'),optimizer)
    after,accuracy=run_epoch(model,loader,torch.device('cpu'))
    assert after < before and accuracy == 1.
    state={k:v.clone() for k,v in model.state_dict().items()}
    run_epoch(model,loader,torch.device('cpu'))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())


def test_early_stopping():
    stopper=EarlyStopping(2,.01)
    assert not stopper.update(1.)
    assert not stopper.update(.995)
    assert stopper.update(.99)
    assert not stopper.update(.9)


def test_training_artifacts_reload_and_selection(datasets,tmp_path):
    output=tmp_path/'run'; summary=train(config(datasets,output))
    for name in ['best.pt','final.pt','metrics.csv','training.png','config.yaml','summary.json']:
        assert (output/name).stat().st_size > 0
    rows=list(csv.DictReader((output/'metrics.csv').open()))
    assert len(rows) == 3
    best_row=min(rows,key=lambda r:float(r['validation_loss']))
    best,payload=load_bc_checkpoint(output/'best.pt')
    _,last=load_bc_checkpoint(output/'final.pt')
    assert payload['epoch'] == int(best_row['epoch']) == summary['best_epoch']
    assert last['epoch'] == 3
    x=torch.zeros(2,3,10,10); again,_=load_bc_checkpoint(output/'best.pt')
    torch.testing.assert_close(best(x),again(x),rtol=0,atol=0)
    assert best.predict(x).shape == (2,)
    assert last['dataset_provenance']['train']['sha256'] == BCDataset(datasets[0],'train').sha256
    initial=(output/'best.pt').read_bytes()
    with pytest.raises(FileExistsError):
        train(config(datasets,output))
    assert (output/'best.pt').read_bytes() == initial


def test_determinism_and_early_stop_training(datasets,tmp_path):
    cfg=config(datasets,tmp_path/'a'); cfg['training'].update(epochs=5,patience=1,min_delta=1000000.)
    train(cfg)
    other=copy.deepcopy(cfg); other['training']['output_directory']=str(tmp_path/'b'); train(other)
    _,a=load_bc_checkpoint(tmp_path/'a/final.pt'); _,b=load_bc_checkpoint(tmp_path/'b/final.pt')
    assert a['epoch'] == b['epoch'] == 2
    assert all(torch.equal(a['model_state_dict'][k],b['model_state_dict'][k]) for k in a['model_state_dict'])
    for name in ('train_loss','train_accuracy','validation_loss','validation_accuracy'):
        assert a['metrics'][name] == b['metrics'][name]


def test_closed_loop_real_outcomes(datasets,tmp_path):
    cfg=config(datasets,tmp_path/'run'); cfg['training']['epochs']=1; train(cfg)
    report=evaluate(tmp_path/'run/best.pt',datasets[1],tmp_path/'rollout.json','cpu',episodes=5)
    assert len(report['results']) == report['episodes'] == 5
    assert report['success_rate'] == sum(r['success'] for r in report['results'])/5
    assert 0 <= report['mean_spl'] <= 1
    assert all(r['success'] or r['truncated'] for r in report['results'])
    assert all(1 <= r['steps'] <= 150 for r in report['results'])
    assert all(not r['success'] or r['steps'] >= r['optimal_steps'] for r in report['results'])
    with pytest.raises(ValueError,match='Expected split'):
        evaluate(tmp_path/'run/best.pt',datasets[0],tmp_path/'bad.json','cpu',episodes=1)
    with pytest.raises(FileExistsError):
        evaluate(tmp_path/'run/best.pt',datasets[1],tmp_path/'rollout.json','cpu',episodes=1)


def test_device_and_config_guards(datasets,tmp_path):
    assert resolve_device('cpu').type == 'cpu'
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='CUDA'):
            resolve_device('cuda')
    cfg=config(datasets,tmp_path/'run'); cfg['data']['test']='must_not_be_read.npz'
    with pytest.raises(ValueError,match='no test input'):
        train(cfg)
    assert not (tmp_path/'run').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in hosted runtime')
def test_cuda_forward_backward():
    seed_everything(2); model=BCNetwork(3,3,16).cuda()
    loss=torch.nn.functional.cross_entropy(model(torch.zeros(2,3,3,3,device='cuda')),
                                          torch.tensor([0,1],device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)


def test_closed_loop_known_success_and_failure(tmp_path):
    from environments.fixed_map_env import FixedMapNavigationEnv
    from imitation.expert_generator import replay_expert
    from imitation.dataset import encode_observation, map_identity
    grid=np.zeros((3,3),dtype=np.int8)
    env=FixedMapNavigationEnv(grid,(1,2),max_steps=5)
    try:
        obs,_=env.reset(options={'start':(1,0)})
        x,a,r=replay_expert(env,obs,[(1,0),(1,1),(1,2)])
    finally:
        env.close()
    identity=map_identity(grid)
    arrays=dict(observations=x,actions=a,rewards=r,episode_offsets=np.array([0,2],dtype=np.int64),
                map_ids=np.array([identity]),obstacles=grid[None],starts=np.array([[1,0]],dtype=np.int32),
                goals=np.array([[1,2]],dtype=np.int32),path_lengths=np.array([2]),map_seeds=np.array([1]),
                terminated=np.array([True]),truncated=np.array([False]))
    environment=dict(height=3,width=3,obstacle_density=0.,max_steps=5,bfs_shaping=False)
    metadata=dict(schema_version=1,split='validation',environment=environment,
                  channels=['obstacles','agent','goal'],actions=[[-1,0],[1,0],[0,-1],[0,1]])
    path=tmp_path/'known.npz'; save_dataset(path,arrays,metadata)
    model=BCNetwork(3,3,16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor.bias[3]=1
    from imitation.bc_network import ARCHITECTURE
    payload=dict(format_version=1,architecture=ARCHITECTURE,
                 network_config=dict(height=3,width=3,hidden_size=16,n_actions=4),
                 model_state_dict=model.state_dict(),epoch=1,environment=environment,
                 channels=metadata['channels'],actions=metadata['actions'],selection_metric='test fixture')
    checkpoint=tmp_path/'known.pt'; torch.save(payload,checkpoint)
    success=evaluate(checkpoint,path,tmp_path/'success.json','cpu',episodes=1)
    assert success['success_rate'] == success['mean_spl'] == 1.
    assert success['mean_steps'] == 2 and success['mean_collisions'] == 0
    assert success['results'][0]['route'][-1] == [1,2]
    with torch.no_grad():
        model.actor.bias.zero_(); model.actor.bias[0]=1
    torch.save(payload,checkpoint)
    failure=evaluate(checkpoint,path,tmp_path/'failure.json','cpu',episodes=1)
    assert failure['success_rate'] == failure['mean_spl'] == 0.
    assert failure['mean_steps'] == 5 and failure['mean_collisions'] == 4
    assert failure['results'][0]['truncated']
