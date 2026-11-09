"""Actual paired BC / A* / random routes; no fabricated or repaired trajectories."""
from pathlib import Path
import numpy as np


def plot_paired_routes(obstacles, start, goal, records, filename, title=''):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    path=Path(filename)
    path.parent.mkdir(parents=True,exist_ok=True)
    figure,axes=plt.subplots(1,3,figsize=(15,6))
    figure.subplots_adjust(left=.05,right=.99,bottom=.12,top=.79,wspace=.25)
    for axis,name in zip(axes,('bc','astar','random')):
        row=records[name]
        axis.imshow(obstacles,cmap='Greys',vmin=0,vmax=1,origin='upper')
        points=np.asarray(row['route'])
        axis.plot(points[:,1],points[:,0],'-o',linewidth=1.2,markersize=2,color='#2463a8')
        axis.scatter(start[1],start[0],marker='s',s=90,color='#e49322',label='Start',zorder=4)
        axis.scatter(goal[1],goal[0],marker='*',s=160,color='#21a366',label='Goal',zorder=5)
        axis.set_xticks(range(obstacles.shape[1]))
        axis.set_yticks(range(obstacles.shape[0]))
        axis.grid(alpha=.2)
        status='goal' if row['success'] else 'timeout'
        axis.set_title(f"{name.upper()}: {status}\nsteps={row['steps']}, collisions={row['collisions']}, loop={row['loop_detected']}",fontsize=10)
        axis.set_xlabel('Column'); axis.set_ylabel('Row'); axis.legend(loc='upper right',fontsize=8)
    figure.suptitle(title or 'Paired navigation on the same held-out scenario',fontsize=13,y=.97)
    figure.savefig(path,dpi=150)
    plt.close(figure)


def plot_benchmark_summary(summary,filename):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    names=['bc','astar','random']
    figure,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for axis,key,label in zip(axes,('success_rate','mean_spl','mean_collisions'),
                             ('Goal success rate','Mean SPL','Mean collisions / episode')):
        axis.bar([n.upper() for n in names],[summary[n][key] for n in names],color=['#2463a8','#21a366','#e49322'])
        axis.set_title(label)
        if key != 'mean_collisions':
            axis.set_ylim(0,1)
        axis.grid(axis='y',alpha=.2)
    Path(filename).parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(filename,dpi=150)
    plt.close(figure)
