"""Own one established ROS launch and one bounded diagnostic batch; no retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from navigation.jackal_generalization_diagnostic import ROOT, OUTPUT_ROOT, SETTINGS, BUDGETS, OUTER_TIMEOUTS


def processes():
    result={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines():
        parts=line.strip().split(None,2)
        if len(parts)==3:result[int(parts[0])]=(int(parts[1]),parts[2])
    return result


def descendants(pid, rows):
    owned={pid}
    while True:
        children={p for p,(parent,_) in rows.items() if parent in owned}
        if children.issubset(owned):return owned
        owned|=children


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=tuple(BUDGETS),default='pilot')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); output=args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists():
        raise ValueError('A new diagnostics output directory is required')
    support=output.with_name(output.name+'_supervisor')
    support.mkdir(parents=True,exist_ok=False)
    if any(os.environ.get(k)!=v for k,v in SETTINGS.items()):
        raise RuntimeError('Documented ROS/transport settings are required')
    before=processes()
    if any('ign gazebo' in cmd and not (' -c ' in cmd or 'codex' in cmd) for _,cmd in before.values()):
        raise RuntimeError('Existing Gazebo process: refusing to launch or stop unrelated simulator')
    launch=None; worker=None; owned=set(); result=dict(mode=args.mode,worker_budget_s=BUDGETS[args.mode],outer_timeout_s=OUTER_TIMEOUTS[args.mode]); started=time.monotonic()
    with (support/'simulator.log').open('w') as simlog,(support/'pilot_console.txt').open('w') as runlog:
        try:
            launch=subprocess.Popen(['ros2','launch','jackal_rl_bringup','jackal_sim.launch.py'],
                cwd=ROOT,stdout=simlog,stderr=subprocess.STDOUT,start_new_session=True)
            result['launch_pid']=launch.pid
            # Established bringup normally activates controllers in <30 seconds.
            time.sleep(35)
            if launch.poll() is not None:raise RuntimeError('Simulator launch exited during startup')
            worker=subprocess.Popen([sys.executable,'-m','navigation.jackal_generalization_diagnostic',args.mode,'--output',str(output)],
                cwd=ROOT,stdout=runlog,stderr=subprocess.STDOUT,start_new_session=True)
            result['pilot_pid']=worker.pid
            try:result['pilot_exit_code']=worker.wait(timeout=OUTER_TIMEOUTS[args.mode])
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGINT)
                try:worker.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGTERM);worker.wait(timeout=10)
                raise RuntimeError('Diagnostic worker exceeded outer deadline; no restart attempted')
        except BaseException as exc:
            result['failure']=f'{type(exc).__name__}: {exc}'
        finally:
            rows=processes()
            if worker is not None and worker.poll() is None:
                try:os.killpg(worker.pid,signal.SIGINT)
                except ProcessLookupError:pass
                try:worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGTERM)
                    worker.wait(timeout=5)
            if launch is not None:
                owned=descendants(launch.pid,rows)
                result['owned_processes']=[dict(pid=p,command=rows[p][1]) for p in sorted(owned) if p in rows]
                if launch.poll() is None:
                    try:os.killpg(launch.pid,signal.SIGINT)
                    except ProcessLookupError:pass
                    try:launch.wait(timeout=12)
                    except subprocess.TimeoutExpired:pass
                # Explicitly terminate only descendants recorded while ownership was intact.
                for p in owned:
                    try:os.kill(p,signal.SIGTERM)
                    except ProcessLookupError:pass
                time.sleep(3)
            alive=[]
            for p in owned:
                try:
                    state=Path(f'/proc/{p}/stat').read_text().rsplit(')',1)[1].split()[0]
                    if state!='Z':alive.append(p)
                except FileNotFoundError:pass
            result.update(remaining_owned_pids=alive,runtime_s=time.monotonic()-started,
                full_matrix_launched=args.mode=='matrix' and worker is not None,training_steps=0)
            (support/'cleanup.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result,indent=2))
    return 0 if result.get('pilot_exit_code')==0 and not result['remaining_owned_pids'] else 1

if __name__=='__main__':raise SystemExit(main())
