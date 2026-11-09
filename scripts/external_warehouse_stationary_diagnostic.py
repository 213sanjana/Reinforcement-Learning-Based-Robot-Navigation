"""One approved, bounded stationary attempt. No velocity publisher or policy."""
from pathlib import Path
import argparse
if __package__:
    from .stationary_readiness import ReadinessGate
else:
    from stationary_readiness import ReadinessGate
import os, sys, json, time, signal, subprocess, hashlib, math, datetime
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from ros_gz_interfaces.msg import Contacts
from controller_manager_msgs.srv import ListControllers, ListHardwareComponents, ListHardwareInterfaces
from rosidl_runtime_py.convert import message_to_ordereddict

ROOT=Path(__file__).resolve().parents[1]
OUT=None
readiness=None
last_readiness=None
START=time.monotonic()
files={}
summary={'approval':'one stationary attempt; 180s live + 15s cleanup; owned stock escalation approved','attempts':0,'commands_received':0,'nonzero_commands':0,'counts':{},'services':{},'reason':None,'owned':{}}
launch=None; launch_start=None; stopping=False; stop_at=None; ready_at=None; query=None; query_index=0; last_graph=0; last_tree=0; scene_proc=None; scene_started=False

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(kind,data):
    if kind not in files:files[kind]=(OUT/(kind+'.jsonl')).open('x')
    files[kind].write(json.dumps({'utc':utc(),'monotonic':time.monotonic(),**data},allow_nan=False)+'\n');files[kind].flush()
def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        return {'pid':pid,'ppid':int(fields[1]),'state':fields[0],'start_ticks':fields[19],'uid':p.stat().st_uid,'args':(p/'cmdline').read_bytes().decode(errors='replace').split('\0'),'cwd':str((p/'cwd').resolve())}
    except OSError:return None
def process_table():
    return {int(p.name):v for p in Path('/proc').iterdir() if p.name.isdigit() and (v:=identity(int(p.name))) is not None}
def matching(table):
    found=[]
    for v in table.values():
        args=v['args'];words=' '.join(args).split();names=[Path(a).name for a in args if a]
        if ('gazebo' in words and any(x in words for x in ('ign','gz'))) or ('sim' in words and 'gz' in words) or 'ros2' in names or any(x in names for x in ('parameter_bridge','robot_state_publisher','ros2_control_node')):
            found.append(v)
    return found
def tree():
    table=process_table();known={launch.pid};changed=True
    while changed:
        added={pid for pid,v in table.items() if v['ppid'] in known};changed=bool(added-known);known|=added
    result=[]
    for pid in known:
        v=table.get(pid)
        if v:
            key=str(pid);old=summary['owned'].get(key)
            if old and old['start_ticks']!=v['start_ticks']:continue
            summary['owned'][key]=v;result.append(v)
    write('process_tree',{'processes':result})
    return result
def request_stop(reason):
    global stopping,stop_at
    if stopping:return
    stopping=True;stop_at=time.monotonic();summary['reason']=reason
    if readiness is not None:readiness.begin_shutdown()
    write('shutdown',{'reason':reason,'action':'begin shutdown','elapsed_live':stop_at-launch_start if launch_start else None})
    print('STOP: '+reason,flush=True)
    if launch is not None and launch.poll() is None:
        tree();v=identity(launch.pid);old=summary['owned'].get(str(launch.pid))
        expected=['/usr/bin/python3','/opt/ros/humble/bin/ros2','launch','jackal_rl_bringup','jackal_external_warehouse.launch.py']
        if v and old and v['start_ticks']==old['start_ticks'] and v['uid']==os.getuid() and v['args'][:5]==expected:
            # launch may already be shutting down after a guarded failed stage.
            text=(OUT/'launch_console.txt').read_text(errors='replace')
            if 'sending signal' in text or 'Shutdown' in text or 'Shutting down' in text:
                write('shutdown',{'action':'launch shutdown already observed; no duplicate signal','pid':launch.pid})
            else:
                os.kill(launch.pid,signal.SIGINT);write('shutdown',{'action':'signal','signal':'SIGINT','identity':v})
        else:
            write('shutdown',{'action':'ownership could not be verified; no signal','identity':v})
            summary['ownership_error']=True

streams={}
def stamp(msg):return msg.sec*1000000000+msg.nanosec
def record_stream(name,s,extra):
    now=time.monotonic();v=streams.setdefault(name,{'count':0,'first_stamp':s,'last_stamp':s,'advances':0,'duplicates':0,'backwards':0,'first_receive':now,'last_receive':now,'max_receive_gap_s':0,'last_advance_receive':now})
    if v['count']:
        v['advances']+=int(s>v['last_stamp']);v['duplicates']+=int(s==v['last_stamp']);v['backwards']+=int(s<v['last_stamp'])
        v['max_receive_gap_s']=max(v['max_receive_gap_s'],now-v['last_receive'])
        if s>v['last_stamp']:v['last_advance_receive']=now
    v['count']+=1;v['last_stamp']=s;v['last_receive']=now
    write(name,{'stamp_ns':s,**extra})
    if readiness is not None:readiness.ingest(name,s,now,extra.get('readiness_valid',True))
def clock(msg):record_stream('clock',stamp(msg.clock),{})
def odom(msg):
    d=message_to_ordereddict(msg);record_stream('odometry',stamp(msg.header.stamp),{'message':d, 'readiness_valid': msg.header.frame_id=='odom' and msg.child_frame_id=='base_link'})
    t=msg.twist.twist;p=msg.pose.pose.position;q=msg.pose.pose.orientation
    vals=[p.x,p.y,p.z,q.x,q.y,q.z,q.w,t.linear.x,t.linear.y,t.angular.z]
    if not all(math.isfinite(v) for v in vals):request_stop('nonfinite odometry')
def scan(msg):
    ranges=list(msg.ranges);valid=[v for v in ranges if math.isfinite(v) and msg.range_min<=v<=msg.range_max]
    digest=hashlib.sha256(str(ranges).encode()).hexdigest()
    extra={'frame':msg.header.frame_id,'range_min':msg.range_min,'range_max':msg.range_max,'measurements':len(ranges),'finite_valid':len(valid),'minimum':min(valid) if valid else None,'ranges_sha256':digest, 'readiness_valid': bool(valid) and msg.header.frame_id=='lidar3d_0_laser'}
    if streams.get('lidar',{}).get('count',0)<3:extra['ranges']=[v if math.isfinite(v) else str(v) for v in ranges]
    record_stream('lidar',stamp(msg.header.stamp),extra)
def command(msg):
    d=message_to_ordereddict(msg);summary['commands_received']+=1;write('velocity_commands',{'message':d})
    if any(v!=0 for v in [msg.linear.x,msg.linear.y,msg.linear.z,msg.angular.x,msg.angular.y,msg.angular.z]):
        summary['nonzero_commands']+=1;print('UNEXPECTED NONZERO VELOCITY COMMAND OBSERVED',flush=True);request_stop('UNEXPECTED NONZERO VELOCITY COMMAND')
def contact(msg):
    record_stream('contacts',stamp(msg.header.stamp),{'message':message_to_ordereddict(msg)})

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Stationary diagnostic; separate live approval required')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute-approved-stationary',action='store_true')
    args=parser.parse_args()
    if not args.execute_approved_stationary:
        parser.error('Separate explicit live authorization is required; execution flag not supplied')
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)  # refuse historical/existing evidence directories
    try:
        table=process_table();existing=matching(table);write('preflight',{'matching_processes':existing,'environment':{k:os.environ.get(k) for k in ('IGN_PARTITION','GZ_PARTITION','ROS_DOMAIN_ID')}})
        if existing:raise RuntimeError('Existing simulation/ROS launch ownership requires investigation; no launch')
        source=ROOT/'jackal_rl_ws/src/jackal_rl_bringup/launch/jackal_external_warehouse.launch.py';installed=ROOT/'jackal_rl_ws/install/jackal_rl_bringup/share/jackal_rl_bringup/launch/jackal_external_warehouse.launch.py'
        expected='73b0b55ebd3c33f14a7919a21d3ffa76edd52dd65b7601112a6bfd2f5679f30f'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        assert installed.resolve()==source and installed.read_bytes()==source.read_bytes()
        rclpy.init();node=Node('external_warehouse_bounded_diagnostic')
        node.create_subscription(Clock,'/clock',clock,qos_profile_sensor_data)
        node.create_subscription(Odometry,'/j100_0000/platform/odom',odom,qos_profile_sensor_data)
        node.create_subscription(LaserScan,'/j100_0000/sensors/lidar3d_0/scan',scan,qos_profile_sensor_data)
        node.create_subscription(Twist,'/j100_0000/platform/cmd_vel_unstamped',command,10)
        node.create_subscription(Contacts,'/jackal_rl/obstacle_contacts',contact,10)
        before=time.monotonic()
        while time.monotonic()-before<3:rclpy.spin_once(node,timeout_sec=.1)
        pubs=node.get_publishers_info_by_topic('/j100_0000/platform/cmd_vel_unstamped');nodes=node.get_node_names_and_namespaces()
        write('preflight',{'nodes':nodes,'command_publishers':[p.node_name for p in pubs]})
        if pubs or any(ns=='/j100_0000' for _,ns in nodes):raise RuntimeError('Pre-existing command publisher/robot graph; no launch')
        (OUT/'launch_ros_logs').mkdir();env=os.environ.copy();env['ROS_LOG_DIR']=str(OUT/'launch_ros_logs')
        cmd=['ros2','launch','jackal_rl_bringup','jackal_external_warehouse.launch.py','mode:=simulation_only','gui:=true']
        console=(OUT/'launch_console.txt').open('x');launch_start=time.monotonic()
        launch=subprocess.Popen(cmd,stdout=console,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        readiness=ReadinessGate(launch_start)
        summary['attempts']=1;summary['launch_pid']=launch.pid;summary['launch_utc']=utc();summary['launch_command']=cmd
        tree();print('LAUNCHED one simulation_only attempt, PID '+str(launch.pid),flush=True)
        services=[('list_controllers',ListControllers),('list_hardware_components',ListHardwareComponents),('list_hardware_interfaces',ListHardwareInterfaces)]
        while True:
            rclpy.spin_once(node,timeout_sec=.05);now=time.monotonic();age=now-launch_start
            if now-last_tree>=1:tree();last_tree=now
            if now-last_graph>=1:
                pubs=node.get_publishers_info_by_topic('/j100_0000/platform/cmd_vel_unstamped')
                write('graph',{'nodes':node.get_node_names_and_namespaces(),'services':node.get_service_names_and_types(),'command_publishers':[{'node':p.node_name,'namespace':p.node_namespace} for p in pubs]});last_graph=now
                if pubs:request_stop('unexpected velocity-command publisher')
            text=(OUT/'launch_console.txt').read_text(errors='replace')
            if not stopping:
                if '[ERROR]' in text and ('process has died' in text or 'failed to terminate' in text):request_stop('launch process failure observed; no retry')
                elif launch.poll() is not None:request_stop('launch exited')
            state=readiness.evaluate(now, launch.poll() is None)
            # Terminal output is diagnostic only, never a readiness input.
            key=(state['status'],state['reason'])
            if key != last_readiness:
                write('readiness',state);last_readiness=key
            if not stopping and state['status'] in ('failed','complete'):
                request_stop(state['reason'])
            if not stopping and not scene_started and age>=16:
                scene_started=True;scene_file=(OUT/'scene.txt').open('x')
                scene_proc=subprocess.Popen(['timeout','8','ign','service','-s','/world/industrial-warehouse/scene/info','--reqtype','ignition.msgs.Empty','--reptype','ignition.msgs.Scene','--timeout','5000','--req',''],stdout=scene_file,stderr=subprocess.STDOUT,env=os.environ.copy())
                write('commands',{'argv':scene_proc.args,'pid':scene_proc.pid,'purpose':'read-only scene'})
            # One request per service; stop on first unresponsive manager.
            if not stopping and age>=20 and query_index<len(services):
                if query is None:
                    name,kind=services[query_index];client=node.create_client(kind,'/j100_0000/controller_manager/'+name)
                    query={'name':name,'kind':kind,'client':client,'start':now,'future':None};write('service_calls',{'service':name,'phase':'discover'})
                q=query
                if q['future'] is None:
                    if q['client'].service_is_ready():q['future']=q['client'].call_async(q['kind'].Request());q['sent']=now;write('service_calls',{'service':q['name'],'phase':'sent'})
                    elif now-q['start']>=3:
                        summary['services'][q['name']]={'status':'FAILED','reason':'not discoverable in 3 seconds'};request_stop('controller manager service discovery failed')
                elif q['future'].done():
                    try:
                        response=message_to_ordereddict(q['future'].result())
                        if q['name']=='list_controllers':readiness.controller_response(response)
                        summary['services'][q['name']]={'status':'VERIFIED','response':response};write('service_calls',{'service':q['name'],'phase':'response','response':response});query_index+=1;query=None
                    except Exception as e:request_stop('controller manager response error: '+str(e))
                elif now-q['sent']>=5:
                    summary['services'][q['name']]={'status':'FAILED','reason':'response timeout 5 seconds'};write('service_calls',{'service':q['name'],'phase':'timeout'});request_stop('controller manager service response timeout; no repeated requests')
            if stopping:
                if launch.poll() is not None:
                    summary['launch_exit_code']=launch.returncode;summary['live_duration_s']=stop_at-launch_start;summary['cleanup_duration_s']=max(0.0,time.monotonic()-stop_at);break
                if now-stop_at>=15:
                    summary['cleanup_deadline_exceeded']=True;summary['remaining_tree']=tree();break
        summary['scene_exit_code']=scene_proc.poll() if scene_proc else None
    except Exception as e:
        summary['exception']=type(e).__name__+': '+str(e);print(summary['exception'],flush=True)
        if launch is not None:request_stop('observer/supervisor exception')
    finally:
        summary['streams']=streams;summary['finished_utc']=utc()
        if readiness is not None:summary['readiness']=readiness.evaluate(time.monotonic(),launch.poll() is None)
        if launch is not None:
            # Normal loop already bounds cleanup. Never add a second launch or direct kill.
            if stopping and launch.poll() is None and time.monotonic()-stop_at<15:
                while launch.poll() is None and time.monotonic()-stop_at<15:time.sleep(.1)
            summary['launch_exit_code']=launch.poll()
            remaining=[]
            for key,old in summary['owned'].items():
                current=identity(int(key))
                if current and current['start_ticks']==old['start_ticks']:remaining.append(current)
            summary['remaining_owned']=remaining
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print(json.dumps({'reason':summary['reason'],'streams':{k:v['count'] for k,v in streams.items()},'services':summary['services'],'remaining_owned':summary.get('remaining_owned',[])}),flush=True)
        for f in files.values():f.close()
        if 'node' in globals():node.destroy_node();rclpy.shutdown()
