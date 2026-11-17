"""Evaluate known-map planners through the canonical no-render evaluator."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'capql'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/localizationrl-matplotlib')
from evaluate import evaluate
from environment.map_dataset import rasterize_map
from planners.rrt_star import PlannerController
from planners.local_controller import LocalObservationController


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Adapter:
    def __init__(self,cfg,env,args):
        self.cfg=cfg; self.env=env; self.args=args

    def start_episode(self,spec,index):
        # Static map specification only. No hidden trajectory state enters the planner.
        world=rasterize_map(spec,int(self.cfg.map_size))
        controller_class = (LocalObservationController if self.args.controller=='local' else PlannerController)
        self.controller=controller_class(world,self.env.unwrapped.goal_point,self.cfg,
            method=self.args.method,seed=self.args.seed*100000+index,
            iterations=self.args.iterations,replan_interval=self.args.replan_interval,
            landmark_weight=self.args.landmark_weight)
        self.elapsed=0; self.latencies=[]

    def eval(self,observation,preference):
        start=time.perf_counter()
        base=self.env.unwrapped
        extra={}
        if self.args.controller=='local':
            size=int(self.cfg.view_size)
            extra['obstacle_image']=np.asarray(observation[:size*size]).reshape(size,size)
        action=self.controller.action(base.x_state.copy(),base.steps,int(base.collision_streak),**extra)
        duration=time.perf_counter()-start
        self.elapsed+=duration; self.latencies.append(duration)
        return action

    def episode_metrics(self):
        metrics=dict(planner_calls=self.controller.calls,planner_failures=self.controller.failures,
                    decision_time_seconds=self.elapsed,decision_latency_mean=float(np.mean(self.latencies)),
                    decision_latency_max=float(np.max(self.latencies)))
        if self.args.controller=='local':
            metrics.update(projected_routing_roots=self.controller.projected_roots,
                           rejected_local_controls=self.controller.local_rejections,
                           no_local_action_steps=self.controller.no_local_action)
        return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['rrt_star','dubins_rrt_star','landmark_rrt_star'],required=True)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--num-maps',type=int,default=0)
    parser.add_argument('--iterations',type=int,default=600)
    parser.add_argument('--replan-interval',type=int,default=25)
    parser.add_argument('--landmark-weight',type=float,default=1.0)
    parser.add_argument('--controller',choices=['global','local'],default='global')
    parser.add_argument('--output-root',default=None)
    parser.add_argument('--split',choices=['validation','final_test'],default='validation')
    parser.add_argument('--evaluation-seed',type=int,default=None)
    parser.add_argument('--locked-test-config',default='configs/experiments/final_test.yaml')
    parser.add_argument('--execute-final-test',action='store_true')
    args=parser.parse_args()
    locked_cfg_path=ROOT/args.locked_test_config
    locked_cfg=OmegaConf.load(locked_cfg_path)
    if args.split=='final_test':
        if not args.execute_final_test:
            parser.error('--execute-final-test is required for the locked final split')
        if str(locked_cfg.split)!='final_test':
            raise RuntimeError('Locked-test config does not declare final_test')
        manifest=ROOT/str(locked_cfg.dataset_dir)/f'{locked_cfg.split}.json'
        actual=file_sha256(manifest)
        if actual!=str(locked_cfg.sha256):
            raise RuntimeError(f'Locked final-test checksum mismatch: {actual}')
        default_eval_seed=int(locked_cfg.evaluation_seed)
    else:
        manifest=ROOT/'data/map_splits/validation.json'
        default_eval_seed=46000
    evaluation_seed=(default_eval_seed if args.evaluation_seed is None else args.evaluation_seed)
    if args.split=='final_test' and evaluation_seed!=int(locked_cfg.evaluation_seed):
        raise RuntimeError('Final-test evaluation seed must match the locked protocol')
    if args.output_root is None:
        label=('final_test' if args.split=='final_test' else 'validation')
        args.output_root=(f'outputs/planners/{label}' if args.controller=='local' else f'outputs/planners_global/{label}')
    if args.iterations<1 or args.replan_interval<1 or args.num_maps<0 or args.landmark_weight<0:
        parser.error('Invalid planner budget or map count')
    cfg=OmegaConf.load(ROOT/'configs/environment.yaml')
    cfg.seed=args.seed; cfg.eval_seed=evaluation_seed; cfg.test_map_split=args.split
    if args.split=='final_test':
        cfg.map_generation_seed=int(locked_cfg.generation_seed)
    cfg.morl_test_weights=[.5,.5]; cfg.full_eval_num_maps=args.num_maps
    cfg.full_eval_output_dir=str(ROOT/args.output_root/args.method/f'seed_{args.seed}')
    out=Path(cfg.full_eval_output_dir); out.mkdir(parents=True,exist_ok=True)
    protocol=dict(vars(args),dataset_sha256=file_sha256(manifest),
                  evaluation_seed=evaluation_seed,information_access='known static global map; estimated pose; contact streak',
                  objective=('path length plus landmark-distance exposure surrogate' if args.method=='landmark_rrt_star' else 'path length'),
                  training='none',goal_planning='goal centre, sampled terminal headings; execution accepts canonical goal region')
    if args.split=='final_test':
        protocol['locked_test_config_sha256']=file_sha256(locked_cfg_path)
    if args.controller=='local':
        protocol['controller_version']=LocalObservationController.VERSION
        protocol['information_access']+='; identical policy egocentric obstacle channel'
        protocol['local_horizon']=3
        protocol['local_steering_samples']=17
        protocol['local_speed_samples']=3
        protocol['routing_root_projection']='nearest globally admissible grid cell; belief unmodified'
        protocol['local_clearance_margin']=.3
        protocol['contact_recovery_streak']=2
        protocol['recovery_duration_steps']=8
        protocol['replan_retry_steps']=5
        protocol['local_controller_sha256']=hashlib.sha256((ROOT/'planners/local_controller.py').read_bytes()).hexdigest()
    else:
        # Preserve equality with archived global-controller protocol files.
        protocol.pop('controller')
    protocol_path=out/'planner_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise RuntimeError('Existing output has a different protocol; choose a new --output-root')
    if (out/'summary_nav_0p50_loc_0p50.json').exists():
        print(f'Skipping completed evaluation: {out}'); return
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    evaluate(cfg,policy_factory=lambda c,e:Adapter(c,e,args),policy_label=args.method,progress=True)


if __name__=='__main__':
    main()
