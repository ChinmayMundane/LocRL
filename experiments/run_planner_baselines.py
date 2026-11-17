"""Run the paired planner matrix on validation or the locked final split."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--methods',nargs='+',default=['dubins_rrt_star','landmark_rrt_star'],
                   choices=['rrt_star','dubins_rrt_star','landmark_rrt_star'])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    p.add_argument('--num-maps',type=int,default=0)
    p.add_argument('--iterations',type=int,default=600)
    p.add_argument('--replan-interval',type=int,default=25)
    p.add_argument('--controller',choices=['global','local'],default='local')
    p.add_argument('--output-root',default=None)
    p.add_argument('--split',choices=['validation','final_test'],default='validation')
    p.add_argument('--evaluation-seed',type=int,default=None)
    p.add_argument('--execute-final-test',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.output_root is None:
        label=('official_final_test_v2' if a.split=='final_test' else 'validation')
        a.output_root=(f'outputs/planners/{label}' if a.controller=='local' else f'outputs/planners_global/{label}')
    for method in a.methods:
        for seed in a.seeds:
            cmd=[sys.executable,str(ROOT/'planners/evaluate.py'),'--method',method,'--seed',str(seed),
                 '--num-maps',str(a.num_maps),'--iterations',str(a.iterations),'--replan-interval',str(a.replan_interval),
                 '--output-root',a.output_root,'--controller',a.controller,'--split',a.split]
            if a.evaluation_seed is not None:
                cmd.extend(['--evaluation-seed',str(a.evaluation_seed)])
            if a.execute_final_test:
                cmd.append('--execute-final-test')
            if a.dry_run: print(' '.join(cmd))
            else: subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__': main()
