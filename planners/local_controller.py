"""Global planning with local-observation trajectory checks, version 1.

Only belief pose, the policy's obstacle channel, and contact history are inputs.
The routing anchor may be projected to a free global cell; the belief is never
changed. Short local trajectories arbitrate execution independently of that map.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from planners.rrt_star import PlannerController


class LocalObservationController(PlannerController):
    VERSION = 'local_observation_v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_plan_step=-1000
        self.projected_roots=0; self.local_rejections=0; self.no_local_action=0
        self.previous_steering=0.0; self.recovery_sign=1.0; self.recovery_until=-1

    def routing_anchor(self, pose):
        """Return a search-only free anchor, not a corrected robot pose."""
        if self.planner.free(pose[None,:]):
            return pose.copy(), False
        yy,xx=np.nonzero(self.planner.clearance > self.planner.margin+math.sqrt(.5))
        if not len(xx):
            return None, False
        j=np.argmin((xx-pose[0])**2+(-yy-pose[1])**2)
        anchor=pose.copy(); anchor[:2]=[xx[j],-yy[j]]
        return anchor, True

    def local_rollout(self, action):
        """Three decisions, midpoint bicycle integration, sampled every <=.1s."""
        v,s=map(float,action); state=np.zeros(3); points=[state.copy()]
        dt=float(self.cfg.continuous_dt); wheelbase=float(self.cfg.continuous_wheelbase)
        n=max(1,int(np.ceil(dt/.1)))
        for _ in range(3):
            origin=state.copy()
            for fraction in np.linspace(1/n,1,n):
                distance=v*dt*fraction; dh=distance*math.tan(s)/wheelbase
                points.append(origin+np.array([distance*math.cos(origin[2]+dh/2),
                                               distance*math.sin(origin[2]+dh/2),dh]))
            state=points[-1]
        return np.asarray(points)

    @staticmethod
    def local_clearance(path, obstacle_image):
        image=np.asarray(obstacle_image,dtype=bool)
        half=image.shape[0]//2
        # Body x is forward (image up), body y left (image left).
        col=np.rint(half-path[:,1]).astype(int)
        row=np.rint(half-path[:,0]).astype(int)
        if np.any(col<0) or np.any(row<0) or np.any(col>=image.shape[1]) or np.any(row>=image.shape[0]):
            return False,0.0
        if np.any(image[row,col]):
            return False,0.0
        if not image.any():
            return True,float(half)
        distances=distance_transform_edt(~image)
        # Pixels are sampled sensor cells. No claim of continuous safety under noise.
        clearance=float(np.min(distances[row,col]))
        # Same raster clearance proxy as the global planner. Reject cells
        # adjacent to sensed occupancy to allow for sub-cell aliasing.
        return clearance>math.sqrt(.5)+.3,clearance

    def action(self, belief, step, collision_streak=0, obstacle_image=None):
        if obstacle_image is None:
            raise ValueError('Local controller requires the policy obstacle image')
        image=np.asarray(obstacle_image)
        size=int(self.cfg.view_size)
        if image.shape!=(size,size) or not np.all(np.isfinite(image)):
            raise ValueError('Invalid obstacle image')
        pose=np.array([belief[0],-belief[1],belief[2]],float)
        due=(step%self.interval==0 or collision_streak==4 or self.path is None)
        if due and step-self.last_plan_step>=5:
            anchor,projected=self.routing_anchor(pose)
            self.projected_roots+=int(projected)
            self.path=None if anchor is None else self.planner.plan(anchor,self.goal)
            self.cursor=0; self.calls+=1; self.failures+=int(self.path is None)
            self.last_plan_step=step
        target=self.goal[:2]
        if self.path is not None:
            # Route cursor only chooses a target; local sensing decides executable controls.
            window=self.path[self.cursor:self.cursor+50]
            self.cursor+=int(np.argmin(np.linalg.norm(window[:,:2]-pose[:2],axis=1)))
            j=self.cursor
            while j<len(self.path)-1 and np.linalg.norm(self.path[j,:2]-pose[:2])<3:
                j+=1
            target=self.path[j,:2]
        delta=target-pose[:2]; c,s=math.cos(pose[2]),math.sin(pose[2])
        body_target=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
        if np.linalg.norm(body_target)>5:
            body_target*=5/np.linalg.norm(body_target)
        max_s=float(self.cfg.continuous_max_steering)
        if collision_streak>=2 and step>=self.recovery_until:
            left=float(np.sum(image[:,:size//2]));right=float(np.sum(image[:,size//2+1:]))
            self.recovery_sign=1.0 if left<=right else -1.0
            self.recovery_until=step+8
        speeds=np.linspace(self.cfg.continuous_min_speed,self.cfg.continuous_max_speed,3)
        candidates=[]
        for v in speeds:
            for steer in np.linspace(-max_s,max_s,17):
                if step<self.recovery_until and (steer*self.recovery_sign < .75*max_s or v>speeds[0]):
                    continue
                action=np.array([v,steer],dtype=np.float32)
                path=self.local_rollout(action)
                free,clearance=self.local_clearance(path,image)
                if not free:
                    self.local_rejections+=1;continue
                endpoint=path[-1]
                target_angle=math.atan2(body_target[1]-endpoint[1],body_target[0]-endpoint[0])
                score=np.linalg.norm(endpoint[:2]-body_target)+.4*(1-math.cos(target_angle-endpoint[2]))
                score+=.1/(clearance+.1)+.03*abs(steer-self.previous_steering)
                if step<self.recovery_until and steer*self.recovery_sign<0:
                    score+=1.0
                candidates.append((float(score),action))
        if candidates:
            action=min(candidates,key=lambda item:item[0])[1]
        else:
            # Positive minimum speed forbids a stop action. Keep a turn direction
            # briefly instead of alternating at contact; simulator may still collide.
            self.no_local_action+=1
            if step>=self.recovery_until:
                left=float(np.sum(image[:,:size//2]));right=float(np.sum(image[:,size//2+1:]))
                self.recovery_sign=1.0 if left<=right else -1.0
                self.recovery_until=step+8
            action=np.array([self.cfg.continuous_min_speed,self.recovery_sign*max_s],dtype=np.float32)
        self.previous_steering=float(action[1])
        return action
