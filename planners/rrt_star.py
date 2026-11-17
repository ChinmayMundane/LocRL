"""Finite-budget RRT* with geometric or directed Dubins connections."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from planners.dubins import connection


class RRTStar:
    def __init__(self, world, *, seed=11, iterations=600, dubins=False,
                 radius=2.0, extension=6.0, spacing=.2, margin=.3, landmark_weight=0.0):
        self.world=np.asarray(world).copy()
        self.rng=np.random.default_rng(seed)
        self.iterations=iterations; self.dubins=dubins; self.radius=radius
        self.extension=extension; self.spacing=spacing; self.margin=margin
        self.landmark_weight=landmark_weight
        self.clearance=distance_transform_edt(self.world != 1)
        self.landmark_distance=(distance_transform_edt(self.world != 2)
                                if np.any(self.world == 2) else np.zeros_like(self.world,dtype=float))

    def cells(self, path):
        # Planner coordinates are Cartesian (x,-grid_y,yaw).
        return np.rint(np.c_[path[:,0],-path[:,1]]).astype(int)

    def free(self, path):
        cells=self.cells(path); x,y=cells.T
        if np.any(x<0) or np.any(y<0) or np.any(x>=self.world.shape[1]) or np.any(y>=self.world.shape[0]):
            return False
        # EDT is centre-to-centre; subtract half the pixel diagonal for a
        # conservative clearance proxy. Collision samples are <= spacing apart.
        return bool(np.all(self.clearance[y,x] > self.margin+math.sqrt(.5)))

    def edge(self, a, b):
        if self.dubins:
            path,length=connection(a,b,self.radius,self.spacing)
        else:
            length=float(np.linalg.norm(b[:2]-a[:2]))
            n=max(2,int(np.ceil(length/self.spacing))+1)
            path=np.linspace(a,b,n)
            path[:,2]=math.atan2(b[1]-a[1],b[0]-a[0])
        if not self.free(path):
            return None,math.inf
        if self.landmark_weight:
            x,y=self.cells(path).T
            ds=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
            density=np.minimum(self.landmark_distance[y,x]/10,1)
            length+=self.landmark_weight*float(np.dot(ds,(density[:-1]+density[1:])/2))
        return path,length

    def plan(self, start, goal):
        start=np.asarray(start,float); goal=np.asarray(goal,float)
        self.nodes=[start.copy()]; self.parents=[-1]; self.costs=[0.0]; self.edges=[None]
        h,w=self.world.shape
        for _ in range(self.iterations):
            sample=(goal.copy() if self.rng.random()<.15 else
                    np.array([self.rng.uniform(1,w-2),-self.rng.uniform(1,h-2),self.rng.uniform(-math.pi,math.pi)]))
            nodes=np.array(self.nodes)
            distances=np.linalg.norm(nodes[:,:2]-sample[:2],axis=1)
            nearest=int(np.argmin(distances)); a=nodes[nearest]
            if self.dubins:
                path,_=connection(a,sample,self.radius,self.spacing)
                lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))]
                new=path[min(np.searchsorted(lengths,self.extension),len(path)-1)].copy()
            else:
                new=sample.copy()
                if distances[nearest]>self.extension:
                    new[:2]=a[:2]+(sample[:2]-a[:2])*self.extension/distances[nearest]
            if np.linalg.norm(new[:2]-a[:2])<1e-5:
                continue
            initial,cost=self.edge(a,new)
            if initial is None:
                continue
            near=np.flatnonzero(np.linalg.norm(nodes[:,:2]-new[:2],axis=1)<=self.extension*1.5)
            parent=nearest; best=self.costs[nearest]+cost; edge=initial
            for j in near:
                if self.costs[j]+np.linalg.norm(nodes[j,:2]-new[:2])>=best:
                    continue
                p,c=self.edge(nodes[j],new)
                if self.costs[j]+c<best:
                    parent=int(j); best=self.costs[j]+c; edge=p
            idx=len(self.nodes)
            self.nodes.append(new); self.parents.append(parent); self.costs.append(best); self.edges.append(edge)
            ancestors=set(); j=parent
            while j>=0:
                ancestors.add(j); j=self.parents[j]
            for j in near:
                j=int(j)
                if j in ancestors or best+np.linalg.norm(new[:2]-nodes[j,:2])>=self.costs[j]:
                    continue
                p,c=self.edge(new,nodes[j])
                if best+c+1e-9<self.costs[j]:
                    delta=best+c-self.costs[j]
                    self.parents[j]=idx; self.edges[j]=p; self.costs[j]+=delta
                    stack=[j]
                    while stack:
                        current=stack.pop()
                        children=[k for k,par in enumerate(self.parents) if par==current]
                        for child in children:
                            self.costs[child]+=delta
                        stack.extend(children)
        # Free goal heading: consider eight terminal headings for Dubins.
        best=math.inf; endpoint=None; last=None
        for j,node in enumerate(self.nodes):
            if self.costs[j]+np.linalg.norm(node[:2]-goal[:2])>=best:
                continue
            headings=np.linspace(-math.pi,math.pi,8,endpoint=False) if self.dubins else [goal[2]]
            for heading in headings:
                target=goal.copy(); target[2]=heading
                path,c=self.edge(node,target)
                if self.costs[j]+c<best:
                    best=self.costs[j]+c; endpoint=j; last=path
        if endpoint is None:
            return None
        parts=[last]; j=endpoint
        while self.parents[j]>=0:
            parts.append(self.edges[j]); j=self.parents[j]
        return np.concatenate(list(reversed(parts)))


class PlannerController:
    """Known-map planner; runtime inputs are belief pose and contact only."""
    def __init__(self, world, goal, cfg, *, method, seed, iterations=600, replan_interval=25, landmark_weight=1.0):
        self.cfg=cfg; self.goal=np.array([goal[0],-goal[1],0.0])
        self.planner=RRTStar(world,seed=seed,iterations=iterations,dubins=method=='dubins_rrt_star',
                             radius=float(cfg.continuous_wheelbase)/math.tan(float(cfg.continuous_max_steering)),
                             landmark_weight=landmark_weight if method=='landmark_rrt_star' else 0)
        self.interval=replan_interval; self.path=None; self.cursor=0
        self.calls=0; self.failures=0

    def action(self, belief, step, collision_streak=0):
        pose=np.array([belief[0],-belief[1],belief[2]],float)
        if self.path is None or step%self.interval==0 or collision_streak==4:
            self.path=self.planner.plan(pose,self.goal); self.cursor=0; self.calls+=1
            self.failures+=int(self.path is None)
        target=self.goal[:2]
        if self.path is not None:
            window=self.path[self.cursor:self.cursor+40]
            self.cursor+=int(np.argmin(np.linalg.norm(window[:,:2]-pose[:2],axis=1)))
            j=self.cursor
            while j<len(self.path)-1 and np.linalg.norm(self.path[j,:2]-pose[:2])<2.0:
                j+=1
            target=self.path[j,:2]
        angle=math.atan2(target[1]-pose[1],target[0]-pose[0])-pose[2]
        angle=(angle+math.pi)%(2*math.pi)-math.pi
        steering=math.atan2(2*float(self.cfg.continuous_wheelbase)*math.sin(angle),max(np.linalg.norm(target-pose[:2]),.5))
        steering=float(np.clip(steering,-self.cfg.continuous_max_steering,self.cfg.continuous_max_steering))
        speed=float(self.cfg.continuous_max_speed)
        action=np.array([speed,steering],dtype=np.float32)
        # Shared one-step known-map safety check; identical for all planners.
        # Uses the environment's midpoint bicycle discretization, not true pose.
        def predicted(control):
            v,s=control; distance=v*float(self.cfg.continuous_dt)
            dh=distance*math.tan(s)/float(self.cfg.continuous_wheelbase)
            p=pose.copy(); p[:2]+=distance*np.array([math.cos(pose[2]+dh/2),math.sin(pose[2]+dh/2)]); p[2]+=dh
            return p
        def safe(control):
            p=predicted(control)
            return self.planner.free(np.linspace(pose,p,12))
        if not safe(action):
            candidates=[np.array([v,s],dtype=np.float32) for v in [float(self.cfg.continuous_min_speed),speed]
                        for s in np.linspace(-self.cfg.continuous_max_steering,self.cfg.continuous_max_steering,15)]
            feasible=[a for a in candidates if safe(a)]
            if feasible:
                action=min(feasible,key=lambda a:np.linalg.norm(predicted(a)[:2]-target))
            else:
                action=np.array([self.cfg.continuous_min_speed,np.copysign(self.cfg.continuous_max_steering,angle)],dtype=np.float32)
        return action
