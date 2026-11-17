"""Six-family shortest Dubins connections, sampled in Cartesian coordinates.

Closed-form families adapted from Atsushi Sakai's PythonRobotics (MIT).
Source: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/DubinsPath/dubins_path_planner.py
See THIRD_PARTY_LICENSE.txt. Sampling/integration and the public API are local.
"""
from math import sin, cos, atan2, sqrt, acos, pi, hypot
import numpy as np


def connection(start, goal, radius, spacing=0.2):
    if radius <= 0 or spacing <= 0:
        raise ValueError('radius and spacing must be positive')
    dx, dy = goal[0]-start[0], goal[1]-start[1]
    theta = atan2(dy, dx)
    a, b, d = (start[2]-theta) % (2*pi), (goal[2]-theta) % (2*pi), hypot(dx,dy)/radius
    sa,sb,ca,cb,cab = sin(a),sin(b),cos(a),cos(b),cos(a-b)
    mod = lambda x: x % (2*pi)
    candidates = []
    p2 = 2+d*d-2*cab+2*d*(sa-sb)
    if p2 >= 0:
        t=atan2(cb-ca,d+sa-sb); candidates.append(('LSL',[mod(-a+t),sqrt(p2),mod(b-t)]))
    p2=2+d*d-2*cab+2*d*(sb-sa)
    if p2 >= 0:
        t=atan2(ca-cb,d-sa+sb); candidates.append(('RSR',[mod(a-t),sqrt(p2),mod(-b+t)]))
    p2=-2+d*d+2*cab+2*d*(sa+sb)
    if p2 >= 0:
        p=sqrt(p2); t=atan2(-ca-cb,d+sa+sb)-atan2(-2,p)
        candidates.append(('LSR',[mod(-a+t),p,mod(-b+t)]))
    p2=d*d-2+2*cab-2*d*(sa+sb)
    if p2 >= 0:
        p=sqrt(p2); t=atan2(ca+cb,d-sa-sb)-atan2(2,p)
        candidates.append(('RSL',[mod(a-t),p,mod(b-t)]))
    t=(6-d*d+2*cab+2*d*(sa-sb))/8
    if abs(t)<=1:
        p=mod(2*pi-acos(t)); q=mod(a-atan2(ca-cb,d-sa+sb)+p/2)
        candidates.append(('RLR',[q,p,mod(a-b-q+p)]))
    t=(6-d*d+2*cab+2*d*(-sa+sb))/8
    if abs(t)<=1:
        p=mod(2*pi-acos(t)); q=mod(-a-atan2(ca-cb,d+sa-sb)+p/2)
        candidates.append(('LRL',[q,p,mod(b-a-q+p)]))
    modes,lengths=min(candidates,key=lambda item:sum(item[1]))
    state=np.asarray(start,dtype=float).copy(); points=[state.copy()]
    for mode,length in zip(modes,lengths):
        physical=length*radius
        n=max(1,int(np.ceil(physical/spacing)))
        ds=physical/n; curvature={'L':1/radius,'R':-1/radius,'S':0}[mode]
        for _ in range(n):
            yaw=state[2]
            if curvature:
                end=yaw+ds*curvature
                state[0]+=(sin(end)-sin(yaw))/curvature
                state[1]+=(-cos(end)+cos(yaw))/curvature
                state[2]=end
            else:
                state[0]+=ds*cos(yaw); state[1]+=ds*sin(yaw)
            points.append(state.copy())
    return np.asarray(points),sum(lengths)*radius
