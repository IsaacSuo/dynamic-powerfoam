"""High-precision circle/half-plane area and first moments for tiny faces.

Exceptional geometric reevaluation, not an area floor. Inputs are the exact
stored binary64 sites/radii. Arc and straight-edge Green integrals use 70
decimal digits; an empty face remains exactly empty. Used only by the opt-in
vacuum-boundary geometry when the ordinary disk integral is cancellation-sized.
"""
import math
import mpmath as mp


def disk_face(points,radii,owner,other,neighbours,ground,dps=70):
    with mp.workdps(dps):
        vec=lambda row:[mp.mpf(float(x)) for x in row]
        dot=lambda a,b:sum(x*y for x,y in zip(a,b))
        sub=lambda a,b:[x-y for x,y in zip(a,b)]
        add=lambda a,b:[x+y for x,y in zip(a,b)]
        mul=lambda a,k:[x*k for x in a]
        cross=lambda a,b:[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        p=vec(points[owner]);ri=mp.mpf(float(radii[owner]))
        q=vec(points[other]);rj=mp.mpf(float(radii[other]));delta=sub(q,p)
        length=mp.sqrt(dot(delta,delta));normal=mul(delta,1/length)
        hi=(length**2+ri**2-rj**2)/(2*length);center=add(p,mul(normal,hi))
        rho2=ri**2-hi**2
        if rho2<=0:return {'area':0.,'arcs':[]}
        rho=mp.sqrt(rho2);axis=[mp.mpf(0)]*3
        axis[min(range(3),key=lambda k:abs(normal[k]))]=mp.mpf(1)
        u=cross(axis,normal);u=mul(u,1/mp.sqrt(dot(u,u)));v=cross(normal,u)
        planes=[]
        for k in neighbours:
            if k<0 or k==owner or k==other:continue
            d=sub(vec(points[k]),p);rk=mp.mpf(float(radii[k]))
            a=[2*dot(d,u),2*dot(d,v)]
            b=dot(d,d)+ri**2-rk**2-2*dot(d,sub(center,p))
            planes.append((a,b))
        if ground is not None:planes.append(([-u[1],-v[1]],center[1]-mp.mpf(float(ground))))
        normalized=[];tiny=mp.mpf(10)**(-dps+12)
        for a,b in planes:
            norm=mp.sqrt(dot(a,a))
            if norm==0:
                if b<0:return {'area':0.,'arcs':[]}
                continue
            a=mul(a,1/norm);b/=norm
            if b<=-rho:return {'area':0.,'arcs':[]}
            if b>=rho:continue
            duplicate=next((k for k,(aa,bb) in enumerate(normalized) if max(abs(a[j]-aa[j]) for j in range(2))<tiny),None)
            if duplicate is None:normalized.append((a,b))
            elif b<normalized[duplicate][1]:normalized[duplicate]=(a,b)
        angles=[mp.mpf(0),2*mp.pi]
        for a,b in normalized:
            phase=mp.atan2(a[1],a[0]);half=mp.acos(b/rho)
            angles.extend([(phase-half)%(2*mp.pi),(phase+half)%(2*mp.pi)])
        angles=sorted(angles);arcs=[];area=mp.mpf(0);mx=mp.mpf(0);my=mp.mpf(0)
        for lo,hi_arc in zip(angles[:-1],angles[1:]):
            if hi_arc-lo<=tiny:continue
            mid=(lo+hi_arc)/2;point=[rho*mp.cos(mid),rho*mp.sin(mid)]
            if any(dot(a,point)>b+tiny for a,b in normalized):continue
            arcs.append([float(lo),float(hi_arc)])
            area+=rho2*(hi_arc-lo)/2
            mx+=rho**3*(mp.sin(hi_arc)-mp.sin(lo))/3
            my+=rho**3*(mp.cos(lo)-mp.cos(hi_arc))/3
        for k,(a,b) in enumerate(normalized):
            foot=mul(a,b);direction=[-a[1],a[0]]
            extent=mp.sqrt(max(mp.mpf(0),rho2-b*b));lo=-extent;high=extent
            for j,(aa,bb) in enumerate(normalized):
                if j==k:continue
                coefficient=dot(aa,direction);rhs=bb-dot(aa,foot)
                if abs(coefficient)<tiny:
                    if rhs < -tiny:high=lo-1;break
                elif coefficient>0:high=min(high,rhs/coefficient)
                else:lo=max(lo,rhs/coefficient)
            if high<=lo:continue
            p0=add(foot,mul(direction,lo));p1=add(foot,mul(direction,high))
            determinant=p0[0]*p1[1]-p0[1]*p1[0]
            area+=determinant/2;mx+=determinant*(p0[0]+p1[0])/6;my+=determinant*(p0[1]+p1[1])/6
        if area<0:
            if abs(area)>tiny*rho2:raise ArithmeticError('negative high-precision disk integral')
            area=mp.mpf(0)
        centroid=add(center,add(mul(u,mx/area),mul(v,my/area))) if area>0 else center
        return {'area':float(area),'centroid':[float(x) for x in centroid],
                'center':[float(x) for x in center],'normal':[float(x) for x in normal],
                'u':[float(x) for x in u],'v':[float(x) for x in v],
                'rho':float(rho),'hi':float(hi),'hj':float(length-hi),'arcs':arcs,
                'digits':dps}
