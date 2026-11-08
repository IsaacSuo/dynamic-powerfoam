"""Independent analytic checks of the exceptional circle integral."""
import json
import math
from pathlib import Path
import mpmath as mp
from free_surface.fv_ale_precise_disk import disk_face


def main():
    # Equal balls centered at +/- x: their shared disk has radius one.
    # Ground clips this disk by y>=h; use exact stored doubles in the reference.
    p=[[-1.,0.,0.],[1.,0.,0.]];r=[math.sqrt(2.)]*2
    results=[]
    for h in [None,0.,.9,1.-1e-8]:
        actual=disk_face(p,r,0,1,[],h)
        with mp.workdps(90):
            radius=mp.sqrt(mp.mpf(r[0])**2-1)
            if h is None:area=mp.pi*radius**2;cy=mp.mpf(0)
            else:
                y=mp.mpf(h);a=mp.sqrt(radius**2-y*y)
                area=radius**2*mp.acos(y/radius)-y*a
                cy=mp.mpf(2)*a**3/(3*area)
            error=abs(actual['area']/float(area)-1)
            centroid_error=abs(actual['centroid'][1]-float(cy))
        assert error<2e-14 and centroid_error<2e-14,(h,error,centroid_error)
        results.append({'ground':h,'area':actual['area'],'area_relative_error':error,'centroid_error':centroid_error})
    assert disk_face(p,r,0,1,[],2.)['area']==0
    result={'passed':True,'analytic_disks':results}
    Path('output/ale/hyperelastic_precise_disk_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
