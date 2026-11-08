"""Incrementally grow the dual cut-hole boundary, preserving legacy edge order.

The removed dual triangles form a disk for a nondegenerate convex cut. Start
with one triangle and attach only along one edge with a new third vertex, or
along two edges. The nxt array maps a boundary plane to its successor. This
avoids comparing each removed edge to every other removed triangle.

An unsupported/degenerate attachment returns -1; the caller uses the original
exhaustive boundary search. rem==2 is temporary and restored before returning.
No geometric predicates, precision, clipping order, or output order change.
"""
import warp as wp


@wp.func
def local_boundary(i:int,nt:int,nplane:int,nrem:int,
                   T:wp.array2d(dtype=wp.vec3i),rem:wp.array2d(dtype=int),
                   nxt:wp.array2d(dtype=int),ea:wp.array2d(dtype=int),eb:wp.array2d(dtype=int)):
    for k in range(nplane):
        nxt[i,k]=-1
    left=nrem
    seeded=int(0)
    for t in range(nt):
        if rem[i,t]==1 and seeded==0:
            tr=T[i,t]
            nxt[i,tr[0]]=tr[1]
            nxt[i,tr[1]]=tr[2]
            nxt[i,tr[2]]=tr[0]
            rem[i,t]=2
            left-=1
            seeded=1
    while left>0:
        progress=int(0)
        for t in range(nt):
            if rem[i,t]==1:
                tr=T[i,t]
                a=tr[0];b=tr[1];c=tr[2]
                ab=int(nxt[i,b]==a)
                bc=int(nxt[i,c]==b)
                ca=int(nxt[i,a]==c)
                hits=ab+bc+ca
                attach=int(hits==2)
                if hits==1:
                    if (ab==1 and nxt[i,c]<0) or (bc==1 and nxt[i,a]<0) or (ca==1 and nxt[i,b]<0):
                        attach=1
                if attach==1:
                    # Remove reversed shared edges before inserting exposed
                    # edges; cached flags make this independent of write order.
                    if ab==1:nxt[i,b]=-1
                    if bc==1:nxt[i,c]=-1
                    if ca==1:nxt[i,a]=-1
                    if ab==0:nxt[i,a]=b
                    if bc==0:nxt[i,b]=c
                    if ca==0:nxt[i,c]=a
                    rem[i,t]=2
                    left-=1
                    progress+=1
        if progress==0:
            break
    nb=int(0)
    # Emit surviving edges in the same triangle/edge order as the reference.
    for t in range(nt):
        if rem[i,t]==2:rem[i,t]=1
        if rem[i,t]==1 and left==0:
            tr=T[i,t]
            for e in range(3):
                x=tr[e];y=tr[(e+1)%3]
                if nxt[i,x]==y:
                    ea[i,nb]=x;eb[i,nb]=y;nb+=1
    if left!=0:
        nb=-1
    return nb
