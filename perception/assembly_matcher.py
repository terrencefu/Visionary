"""Local CAD hypotheses with camera-depth occlusion, for coloured additions.

Frames and K are already undistorted. Scores are evidence, not probabilities.
No new height is inferred and no registration/calibration is changed.
"""
import cv2
import numpy as np
import config
from perception.colour_change import colour_mask, part_colour
from perception.cad_colour import matches_colour, colour_label


def depth_image(triangles, pose, K, shape):
    """Rasterize triangles with perspective-correct camera Z and a z-buffer."""
    points = np.asarray(triangles).reshape(-1,3,3)
    camera = points @ cv2.Rodrigues(pose.rvec)[0].T + np.asarray(pose.tvec).reshape(3)
    if not np.isfinite(camera).all() or np.any(camera[...,2] <= 0):
        raise ValueError('CAD crosses camera plane')
    pixel = camera @ K.T
    pixel = pixel[...,:2]/pixel[...,2:]
    depth = np.full(shape,np.inf,np.float32)
    lows = np.maximum(np.ceil(pixel.min(axis=1)).astype(int),0)
    highs = np.minimum(np.floor(pixel.max(axis=1)).astype(int),[shape[1]-1,shape[0]-1])
    ab,ac = pixel[:,1]-pixel[:,0],pixel[:,2]-pixel[:,0]
    area = ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    selected = np.flatnonzero(np.all(lows<=highs,axis=1)&(np.abs(area)>1e-9))
    for index in selected:
        uv,z,lo,hi = pixel[index],camera[index,:,2],lows[index],highs[index]
        a,b,c = uv
        den = area[index]
        yy,xx = np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        u = ((xx-a[0])*(c[1]-a[1])-(yy-a[1])*(c[0]-a[0]))/den
        v = ((b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0]))/den
        inside = (u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)
        inv = (1-u-v)/z[0]+u/z[1]+v/z[2]
        valid = inside & (inv>0)
        patch = depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        np.minimum(patch,np.where(valid,1/np.maximum(inv,1e-30),np.inf),out=patch)
    return depth


def check_placement(assembly, frame, region, pose, K, colour, search_mask=None):
    """Compare expected pose and bounded nearby offsets; fail closed on ambiguity."""
    index = assembly.index
    mesh = assembly.meshes[index]
    T = assembly.transform
    target = mesh @ T[:3,:3].T + T[:3,3]
    center_root = (mesh.min(axis=(0,1))+mesh.max(axis=(0,1)))/2
    center_root[2] = mesh[...,2].min()
    center = T[:3,:3] @ center_root+T[:3,3]
    yaw = float(np.degrees(np.arctan2(T[1,0],T[0,0]))%360)
    # Crop around both CAD and detection, with room for the bounded search.
    uv = cv2.projectPoints(target.reshape(-1,3),pose.rvec,pose.tvec,K,np.zeros(5))[0].reshape(-1,2)
    x,y,w,h = region.bbox
    lo = np.maximum(np.floor(np.minimum(uv.min(axis=0),[x,y])-100).astype(int),0)
    hi = np.minimum(np.ceil(np.maximum(uv.max(axis=0),[x+w,y+h])+100).astype(int),[frame.shape[1],frame.shape[0]])
    if np.any(hi<=lo):
        raise ValueError('Expected target outside camera view')
    x0,y0 = lo; x1,y1 = hi
    scale = min(1.,400/max(hi-lo))
    observed = cv2.resize(colour_mask(frame,colour)[y0:y1,x0:x1],None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)>0
    novelty = cv2.resize(region.mask[y0:y1,x0:x1],(observed.shape[1],observed.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    valid = np.ones(observed.shape,bool) if search_mask is None else cv2.resize(
        search_mask[y0:y1,x0:x1],(observed.shape[1],observed.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    crop_K = np.array([[scale,0,-scale*x0],[0,scale,-scale*y0],[0,0,1]])@K
    installed = np.full(observed.shape,np.inf,np.float32)
    installed_colour = np.zeros(observed.shape,bool)
    profiles = getattr(assembly,'colours',[part_colour(name) for name in assembly.names])
    for profile,part in zip(profiles[:index],assembly.meshes[:index]):
        z = depth_image(part@T[:3,:3].T+T[:3,3],pose,crop_K,observed.shape)
        nearer = z<installed
        installed_colour[nearer] = matches_colour(profile,colour)
        installed = np.minimum(installed,z)
    # Score over a fixed neighbourhood, so a candidate cannot win by hiding
    # troublesome observed pixels or shrinking its comparison window.
    expected_depth = depth_image(target,pose,crop_K,observed.shape)
    roi = cv2.dilate(np.isfinite(expected_depth).astype(np.uint8),np.ones((61,61),np.uint8))>0
    valid &= roi
    cache = {}
    def evaluate(params):
        key = tuple(float(v) for v in params)
        if key in cache:
            return cache[key]
        dx,dy,angle = key
        theta = np.radians(angle); c,s = np.cos(theta),np.sin(theta)
        R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        moved = (target-center)@R.T+center+[dx,dy,0]
        z = expected_depth if key==(0.,0.,0.) else depth_image(moved,pose,crop_K,observed.shape)
        full = np.isfinite(z)
        visible = full & (z<installed-0.05) & valid
        predicted = (visible | (installed_colour & ~(full & (z<installed-0.05)))) & valid
        union = np.count_nonzero(predicted | (observed & valid))
        score = np.count_nonzero(predicted & observed)/max(1,union)
        visible_count = np.count_nonzero(visible)
        recall = np.count_nonzero(visible & observed)/max(1,visible_count)
        new_support = np.count_nonzero(visible & novelty)/max(1,visible_count)
        enough = (visible_count>=config.ASSEMBLY_MIN_VISIBLE_PIXELS
                  and visible_count/max(1,np.count_nonzero(full))>=config.ASSEMBLY_MIN_VISIBLE_FRACTION
                  and new_support>=config.ASSEMBLY_MIN_NEW_SUPPORT)
        result = dict(offset=list(key),score=float(score),visible_pixels=int(visible_count),
                      visible_fraction=float(visible_count/max(1,np.count_nonzero(full))),
                      colour_coverage=float(recall),new_support=float(new_support),enough_visibility=bool(enough))
        cache[key] = result
        return result
    expected = evaluate((0,0,0))
    # Deterministic local search. Far-away or boundary winners are uncertain,
    # not a claim to have recovered an arbitrary global part pose.
    xy_limit,angle_limit=config.ASSEMBLY_SEARCH_XY_MM,config.ASSEMBLY_SEARCH_YAW_DEG
    if not np.isfinite([xy_limit,angle_limit]).all() or min(xy_limit,angle_limit)<=0:
        raise ValueError('Assembly search limits must be finite and positive')
    for axis,limit in enumerate((xy_limit,xy_limit,angle_limit)):
        steps=[fraction*limit for fraction in (-.8,-.4,.4,.8)]
        for step in steps:
            params = [0,0,0]; params[axis]=step; evaluate(params)
    best = max(cache.values(),key=lambda r:r['score'])
    for mm,deg in ((xy_limit*.2,angle_limit*.2),(xy_limit*.1,angle_limit*.1)):
        for _ in range(2):
            start = best
            for axis,step in enumerate((mm,mm,deg)):
                for sign in (-1,1):
                    params = list(start['offset']); params[axis]+=sign*step
                    if abs(params[0])<=xy_limit and abs(params[1])<=xy_limit and abs(params[2])<=angle_limit:
                        candidate = evaluate(params)
                        if candidate['score']>best['score']:
                            best = candidate
            if best is start:
                break
    def outside(r):
        dx,dy,angle = r['offset']
        return np.hypot(dx,dy)>config.PLACEMENT_TOLERANCE_MM or abs(angle)>config.PLACEMENT_TOLERANCE_DEG
    # Placement ambiguity is separate from isolated-part identity confidence.
    strong = (best['enough_visibility'] and best['score']>=config.ANCHOR_MIN_OVERLAP
              and best['colour_coverage']>=config.ANCHOR_MIN_OVERLAP)
    alternatives = [r for r in cache.values() if outside(r)!=outside(best) and r['enough_visibility']]
    margin = config.PLACEMENT_AMBIGUITY_MARGIN
    if not np.isfinite(margin) or not 0<=margin<=1:
        raise ValueError('PLACEMENT_AMBIGUITY_MARGIN must be finite and between 0 and 1')
    gap = best['score']-max(r['score'] for r in alternatives) if alternatives else None
    ambiguous = gap is not None and gap<margin
    boundary = max(abs(best['offset'][0]),abs(best['offset'][1]))>=xy_limit or abs(best['offset'][2])>=angle_limit
    passed = bool(strong and not outside(best) and not ambiguous and not boundary)
    status = 'PLACEMENT OK (visible evidence)' if passed else 'INSUFFICIENT EVIDENCE'
    if strong and outside(best) and not ambiguous and not boundary:
        status = 'ADJUST PLACEMENT'
    dx,dy,angle = best['offset']
    gap_text = f'{gap:.3f}' if gap is not None else 'n/a'
    message = (f'{status}: expected score={expected["score"]:.3f}; best={best["score"]:.3f}; '
               f'visible={best["visible_fraction"]:.0%}; new support={best["new_support"]:.0%}; '
               f'detected colour={colour_label(colour)}; ambiguity margin={margin:.3f}; '
               f'gap={gap_text}')
    if status=='ADJUST PLACEMENT':
        message += f'; correction X={-dx:+.2f}, Y={-dy:+.2f} mm, yaw={-angle:+.1f} deg'
    reasons = []
    if not best['enough_visibility']:
        reasons.append('too little visible/new target evidence')
    if best['score']<config.ANCHOR_MIN_OVERLAP:
        reasons.append(f'model agreement below {config.ANCHOR_MIN_OVERLAP:.3f}')
    if best['colour_coverage']<config.ANCHOR_MIN_OVERLAP:
        reasons.append('predicted visible target lacks expected colour')
    if ambiguous:
        reasons.append('correct and displaced hypotheses are too similar')
    if boundary:
        reasons.append('best candidate reaches local search boundary')
    if status=='INSUFFICIENT EVIDENCE':
        message += '; '+ '; '.join(reasons)
    check = dict(component=assembly.names[index],part=assembly.operations[index]['part'],step_index=index,
                 expected_xyz=center.tolist(),expected_yaw=yaw,initial_yaw=yaw,
                 cad_to_board=T.tolist(),colour_verified=colour,method='assembly_visible_hypotheses',
                 assembly_colours=profiles,
                 expected_hypothesis=expected,best_hypothesis=best,hypotheses=list(cache.values()),
                 ambiguity_margin=float(margin),ambiguity_gap=gap,
                 uncertainty_reasons=reasons,scores=[],fit_rejection=None if passed else message)
    # A low-confidence candidate must not appear as a trusted measured center.
    if strong and not ambiguous and not boundary:
        check.update(measured_xy=(center[:2]+[dx,dy]).tolist(),measured_yaw=(yaw+angle)%360,
                     fitted_overlap=best['score'])
    debug = frame.copy()
    comparison = np.zeros((*observed.shape,3),np.uint8)
    comparison[...,1] = (np.isfinite(expected_depth)&(expected_depth<installed-.05)&valid)*255
    comparison[...,2] = (observed&valid)*255
    return passed,message,debug,check,comparison
