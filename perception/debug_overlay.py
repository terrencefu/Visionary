"""Display-only boxes in undistorted, unrotated image coordinates."""
import cv2
import numpy as np
from perception.stl_matcher import silhouette


def placement_overlay(frame, region, *, pose=None, K=None, model=None,
                      expected_xyz=None, expected_yaw=0., fitted=None, base_z=0.):
    """Keep pixel boxes separate from metric model poses; never infer depth from a box."""
    view = frame.copy()
    rows = [('UNDISTORTED / UNROTATED | image pixels (u,v)', (255,255,255))]
    def layer(mask, name, color):
        points = cv2.findNonZero(mask)
        if points is None:
            return
        x,y,w,h = cv2.boundingRect(points)
        right,bottom = x+w-1,y+h-1
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view,contours,-1,color,1)
        cv2.rectangle(view,(x,y),(right,bottom),color,2)
        label_row = {'OBSERVED':0,'EXPECTED':1,'FITTED':2}[name]
        label_y = y-6-18*label_row
        if label_y < 16:
            label_y = min(frame.shape[0]-4,bottom+18+18*label_row)
        cv2.putText(view,name,(x,label_y),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
        rows.append((f'{name}: TL=({x},{y}) BR=({right},{bottom}) px; W/H={w}/{h} px',color))
    if region is not None:
        layer(region.mask,'OBSERVED',(0,255,255))
        u,v = region.centroid_px
        cv2.drawMarker(view,(round(u),round(v)),(0,255,255),cv2.MARKER_CROSS,14,2)
        rows.append((f'Mask centroid: u={u:.1f}, v={v:.1f} px (not a physical part center)',(0,255,255)))
    else:
        rows.append(('No accepted region', (0,255,255)))
    if expected_xyz is not None:
        xyz = np.asarray(expected_xyz)
        layer(silhouette(model,expected_yaw,xyz[:2],pose,K,frame.shape[:2],base_z=xyz[2]),
              'EXPECTED',(0,255,0))
        rows.append((f'Expected board center: X={xyz[0]:.2f} Y={xyz[1]:.2f} Z={xyz[2]:.2f} mm; yaw={expected_yaw:.1f} deg',(0,255,0)))
    if fitted is not None:
        xy = np.asarray(fitted['xy_mm'])
        yaw = fitted['yaw_deg']
        layer(silhouette(model,yaw,xy,pose,K,frame.shape[:2],base_z=base_z),'FITTED',(0,0,255))
        rows.append((f'Fitted board center: X={xy[0]:.2f} Y={xy[1]:.2f} mm; assumed Z={base_z:.2f}; yaw={yaw:.1f} deg',(0,0,255)))
        if expected_xyz is not None:
            delta = np.asarray(expected_xyz)[:2]-xy
            rows.append((f'Correction: dX={delta[0]:+.2f} dY={delta[1]:+.2f} mm; XY error={np.linalg.norm(delta):.2f} mm',(255,255,255)))
    panel_width = max(frame.shape[1], max(cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,.55,1)[0][0] for t,_ in rows)+24)
    canvas = np.zeros((frame.shape[0]+28*len(rows)+12,panel_width,3),np.uint8)
    canvas[:frame.shape[0],:frame.shape[1]] = view
    for i,(text,color) in enumerate(rows):
        cv2.putText(canvas,text,(12,frame.shape[0]+24+28*i),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
    return canvas
