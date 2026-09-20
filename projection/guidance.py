"""Live board guidance through the calibrated projector model and current RAW pose."""
import argparse
import time
import cv2
import numpy as np
import config
from calibration.common import inside_board,load_camera,run_cli
from hardware.camera import Camera,show_preview
from hardware.projector import Projector
from perception.aruco import BoardTracker
from projection.world import load_projector,board_point_to_projector


def footprint_scene(corners,part_id,center):
    if not all(inside_board(p) for p in corners):
        raise ValueError('Footprint is outside BOARD_BOUNDS_MM.')
    x,y=center
    radius=min(2.,np.min(np.linalg.norm(corners-np.roll(corners,1,axis=0),axis=1))/4)
    edges=[(a,b,(0,255,0)) for a,b in zip(corners,np.roll(corners,-1,axis=0))]
    edges += [((x-radius,y),(x+radius,y),(0,255,0)),((x,y-radius),(x,y+radius),(0,255,0))]
    return edges,[(np.asarray(center),part_id,(0,255,0))]


def board_scene():
    points=[config.MARKER_CENTERS_MM[i] for i in (0,1,3,2)]
    edges=[(a,b,(0,0,255)) for a,b in zip(points,points[1:]+points[:1])]
    labels=[]
    for i,(x,y) in config.MARKER_CENTERS_MM.items():
        edges += [((x-3,y),(x+3,y),(0,0,255)),((x,y-3),(x,y+3),(0,0,255))]
        labels.append(((x,y),f'ID {i}',(0,0,255)))
    x0,y0,x1,y1=config.BOARD_BOUNDS_MM
    points=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    edges += [(a,b,(0,255,0)) for a,b in zip(points,points[1:]+points[:1])]
    return edges,labels


def render_scene(scene,pose,calibration):
    w,h=config.PROJECTOR_SIZE
    frame=np.zeros((h,w,3),np.uint8)
    def project(xy):
        point = [*xy,0.] if len(xy) == 2 else xy
        uv=board_point_to_projector(point,pose.rvec,pose.tvec,*calibration)
        if not np.isfinite(uv).all() or np.max(np.abs(uv))>1e7:
            raise ValueError('Invalid projector coordinates.')
        return uv
    # Sample board-space edges: projector distortion can curve projected lines.
    all_pixels=[]
    for a,b,color in scene[0]:
        a,b=np.asarray(a),np.asarray(b)
        points=np.array([project(p) for p in np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/2)+1))])
        if np.any(points<0) or np.any(points>=[w,h]):
            raise ValueError('Guidance outside projector image; reposition the rigid head.')
        all_pixels.extend(points)
        cv2.polylines(frame,[np.rint(points).astype(np.int32)],False,color,2,cv2.LINE_AA)
    for xy,text,color in scene[1]:
        uv=project(xy)
        # Label beside the footprint/center in screen space, without changing geometry.
        if len(scene[1])==1 and all_pixels:
            uv=np.max(all_pixels,axis=0)+[8,0]
        anchor=np.rint(uv+[6,16]).astype(int)
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
        anchor=np.clip(anchor,[0,th],[max(0,w-tw-1),h-3])
        cv2.putText(frame,text,tuple(anchor),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    return frame


def run_guidance(scene):
    calibration=load_projector()
    tracker=BoardTracker(*load_camera(),max_rms_px=config.COLLECTOR_MAX_ARUCO_RMS_PX)
    print(f'3D guidance: {config.PROJECTOR_CALIBRATION}. Live pose updates; rigid-head movement supported.')
    print('Camera and projector must remain rigid relative to each other. Space: start; Esc/Q: exit. No servo commands.')
    active=False; missing=None; last_message=None
    with Projector() as projector,Camera() as camera:
        while True:
            raw=camera.read()
            pose=tracker.estimate(raw)
            status='READY - Space to project' if pose is not None else 'Tracking recovery - projector blank'
            if active and pose is not None:
                missing=None
                try:
                    frame=render_scene(scene,pose,calibration)
                    cv2.imshow(projector.name,frame)
                    status='3D guidance active'
                except ValueError as exc:
                    projector.black()
                    status=str(exc)
                    if status!=last_message: print(status)
            else:
                projector.black()
                if active:
                    if missing is None: missing=time.monotonic()
                    if time.monotonic()-missing>=2:
                        raise ValueError('Tracking did not recover within 2 seconds; projection stopped.')
            last_message=status
            show_preview(raw,status)
            key=cv2.waitKey(1)&0xff
            if key in (27,ord('q')):return
            if key==32 and pose is not None:active=True


def main():
    parser=argparse.ArgumentParser(description='3D marker-center and workspace alignment overlay.')
    parser.parse_args()
    print('Red: marker-center outline (not cardboard edge). Green: usable workspace.')
    run_guidance(board_scene())


if __name__=='__main__':
    run_cli(main)
