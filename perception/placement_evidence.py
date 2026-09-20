"""Save a failed placement exactly; replay metric fitting without camera hardware."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import config
from perception.stl_matcher import silhouette
from perception.anchor import estimate_anchor
from perception.debug_overlay import placement_overlay


def save_check(assembly, before, after, region, pose, K, message, valid_mask=None):
    out = config.DATA_DIR / f'placement_check_{time.time_ns()}'
    out.mkdir(parents=True)
    report = dict(assembly.last_check)
    report.update(message=message, frame_convention='undistorted, unrotated camera pixels',
                  position_tolerance_mm=config.PLACEMENT_TOLERANCE_MM,
                  angle_tolerance_deg=config.PLACEMENT_TOLERANCE_DEG,
                  board_rms_px=float(pose.reprojection_error) if hasattr(pose,'reprojection_error') else None,
                  board_marker_ids=list(map(int,getattr(pose,'visible_ids',[]))))
    model = assembly.models[report['component']]
    xyz = report['expected_xyz']
    predicted = silhouette(model,report['expected_yaw'],xyz[:2],pose,K,region.mask.shape,base_z=xyz[2])
    union = np.count_nonzero(region.mask | predicted)
    report['expected_pose_full_frame_overlap'] = float(np.count_nonzero(region.mask & predicted)/max(1,union))
    fitted = (dict(xy_mm=report['measured_xy'],yaw_deg=report['measured_yaw'])
              if 'measured_xy' in report else None)
    overlay = placement_overlay(after,region,pose=pose,K=K,model=model,
                                expected_xyz=xyz,expected_yaw=report['expected_yaw'],
                                fitted=fitted,base_z=xyz[2])
    cv2.imwrite(str(out/'before_undistorted.png'),before)
    cv2.imwrite(str(out/'after_undistorted.png'),after)
    cv2.imwrite(str(out/'observed_mask.png'),region.mask)
    cv2.imwrite(str(out/'overlay_yellow_observed_green_expected_red_fit.png'),overlay)
    # Save the exact model and K/pose: replay does not depend on a later Fusion
    # re-export or modifications to the active camera calibration file.
    extra = {}
    if report.get('method')=='assembly_visible_hypotheses':
        meshes = assembly.meshes[:assembly.index+1]
        extra = dict(assembly_meshes=np.concatenate(meshes),mesh_counts=[len(m) for m in meshes],
                     assembly_names=assembly.names[:assembly.index+1],cad_to_board=assembly.transform,
                     valid_mask=np.full(region.mask.shape,255,np.uint8) if valid_mask is None else valid_mask)
    np.savez_compressed(out/'inputs.npz',model=model,K=K,rvec=pose.rvec,tvec=pose.tvec,
                        mask=region.mask,bbox=np.asarray(region.bbox),centroid=np.asarray(region.centroid_px),**extra)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Placement evidence saved: {out}')
    return out


def replay(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    with np.load(folder/'inputs.npz',allow_pickle=False) as saved:
        region = SimpleNamespace(mask=saved['mask'],bbox=tuple(saved['bbox']),centroid_px=saved['centroid'])
        pose = SimpleNamespace(rvec=saved['rvec'],tvec=saved['tvec'])
        if report.get('method')=='assembly_visible_hypotheses':
            from perception.assembly_matcher import check_placement
            meshes = np.split(saved['assembly_meshes'],np.cumsum(saved['mesh_counts'])[:-1])
            index = len(meshes)-1
            assembly = SimpleNamespace(index=index,meshes=meshes,names=saved['assembly_names'].tolist(),
                                       transform=saved['cad_to_board'],operations=[{'part':report['part']}]*len(meshes))
            frame = cv2.imread(str(folder/'after_undistorted.png'))
            result = check_placement(assembly,frame,region,pose,saved['K'],report['colour_verified'],saved['valid_mask'])[:2]
        else:
            initial = report.get('initial_yaw')
            if initial is None:
                initial = report['scores'][0][2]
            result = estimate_anchor(saved['model'],region,pose,saved['K'],initial,base_z=report['expected_xyz'][2])
    print('Saved check:',report['message'])
    print('Replayed fit:',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    replay(parser.parse_args().folder)
