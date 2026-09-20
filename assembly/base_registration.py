"""Flat moving-base registration, independent of placement acceptance."""
import cv2
import numpy as np
import config
from perception.aruco import make_detector
from perception.geometry import pose_matrix
from perception.moving_base import solve_marker


def planar_marker_board(marker_camera, board_pose):
    relative = np.linalg.inv(pose_matrix(board_pose.rvec,board_pose.tvec)) @ marker_camera
    tilt = np.degrees(np.arccos(np.clip(relative[2,2],-1,1)))
    if abs(relative[2,3])>5 or tilt>15:
        raise ValueError('Set the moving base flat on the board before continuing')
    # Flat-base MVP: constrain depth/tilt to the fixed board plane. Single-marker
    # depth noise must not tilt the whole CAD assembly or change support heights.
    yaw = np.arctan2(relative[1,0],relative[0,0])
    c,s = np.cos(yaw),np.sin(yaw)
    result = np.eye(4)
    result[:2,:2] = [[c,-s],[s,c]]
    result[:2,3] = relative[:2,3]
    return result


class MovingRegistration:
    def __init__(self, marker_id, size, K, dist):
        self.detector = make_detector()
        count = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,config.ARUCO_DICTIONARY)).bytesList.shape[0]
        if marker_id in config.MARKER_CENTERS_MM or not 0<=marker_id<count or not np.isfinite(size) or size<=0:
            raise ValueError('Moving marker needs a unique ID outside 0-3 and a positive measured size')
        self.marker_id,self.size,self.K,self.dist = marker_id,size,K,dist
        self.marker_from_cad = None
        self.current = None
        self.reference_corners = None
        self.invalidated = False
        self.adapt_motion = False
        self.reason = 'Moving marker not acquired'

    def observe(self,raw):
        self.current = None
        corners,ids,_ = self.detector.detectMarkers(raw)
        indices = [] if ids is None else np.flatnonzero(ids.ravel()==self.marker_id)
        if len(indices)!=1:
            self.reason = f'Need exactly one visible moving marker ID {self.marker_id}'
            return
        try:
            self.current = solve_marker(corners[indices[0]].reshape(4,2),self.size,self.K,self.dist,config.MAX_ARUCO_RMS_PX)
            if not self.adapt_motion and self.reference_corners is not None and np.max(np.linalg.norm(self.current.corners-self.reference_corners,axis=1))>config.MOVING_BASE_MAX_DRIFT_PX:
                self.invalidated = True
            self.reason = 'Moving marker tracked'
        except (ValueError,cv2.error) as exc:
            self.reason = str(exc)

    def bind(self,board_pose,cad_board):
        if self.current is None:
            raise ValueError(self.reason)
        self.marker_from_cad = np.linalg.inv(planar_marker_board(self.current.matrix,board_pose))@cad_board

    def transform(self,board_pose):
        if self.current is None or self.marker_from_cad is None:
            raise ValueError(self.reason)
        return planar_marker_board(self.current.matrix,board_pose)@self.marker_from_cad

    def begin_move(self):
        self.reference_corners = None
        self.invalidated = False

    def arm(self):
        if self.current is None:
            raise ValueError(self.reason)
        self.reference_corners = self.current.corners.copy()
        self.invalidated = False

    def marker_quad(self):
        if self.current is None:
            return []
        return [cv2.undistortPoints(self.current.corners.reshape(-1,1,2),self.K,self.dist,P=self.K).reshape(4,2)]
