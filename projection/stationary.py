"""Filtered placement-only stationary check, always anchored to saved calibration."""
from collections import deque
import cv2
import numpy as np


class StationaryMonitor:
    def __init__(self, reference, K, dist, limit=1.5, window=9, persist=.5, recovery=2.):
        if not np.isfinite([limit,persist,recovery]).all() or min(limit,persist,recovery) <= 0 or window < 3:
            raise ValueError('Movement settings must be positive; window must be >=3.')
        self.reference,self.K,self.dist = reference,K,dist
        self.limit,self.window,self.persist,self.recovery = limit,window,persist,recovery
        self.samples=deque(maxlen=window)
        self.history=deque(maxlen=30)
        self.base=None
        self.bad_since=None
        self.missing_since=None
        self.reference_pixels=self.project(reference)

    def project(self,pose):
        return cv2.projectPoints(self.reference.object_points,pose.rvec,pose.tvec,self.K,self.dist)[0].reshape(-1,2)

    def update(self,pose,now):
        info=dict(ready=False,stop=False,status='TRACKING RECOVERY',limit_px=self.limit,
                  raw_drift_px=None,filtered_drift_px=None,jitter_px=None,baseline_drift_px=None)
        if pose is None:
            self.samples.clear()
            self.bad_since=None
            if self.missing_since is None: self.missing_since=now
            info['stop']=now-self.missing_since >= self.recovery
            return info
        self.missing_since=None
        pixels=self.project(pose)
        raw=float(np.max(np.linalg.norm(pixels-self.reference_pixels,axis=1)))
        self.history.append(raw)
        self.samples.append((now,pixels))
        R0=cv2.Rodrigues(self.reference.rvec)[0]
        R=cv2.Rodrigues(pose.rvec)[0]
        info.update(raw_drift_px=raw,history_px=list(self.history),
                    translation_delta_mm=(pose.tvec.reshape(3)-self.reference.tvec.reshape(3)).tolist(),
                    translation_norm_mm=float(np.linalg.norm(pose.tvec.reshape(3)-self.reference.tvec.reshape(3))),
                    rotation_delta_deg=float(np.degrees(np.arccos(np.clip((np.trace(R@R0.T)-1)/2,-1,1)))),
                    current_rvec=pose.rvec.reshape(3).tolist(),current_tvec=pose.tvec.reshape(3).tolist())
        if len(self.samples)<self.window:
            info['status']='ACQUIRING STABLE WINDOW'
            return info
        values=np.array([p for _,p in self.samples])
        median=np.median(values,axis=0)
        filtered=float(np.max(np.linalg.norm(median-self.reference_pixels,axis=1)))
        jitter=float(np.median(np.max(np.linalg.norm(values-median,axis=2),axis=1)))
        baseline=0. if self.base is None else float(np.max(np.linalg.norm(median-self.base,axis=1)))
        info.update(filtered_drift_px=filtered,jitter_px=jitter,baseline_drift_px=baseline)
        # Never absorb an offset from the saved pose into a new local reference.
        bad=max(filtered,baseline)>self.limit
        unstable=jitter>self.limit/2
        if bad or unstable:
            if self.bad_since is None: self.bad_since=now
            duration=now-self.bad_since
            info['status']='PERSISTENT POSE OFFSET (movement or systematic bias)' if bad else 'UNSTABLE POSE ESTIMATES'
            info['stop']=duration >= (self.persist if bad else self.recovery)
            info['duration_s']=duration
            return info
        self.bad_since=None
        if self.base is None: self.base=median.copy()
        info['ready']=raw<=self.limit
        info['status']='STABLE' if info['ready'] else 'ISOLATED RAW OUTLIER - BLANKED'
        return info
