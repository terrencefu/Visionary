"""Offline rendered-ArUco simulation. No camera, calibration file or serial port."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from perception.aruco import BoardTracker, marker_corners
from tracking.controller import FollowController, board_center_pixel
import config


def render_board(K, R, t, hidden=False):
    frame = np.full((540, 960, 3), 220, np.uint8)
    if hidden:
        return frame
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICTIONARY))
    rv = cv2.Rodrigues(R)[0]
    for marker_id in config.MARKER_CENTERS_MM:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
        quad = cv2.projectPoints(marker_corners(marker_id), rv, t, K, np.zeros(5))[0].reshape(4, 2)
        H = cv2.getPerspectiveTransform(np.float32([[0,0],[159,0],[159,159],[0,159]]), quad.astype(np.float32))
        ink = cv2.warpPerspective(marker, H, (960,540), borderValue=255)
        frame = np.minimum(frame, cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR))
    return frame


def run(output, wrong_sign=False):
    output.mkdir(parents=True, exist_ok=True)
    K = np.array([[800.,0,480.],[0,800.,270.],[0,0,1.]])
    tracker = BoardTracker(K, np.zeros(5))
    control = FollowController(pan_sign=-1 if wrong_sign else 1)
    actual = np.array([1500.,1000.])
    neutral = actual.copy()
    x0,y0,x1,y1 = config.BOARD_BOUNDS_MM
    center = np.array([(x0+x1)/2,(y0+y1)/2,0.])
    mount = np.diag([-1.,-1.,1.])  # upside-down camera, as on the real rig
    records, frames = [], []
    for i in range(500):
        now = i / 10
        # A deliberately simple plant: 0.002 rad/us, 0.18 s mechanical lag.
        # These are assumptions, NOT measurements of the user's servos.
        actual += (1-np.exp(-.1/.18))*(np.array([control.pan, control.tilt])-actual)
        angles = (actual-neutral)*.002
        if now >= 16:
            angles += [.12,-.06]  # base/head orientation disturbed
        pan,tilt = angles
        Ry = np.array([[np.cos(pan),0,-np.sin(pan)],[0,1,0],[np.sin(pan),0,np.cos(pan)]])
        Rx = np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]])
        camera_rotation = Rx @ Ry
        shift = np.array([150.,-80.,900.])
        if now >= 8:
            shift += [100.,50.,0.]
        if now >= 24:
            shift += [70.,-30.,0.]  # target also moves while hidden
        if now >= 30:
            shift += [-160.,80.,0.]
        R = camera_rotation @ mount
        t = camera_rotation @ shift - R @ center
        hidden = 24 <= now < 27
        raw = render_board(K,R,t,hidden)
        pose = tracker.estimate(raw)  # real detector + solvePnP + rejection gates
        target = board_center_pixel(pose,K,np.zeros(5),config.BOARD_BOUNDS_MM) if pose else None
        if target is not None and (np.any(target<0) or np.any(target >= [960,540])):
            target=None
        commands = control.update(target, [480,270], now)
        truth = cv2.projectPoints(center.reshape(1,3),cv2.Rodrigues(R)[0],t,K,np.zeros(5))[0].reshape(2)
        error = truth-[480,270]
        records.append(dict(time=now,error_x=float(error[0]),error_y=float(error[1]),
                            detected=target is not None,hidden=hidden,commands=len(commands),
                            pan=control.pan,tilt=control.tilt,status=control.status))
        if i % 5 == 0:
            cv2.drawMarker(raw,(480,270),(255,170,0),cv2.MARKER_CROSS,28,2)
            cv2.rectangle(raw,(455,245),(505,295),(255,170,0),1)
            if target is not None:
                cv2.circle(raw,tuple(np.rint(target).astype(int)),7,(0,180,0),2)
            event = 'INITIAL OFFSET' if now<8 else 'BOARD MOVED' if now<16 else 'RIG BUMPED' if now<24 else 'MARKERS HIDDEN' if now<27 else 'REACQUIRED' if now<30 else 'BOARD MOVED AGAIN'
            cv2.putText(raw,f'{now:04.1f}s  {event}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),2)
            cv2.putText(raw,f'{control.status} | P {control.pan} T {control.tilt}',(15,515),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),1)
            frames.append(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
    loss = [r for r in records if r['hidden']]
    final = records[-1]
    report = dict(scenario='wrong pan sign' if wrong_sign else 'nominal',
                  real_aruco_detection=True,real_controller=True,physical_hardware=False,
                  assumptions=dict(radians_per_us=.002,mechanical_time_constant_s=.18,fps=10),
                  final_error_px=[final['error_x'],final['error_y']],
                  final_inside_deadband=max(abs(final['error_x']),abs(final['error_y']))<=control.deadband,
                  commands_during_marker_loss=sum(r['commands'] for r in loss),
                  detected_frames=sum(r['detected'] for r in records),total_frames=len(records),
                  pulses_within_limits=all(400<=r['pan']<=2700 and 700<=r['tilt']<=1500 for r in records))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'trace.json').write_text(json.dumps(records,indent=2)+'\n')
    from PIL import Image
    images=[Image.fromarray(frame).resize((768,432)) for frame in frames]
    images[0].save(output/'simulation.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times=[r['time'] for r in records]
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for key,label in [('error_x','Horizontal error'),('error_y','Vertical error')]:
        axes[0].plot(times,[r[key] for r in records],label=label)
    axes[0].axhspan(-25,25,color='green',alpha=.12,label='25 px deadband')
    axes[0].set_ylabel('Centre error (pixels)');axes[0].legend(loc='upper right')
    for key in ['pan','tilt']:
        axes[1].plot(times,[r[key] for r in records],label=key)
    axes[1].set_ylabel('Command (microseconds)');axes[1].set_xlabel('Simulated time (seconds)');axes[1].legend()
    for ax in axes:
        ax.axvspan(24,27,color='grey',alpha=.2)
        for event in [8,16,30]:ax.axvline(event,color='grey',linestyle=':',linewidth=1)
        ax.grid(alpha=.2)
    axes[0].set_title('Rendered ArUco + actual tracking code; assumed servo dynamics\nBoard moves: 8/30 s | rig bump: 16 s | markers hidden: 24–27 s')
    fig.tight_layout();fig.savefig(output/'results.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))
    if report['commands_during_marker_loss'] or not report['pulses_within_limits']:
        raise RuntimeError('Simulation violated loss/limit behaviour; inspect report.')
    if not wrong_sign and not report['final_inside_deadband']:
        raise RuntimeError('Nominal simulation did not converge; inspect report.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/tracking_simulation'))
    parser.add_argument('--wrong-sign',action='store_true')
    args=parser.parse_args()
    run(args.output,args.wrong_sign)


if __name__=='__main__':main()
