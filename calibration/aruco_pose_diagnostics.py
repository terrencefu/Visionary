"""Independent diagnostic fits; never replace BoardTracker's production pose."""
import cv2
import numpy as np
import config
from perception.aruco import marker_corners

CORNER_NAMES = ('TL', 'TR', 'BR', 'BL')
SUBSET_IDS = (0, 1, 3)


def parse_rotations(values):
    """Only diagnostic overrides; positive angles turn board +X toward +Y."""
    rotations = {i: 0.0 for i in config.MARKER_CENTERS_MM}
    for value in values:
        try:
            key, angle = value.split('=')
            marker_id, degrees = int(key), float(angle)
        except (ValueError, TypeError):
            raise ValueError('Use --marker-rotation ID=DEGREES, for example 2=90.') from None
        if marker_id not in rotations or not np.isfinite(degrees):
            raise ValueError('Diagnostic rotations require a configured marker ID and finite degrees.')
        rotations[marker_id] = degrees
    return rotations


def local_corners():
    h = config.MARKER_SIZE_MM / 2
    # Board convention is X right, Y down. Do not use IPPE_SQUARE's different
    # required Y-up ordering with these points. ITERATIVE supports planar quads.
    return np.array([[-h, -h, 0.], [h, -h, 0.], [h, h, 0.], [-h, h, 0.]])


def physical_corners(marker_id, degrees):
    """Map decoded canonical corners 0..3 to the declared physical board frame."""
    if degrees == 0:
        return marker_corners(marker_id)  # Exactly the production mapping.
    angle = np.radians(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    return local_corners() @ R.T + np.r_[config.MARKER_CENTERS_MM[marker_id], 0.]


def residuals(observed, projected):
    errors = np.linalg.norm(projected-observed, axis=1)
    return dict(rms_px=float(np.sqrt(np.mean(errors**2))),
                mean_px=float(np.mean(errors)), max_px=float(np.max(errors)),
                corner_errors_px=errors.tolist(), projected_points=projected.tolist())


def fit_pose(objects, pixels, K, dist):
    """Return a diagnostic fit and its evidence, including failed checks."""
    result = dict(solvepnp_succeeded=None, rms_px=None, mean_px=None, max_px=None,
                  positive_depth=None, passes_checks=False, rejection_reason=None,
                  rvec=None, tvec=None, projected_points=None)
    try:
        ok, r, t = cv2.solvePnP(np.asarray(objects, dtype=float), np.asarray(pixels, dtype=float),
                                K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        result['solvepnp_succeeded'] = bool(ok)
        if not ok:
            result['rejection_reason'] = 'solvePnP returned False.'
            return result
        result.update(rvec=r.ravel().tolist(), tvec=t.ravel().tolist())
        projected = cv2.projectPoints(objects, r, t, K, dist)[0].reshape(-1, 2)
        result.update(residuals(pixels, projected))
        depths = (objects @ cv2.Rodrigues(r)[0].T + t.reshape(3))[:, 2]
        result['positive_depth'] = bool(np.isfinite(depths).all() and np.all(depths > 0))
        reasons = []
        if not np.isfinite(result['rms_px']):
            reasons.append('Non-finite reprojection RMS')
        elif result['rms_px'] > config.MAX_ARUCO_RMS_PX:
            reasons.append(f"RMS {result['rms_px']:.3f} > {config.MAX_ARUCO_RMS_PX:.3f} px")
        if not result['positive_depth']:
            reasons.append('Non-finite/non-positive corner depth')
        result['passes_checks'] = not reasons
        result['rejection_reason'] = '; '.join(reasons) or None
    except cv2.error as exc:
        result['rejection_reason'] = 'OpenCV error: ' + str(exc)
    return result


def combined_fit(ids, observations, rotations, K, dist, minimum):
    missing = [i for i in ids if i not in observations or i not in config.MARKER_CENTERS_MM]
    result = dict(ids=list(ids), available=False, missing_ids=missing, per_marker_residuals={},
                  solvepnp_succeeded=None, rms_px=None, passes_checks=False)
    if missing or len(ids) < minimum:
        result['rejection_reason'] = f'Requires IDs {list(ids)}; missing/ambiguous {missing}; minimum {minimum}.'
        return result
    objects = np.concatenate([physical_corners(i, rotations[i]) for i in ids])
    pixels = np.concatenate([observations[i] for i in ids])
    result.update(fit_pose(objects, pixels, K, dist), available=True)
    if result['solvepnp_succeeded'] and result['projected_points'] is not None:
        r, t = np.array(result['rvec']), np.array(result['tvec'])
        # Include held-out marker 2 in EVALUATION, never in the 0/1/3 fit.
        for i, observed in sorted(observations.items()):
            if i not in config.MARKER_CENTERS_MM:
                continue
            projected = cv2.projectPoints(physical_corners(i, rotations[i]), r, t, K, dist)[0].reshape(-1, 2)
            result['per_marker_residuals'][i] = dict(residuals(observed, projected), in_fit=i in ids)
    return result


def compare_poses(debug, K, dist, rotations, audit, display_fit='subset'):
    observations = {}
    duplicates = {i for i in debug.detected_ids if debug.detected_ids.count(i) > 1}
    individuals = []
    for index, (i, quad) in enumerate(zip(debug.detected_ids, debug.corners)):
        observed = np.asarray(quad, dtype=float).reshape(4, 2)
        fit = fit_pose(local_corners(), observed, K, dist)
        fit.update(id=i, detection_index=index, side_mm=config.MARKER_SIZE_MM)
        individuals.append(fit)
        if i not in duplicates:
            observations[i] = observed
    known_ids = sorted(i for i in observations if i in config.MARKER_CENTERS_MM)
    full = combined_fit(known_ids, observations, rotations, K, dist, config.MIN_VISIBLE_MARKERS)
    subset = combined_fit(SUBSET_IDS, observations, rotations, K, dist, len(SUBSET_IDS))
    mappings = {}
    for i in known_ids:
        declared = rotations[i]
        estimated = audit.get(i, {}).get('rotation_deg')
        difference = None if estimated is None else (estimated-declared+180) % 360-180
        quarter_turn = round(declared/90)
        physical_labels = ([CORNER_NAMES[(j+quarter_turn) % 4] for j in range(4)]
                           if abs(declared-quarter_turn*90) < 1e-6 else None)
        # Advisory only: do not round/reorder observations to improve residuals.
        orientation = 'unavailable: need center-based audit'
        if difference is not None:
            orientation = ('large orientation mismatch: inspect declared rotation/corner mapping'
                           if abs(difference) > 45 else
                           'no quarter-turn mismatch indicated; small angle/warp remains possible')
        mappings[i] = dict(declared_rotation_deg=declared, estimated_rotation_deg=estimated,
                           difference_deg=difference, physical_corner_labels=physical_labels,
                           board_corners_mm=physical_corners(i, declared).tolist(),
                           detected_corners_px=observations[i].tolist(), assessment=orientation)
    complete = bool(config.MARKER_CENTERS_MM) and set(config.MARKER_CENTERS_MM) <= set(known_ids)
    known_fits = [f for f in individuals if f['id'] in config.MARKER_CENTERS_MM]
    all_solve = all(f['solvepnp_succeeded'] for f in known_fits) if complete else None
    all_pass = all(f['passes_checks'] for f in known_fits) if complete else None
    contrast = (bool(all_pass and not full['passes_checks']) if complete and full['available'] else None)
    report = dict(individuals=individuals, full_board=full, subset_013=subset,
                  solver='SOLVEPNP_ITERATIVE', camera_matrix=np.asarray(K).tolist(),
                  dist_coeffs=np.asarray(dist).tolist(), marker_size_mm=config.MARKER_SIZE_MM,
                  marker_centers_mm={i: list(xy) for i, xy in config.MARKER_CENTERS_MM.items()},
                  corner_mapping=mappings, duplicate_ids=sorted(duplicates),
                  configured_rotations_deg=rotations, rms_limit_px=config.MAX_ARUCO_RMS_PX,
                  complete_fixture=complete, individual_solvers_succeeded=all_solve,
                  individuals_pass_checks=all_pass, individuals_pass_combined_fails=contrast,
                  production_board_rms_px=debug.rms_error_px,
                  production_rejection=debug.rejection_reason, display_fit=display_fit)
    report['interpretation'] = interpret(report)
    return report


def interpret(report):
    """Conditional conclusions, not a claim that a single frame proves flatness."""
    lines = []
    if not report['complete_fixture']:
        lines.append('Incomplete fixture: cannot compare all individual markers with the full board.')
    elif report['individuals_pass_combined_fails']:
        lines.append('Local square fits pass but the combined rigid planar-board fit is rejected.')
        if report['subset_013']['passes_checks']:
            lines.append('The 0/1/3 fit passes: supports a marker-2-specific geometry/height/orientation inconsistency.')
        elif report['subset_013']['available']:
            lines.append('The 0/1/3 fit also fails: removing marker 2 alone does not resolve the inconsistency.')
    elif report['individuals_pass_checks'] and report['full_board']['passes_checks']:
        lines.append('Individual and full-board fits pass in this frame; failure not reproduced.')
    elif report['complete_fixture']:
        lines.append('At least one individual square fit fails; inspect its corners, paper shape, blur and lens model too.')
    if any(abs(m['difference_deg']) > 45 for m in report['corner_mapping'].values() if m['difference_deg'] is not None):
        lines.append('Corner-direction audit flags a large rotation mismatch; verify printed rotations before attributing error to lift.')
    lines.append('Low single-marker RMS does not validate intrinsics, absolute pose or scale; each fit has only four corners.')
    lines.append('These fits cannot uniquely separate nonplanarity, small mounting rotations and lens-model error.')
    return lines


def comparison_lines(report, detailed=False):
    number = lambda x: 'n/a' if x is None else f'{x:.3f}'
    lines = ['DIAGNOSTIC FITS ONLY - production pose and gates unchanged.',
             f"Individual fits use 4 canonical corners and side {config.MARKER_SIZE_MM:g} mm each."]
    for fit in report['individuals']:
        lines.append(f"ID {fit['id']} alone: solve={fit['solvepnp_succeeded']}, RMS={number(fit['rms_px'])} px, pass={fit['passes_checks']}")
        if fit['rejection_reason']:
            lines.append('  ' + fit['rejection_reason'])
    for name, fit in [('All known markers', report['full_board']), ('Subset [0,1,3] (2 excluded)', report['subset_013'])]:
        lines.append(f"{name}: solve={fit['solvepnp_succeeded']}, RMS={number(fit['rms_px'])} px, pass={fit['passes_checks']}")
        if not fit['available']:
            lines.append('  ' + fit['rejection_reason'])
    lines.append('Residuals evaluated using the 0/1/3 pose:')
    for i, error in report['subset_013']['per_marker_residuals'].items():
        role = 'in fit' if error['in_fit'] else 'HELD OUT'
        lines.append(f"  ID {i} ({role}): RMS={error['rms_px']:.3f}, mean={error['mean_px']:.3f}, max={error['max_px']:.3f} px")
    lines += [f"All individual solvers succeed: {report['individual_solvers_succeeded']}",
              f"Individuals pass but combined board fails: {report['individuals_pass_combined_fails']}",
              'Corner indices: 0=canonical TL, 1=TR, 2=BR, 3=BL; never screen-sort.']
    for i, mapping in report['corner_mapping'].items():
        estimate = 'n/a' if mapping['estimated_rotation_deg'] is None else f"{mapping['estimated_rotation_deg']:+.1f}"
        labels = mapping['physical_corner_labels'] or 'see exact rotated board XY below'
        lines.append(f"ID {i}: declared rotation {mapping['declared_rotation_deg']:+.1f} deg; audit {estimate} deg; physical corners {labels}")
        if detailed:
            lines.append('  ' + mapping['assessment'])
            for j, (xyz, uv) in enumerate(zip(mapping['board_corners_mm'], mapping['detected_corners_px'])):
                lines.append(f"  corner {j} ({CORNER_NAMES[j]}): board {np.round(xyz, 3).tolist()} mm -> RAW {np.round(uv, 2).tolist()} px")
    lines += ['D0..D3 green = detected; P0..P3 magenta = reprojected; numbering matches.',
              f"Numbered RAW view uses {report['display_fit']} fit; no image rotation.",
              'Positive rotation is board +X toward +Y (clockwise in an X-right/Y-down drawing).']
    lines += report['interpretation']
    return lines


def annotate_comparison(raw, debug, report):
    """Return an UNROTATED copy with canonical numbered D/P correspondences."""
    view = raw.copy()
    mode = report['display_fit']
    fit = report['subset_013'] if mode == 'subset' else report['full_board']
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(text, point, color):
        point = tuple(np.rint(point).astype(int))
        cv2.putText(view, text, point, font, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, point, font, .45, color, 1, cv2.LINE_AA)

    for index, (i, quad) in enumerate(zip(debug.detected_ids, debug.corners)):
        observed = np.asarray(quad).reshape(4, 2)
        center = observed.mean(axis=0)
        if mode == 'individual':
            projected = report['individuals'][index]['projected_points']
        else:
            projected = fit['per_marker_residuals'].get(i, {}).get('projected_points')
        label(f'ID {i}', center + [-16, 4], (255, 255, 255))
        for j, point in enumerate(observed):
            direction = (point-center)/max(float(np.linalg.norm(point-center)), 1e-6)
            cv2.circle(view, tuple(np.rint(point).astype(int)), 3, (0, 255, 0), -1)
            label(f'D{j}', point+18*direction+[-8, 4], (0, 255, 0))
            if projected is not None:
                prediction = np.asarray(projected[j])
                if not np.isfinite(prediction).all() or np.max(np.abs(prediction)) > 1e7:
                    continue
                p = tuple(np.rint(prediction).astype(int))
                cv2.line(view, tuple(np.rint(point).astype(int)), p, (0, 255, 255), 1)
                cv2.circle(view, p, 5, (255, 0, 255), 1)
                label(f'P{j}', prediction+36*direction+[-8, 4], (255, 0, 255))
    return view
