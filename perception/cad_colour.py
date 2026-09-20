"""Per-occurrence CAD colour profiles; no component-name rules in strict mode."""
import cv2
import numpy as np
import config


def resolve_colour(part, definition, *, strict=False):
    record = part['color'] if 'color' in part else definition.get('color')
    if record is None:
        if strict:
            raise ValueError(f'{part["id"]}: missing CAD color; re-export with the updated AssemblyGuide')
        return config.PLACEMENT_PART_COLOURS.get(definition.get('fusion_component_name',part['component']))
    rgb = np.asarray(record.get('rgb'),dtype=float)
    if (record.get('status') not in ('solid','mixed') or rgb.shape!=(3,)
            or not np.isfinite(rgb).all() or np.any(rgb<0) or np.any(rgb>255)):
        raise ValueError(f'{part["id"]}: CAD color must provide a solid representative RGB value')
    fraction = record.get('representative_area_fraction')
    if record['status']=='mixed' and (fraction is None or fraction<config.CAD_COLOUR_MIN_DOMINANCE):
        raise ValueError(f'{part["id"]}: mixed-colour part has no dominant colour; unsupported by this matcher')
    rgb = np.rint(rgb).astype(np.uint8)
    h,s,v = map(int,cv2.cvtColor(rgb.reshape(1,1,3),cv2.COLOR_RGB2HSV)[0,0])
    if v<config.CAD_COLOUR_BLACK_VALUE:
        ranges=[[[0,0,0],[179,255,config.CAD_COLOUR_BLACK_VALUE]]]
    elif s<config.CAD_COLOUR_NEUTRAL_SATURATION:
        low = config.CAD_COLOUR_WHITE_VALUE if v>=config.CAD_COLOUR_WHITE_VALUE else config.CAD_COLOUR_MIN_VALUE
        high = 255 if v>=config.CAD_COLOUR_WHITE_VALUE else config.CAD_COLOUR_WHITE_VALUE
        ranges=[[[0,0,low],[179,config.CAD_COLOUR_NEUTRAL_SATURATION,high]]]
    else:
        width=config.CAD_COLOUR_HUE_TOLERANCE
        lower,upper=h-width,h+width
        bands=([(0,upper),(180+lower,179)] if lower<0 else
               [(lower,179),(0,upper-180)] if upper>179 else [(lower,upper)])
        ranges=[[[a,config.CAD_COLOUR_MIN_SATURATION,config.CAD_COLOUR_MIN_VALUE],[b,255,255]] for a,b in bands]
    return dict(label='#'+''.join(f'{x:02X}' for x in rgb),rgb=rgb.tolist(),hsv_ranges=ranges,
                source='part.color' if 'color' in part else 'component.color')


def colour_label(profile):
    return profile['label'] if isinstance(profile,dict) else str(profile)


def matches_colour(source, target):
    if source is None:
        return False
    if isinstance(source,dict):
        from perception.colour_change import colour_mask
        bgr=np.array(source['rgb'][::-1],np.uint8).reshape(1,1,3)
        return bool(colour_mask(bgr,target)[0,0])
    return source==target
