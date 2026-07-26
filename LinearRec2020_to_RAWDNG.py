#!/usr/bin/env python3
"""
exr2dng.py
Convert a linear Rec.2020 RGB EXR to a synthetic Bayer-raw DNG.
Sensor gain is set by max_stop only. min_stop defines the noise floor.
"""

import argparse
import glob
import os
import sys
import numpy as np
from fractions import Fraction


# Single source of truth for all default parameters
DEFAULTS = {
    # Input filters: Rec.709 primaries expressed in Rec.2020 RGB space
    "r_filter": [0.627404, 0.069097, 0.016391],
    "g_filter": [0.329283, 0.919540, 0.088013],
    "b_filter": [0.043313, 0.011362, 0.895595],

    # Bayer pattern
    "pattern": "RGGB",

    # Sensor model (stops relative to 0.18 middle gray)
    "min_stop": -10.0,
    "max_stop": 4.0,

    # Sensor black level (DN for zero scene-linear signal)
    "sensor_black_level": 256,

    # Middle gray DN (fixed regardless of max_stop/min_stop)
    # Default: 18% of 14-bit range above sensor_black_level
    "middle_gray_dn": None,  # None = auto-compute as sensor_black_level + 0.18 * (16383 - 256)

    # DNG output levels (will be set to fit full sensor DR [min_stop, max_stop])
    # These are computed automatically if not specified
    "white_level": None,
    "black_level": None,
    "bit_depth": 16,

    # Noise model
    "read_noise": 1.5,
    "shot_noise": True,
    "no_shot_noise": False,
}


def compute_cm1(r_filter, g_filter, b_filter):
    """
    Compute ColorMatrix1 (XYZ to Camera Native) using the algorithmic algorithm.

    Algorithm:
    1. Rec.2020 primary chromaticities (x, y) with Y=1:
       R = (0.708, 0.292), G = (0.170, 0.797), B = (0.131, 0.046)
    2. D65 white point: (0.3127, 0.3290)
    3. Solve for scaling factors s = [sr, sg, sb] such that
       [sr*Xr, sg*Xg, sb*Xb] * [1, 1, 1]^T = [Xw, Yw, Zw]
    4. M_XYZ_from_R2020 = [sr*Xr, sg*Xg, sb*Xb;
                           sr*Yr, sg*Yg, sb*Yb;
                           sr*Zr, sg*Zg, sb*Zb]  (3x3)
    5. Filter matrix F (3x3): rows = r_filter, g_filter, b_filter in Rec.2020 RGB
    6. M_cam_from_XYZ = F @ inv(M_XYZ_from_R2020)  (3x3)
    7. CM1 = M_cam_from_XYZ in DNG rational format (row-major, 18 rationals)
    """
    # Rec.2020 primary chromaticities (x, y) with Y=1
    r_xy = np.array([0.708, 0.292])
    g_xy = np.array([0.170, 0.797])
    b_xy = np.array([0.131, 0.046])

    # D65 white point
    white_xy = np.array([0.3127, 0.3290])

    # Convert (x, y) to XYZ with Y=1
    def xy_to_XYZ(xy):
        x, y = xy
        X = x / y
        Y = 1.0
        Z = (1 - x - y) / y
        return np.array([X, Y, Z])

    Xr, Yr, Zr = xy_to_XYZ(r_xy)
    Xg, Yg, Zg = xy_to_XYZ(g_xy)
    Xb, Yb, Zb = xy_to_XYZ(b_xy)
    Xw, Yw, Zw = xy_to_XYZ(white_xy)

    # Solve for scaling factors: M @ s = w
    M = np.array([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb]
    ])
    w = np.array([Xw, Yw, Zw])
    s = np.linalg.solve(M, w)
    sr, sg, sb = s

    # M_XYZ_from_R2020 (columns are scaled primaries)
    M_XYZ_from_R2020 = np.array([
        [sr * Xr, sg * Xg, sb * Xb],
        [sr * Yr, sg * Yg, sb * Yb],
        [sr * Zr, sg * Zg, sb * Zb]
    ])

    # Filter matrix F (rows = r, g, b filters in Rec.2020 RGB)
    F = np.array([r_filter, g_filter, b_filter])

    # M_cam_from_XYZ = F @ inv(M_XYZ_from_R2020)
    M_cam_from_XYZ = F @ np.linalg.inv(M_XYZ_from_R2020)

    # Convert to DNG rational format (18 rationals, row-major)
    # Each rational is (numerator, denominator)
    rationals = []
    for row in M_cam_from_XYZ:
        for val in row:
            frac = Fraction(val).limit_denominator(10000)
            rationals.append(frac.numerator)
            rationals.append(frac.denominator)

    return tuple(rationals)


def read_exr(path, expect_rgb=True):
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chans = f.header()['channels']

        def read_ch(name):
            return np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(h, w)

        if 'R' in chans and 'G' in chans and 'B' in chans and expect_rgb:
            r = read_ch('R')
            g = read_ch('G')
            b = read_ch('B')
            return np.stack([r, g, b], axis=-1)
        elif 'Y' in chans:
            return read_ch('Y')
        elif 'R' in chans:
            return read_ch('R')
        elif 'Data' in chans:
            return read_ch('Data')
        else:
            first = list(chans.keys())[0]
            return read_ch(first)

    except ImportError:
        pass

    try:
        import imageio.v3 as iio
        img = iio.imread(path)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if expect_rgb and img.ndim == 3 and img.shape[-1] >= 3:
            return img[..., :3]
        elif img.ndim == 3:
            return img[..., 0]
        return img
    except Exception as e:
        raise RuntimeError(
            "Cannot read EXR. Install OpenEXR:  pip install OpenEXR\n"
            "Or:  pip install imageio imageio-freeimage"
        ) from e


# ------------------------------------------------------------------
# 2. Bayer Filtering
# ------------------------------------------------------------------
def filter_rgb_to_scalar(rgb, r_filter, g_filter, b_filter):
    r_plane = np.dot(rgb, r_filter)
    g_plane = np.dot(rgb, g_filter)
    b_plane = np.dot(rgb, b_filter)
    return r_plane, g_plane, b_plane


# ------------------------------------------------------------------
# 3. Interleave Bayer mosaic
# ------------------------------------------------------------------
def build_bayer(r_plane, g1_plane, g2_plane, b_plane, pattern='RGGB'):
    h, w = r_plane.shape
    h = (h // 2) * 2
    w = (w // 2) * 2

    r = r_plane[:h, :w]
    g1 = g1_plane[:h, :w]
    g2 = g2_plane[:h, :w]
    b = b_plane[:h, :w]

    mosaic = np.zeros((h, w), dtype=np.float32)

    if pattern == 'RGGB':
        mosaic[0::2, 0::2] = r[0::2, 0::2]
        mosaic[0::2, 1::2] = g1[0::2, 1::2]
        mosaic[1::2, 0::2] = g2[1::2, 0::2]
        mosaic[1::2, 1::2] = b[1::2, 1::2]
    elif pattern == 'BGGR':
        mosaic[0::2, 0::2] = b[0::2, 0::2]
        mosaic[0::2, 1::2] = g1[0::2, 1::2]
        mosaic[1::2, 0::2] = g2[1::2, 0::2]
        mosaic[1::2, 1::2] = r[1::2, 1::2]
    elif pattern == 'GRBG':
        mosaic[0::2, 0::2] = g1[0::2, 0::2]
        mosaic[0::2, 1::2] = r[0::2, 1::2]
        mosaic[1::2, 0::2] = b[1::2, 0::2]
        mosaic[1::2, 1::2] = g2[1::2, 1::2]
    elif pattern == 'GBRG':
        mosaic[0::2, 0::2] = g1[0::2, 0::2]
        mosaic[0::2, 1::2] = b[0::2, 1::2]
        mosaic[1::2, 0::2] = r[1::2, 0::2]
        mosaic[1::2, 1::2] = g2[1::2, 1::2]
    else:
        raise ValueError(f"Unknown pattern {pattern}")

    return mosaic


# ------------------------------------------------------------------
# 4. Sensor model
# ------------------------------------------------------------------
def quantize_to_sensor(mosaic,
                       min_stop=DEFAULTS["min_stop"],
                       max_stop=DEFAULTS["max_stop"],
                       sensor_black_level=DEFAULTS["sensor_black_level"],
                       middle_gray_dn=DEFAULTS["middle_gray_dn"],
                       white_level=DEFAULTS["white_level"],
                       black_level=DEFAULTS["black_level"],
                       read_noise=DEFAULTS["read_noise"],
                       shot_noise=DEFAULTS["shot_noise"]):
    """
    Linear sensor model with fixed middle gray DN.
    
    Middle gray (0.18 scene-linear) maps to middle_gray_dn (fixed).
    Gain is set so: 0.18 * gain = middle_gray_dn - sensor_black_level
    Sensor DR [min_stop, max_stop] maps to [floor_dn, clip_dn].
    DNG BlackLevel/WhiteLevel are set to floor_dn/clip_dn to fit full sensor DR.
    
    Parameters:
    - sensor_black_level: DN for zero scene-linear signal (default 256)
    - middle_gray_dn: Fixed DN for 0.18 scene-linear (default: 18% of 14-bit range)
    - min_stop/max_stop: Stops relative to 0.18 where sensor clips/floors
    - white_level/black_level: If provided, override auto-computed clip/floor
    """
    # Default middle_gray_dn: 18% of 14-bit usable range (16383 - 256 = 16127)
    if middle_gray_dn is None:
        middle_gray_dn = sensor_black_level + 0.18 * (16383 - sensor_black_level)

    # Gain: DN per scene-linear unit, fixed by middle gray constraint
    # 0.18 * gain = middle_gray_dn - sensor_black_level
    gain = (middle_gray_dn - sensor_black_level) / 0.18

    # Scene-linear clip and floor points
    clip = 0.18 * (2.0 ** max_stop)
    floor = 0.18 * (2.0 ** min_stop)

    # Where clip and floor land in DN
    clip_dn = sensor_black_level + clip * gain
    floor_dn = sensor_black_level + floor * gain

    # DNG BlackLevel/WhiteLevel: use provided or auto-compute to fit full sensor DR
    dng_black = black_level if black_level is not None else floor_dn
    dng_white = white_level if white_level is not None else clip_dn

    print(f"  max_stop: {max_stop:+.1f}  -> clip = {clip:.4f} -> {clip_dn:.1f} DN")
    print(f"  min_stop: {min_stop:+.1f}  -> floor = {floor:.6f} -> {floor_dn:.1f} DN")
    print(f"  Gain: {gain:.2f} DN per scene-linear unit")
    print(f"  0.18 middle gray -> {middle_gray_dn:.1f} DN (fixed)")
    print(f"  DNG BlackLevel: {dng_black:.1f}, WhiteLevel: {dng_white:.1f}")
    print(f"  Sensor DR (stops): {max_stop - min_stop:.2f}")

    # Warn if user-provided levels don't cover full sensor DR
    if black_level is not None and floor_dn < dng_black:
        print(f"  WARNING: Floor ({floor_dn:.1f}) below DNG BlackLevel ({dng_black}), shadows will clip")
    if white_level is not None and clip_dn > dng_white:
        print(f"  WARNING: Clip ({clip_dn:.1f}) above DNG WhiteLevel ({dng_white}), highlights will clip")

    dn = mosaic * gain + sensor_black_level

    # Photon shot noise
    if shot_noise:
        electrons = np.maximum(dn - sensor_black_level, 0.0)
        noise = np.random.normal(0, np.sqrt(electrons))
        dn = dn + noise

    # Read noise
    if read_noise > 0:
        dn = dn + np.random.normal(0, read_noise, dn.shape)

    # Clip to DNG range
    dn = np.clip(dn, dng_black, dng_white)
    
    # Use uint32 if WhiteLevel exceeds 16-bit range
    if dng_white > 65535:
        return np.rint(dn).astype(np.uint32), int(np.floor(dng_black)), int(np.ceil(dng_white))
    return np.rint(dn).astype(np.uint16), int(np.floor(dng_black)), int(np.ceil(dng_white))


# ------------------------------------------------------------------
# 5. Write DNG
# ------------------------------------------------------------------
def write_dng(mosaic, path,
              black_level=DEFAULTS["black_level"],
              white_level=DEFAULTS["white_level"],
              bit_depth=DEFAULTS["bit_depth"],
              pattern=DEFAULTS["pattern"],
               r_filter=None,
               g_filter=None,
               b_filter=None):
    import tifffile

    h, w = mosaic.shape
    cfa_map = {
        'RGGB': [0, 1, 1, 2],
        'BGGR': [2, 1, 1, 0],
        'GRBG': [1, 2, 0, 1],
        'GBRG': [1, 0, 2, 1],
    }
    cfa = cfa_map.get(pattern, [0, 1, 1, 2])

    if r_filter is None:
        # Rec.709 primaries in Rec.2020
        r_filter = np.array([0.627404, 0.069097, 0.016391])
        g_filter = np.array([0.329283, 0.919540, 0.088013])
        b_filter = np.array([0.043313, 0.011362, 0.895595])

    # ColorMatrix1: XYZ to Camera Native (computed from Rec.2020 primaries + filter matrix)
    # Rec.2020: R=(0.708,0.292), G=(0.170,0.797), B=(0.131,0.046), D65=(0.3127,0.3290)
    cm1 = compute_cm1(r_filter, g_filter, b_filter)


# CameraCalibration1: identity
    cc1 = (
        1, 1, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 1,
        0, 1, 0, 1, 1, 1
    )

    r_sum = float(np.sum(r_filter))
    g_sum = float(np.sum(g_filter))
    b_sum = float(np.sum(b_filter))

    asn_r = int(round((r_sum / g_sum) * 10000))
    asn_b = int(round((b_sum / g_sum) * 10000))

    asn = (asn_r, 10000, 1, 1, asn_b, 10000)
    ab = (1, 1, 1, 1, 1, 1)
    ds = (1, 1, 1, 1)

    # Determine bit depth from parameter (always use user-specified bit_depth)
    bits_per_sample = bit_depth

    # Convert data to match bit depth if needed
    if bit_depth == 32 and mosaic.dtype != np.uint32:
        mosaic = mosaic.astype(np.uint32)
    elif bit_depth <= 16 and mosaic.dtype != np.uint16:
        mosaic = mosaic.astype(np.uint16)

    extratags = [
        (262, 3, 1, 32803, False),        # PhotometricInterpretation = CFA
        (33421, 3, 2, (2, 2), False),     # CFARepeatPatternDim
        (33422, 1, 4, tuple(cfa), False), # CFAPattern
        (50706, 1, 4, (1, 4, 0, 0), False),          # DNGVersion
        (50707, 1, 4, (1, 1, 0, 0), False),          # DNGBackwardVersion
        (50708, 2, 1, b"Synthetic", False),         # UniqueCameraModel
        (50714, 4, 1, int(black_level), False),       # BlackLevel
        (50717, 4, 1, int(white_level), False),       # WhiteLevel
        (50718, 5, 2, ds, False),                     # DefaultScale
        (50721, 10, 9, cm1, False),                   # ColorMatrix1
        (50723, 10, 9, cc1, False),                   # CameraCalibration1
        (50728, 5, 3, asn, False),                    # AsShotNeutral
        (50778, 3, 1, 21, False),                     # CalibrationIlluminant1 (D65)
        (50727, 5, 3, ab, False),                     # AnalogBalance
    ]

    tifffile.imwrite(
        path,
        mosaic,
        photometric=32803,
        planarconfig='contig',
        compression=None,
        bitspersample=bits_per_sample,
        extratags=extratags,
    )
    print(f"Wrote DNG: {path}  ({w}x{h}, {pattern}, {bit_depth}-bit)")
    print(f"  BlackLevel={black_level}, WhiteLevel={white_level}")
    print(f"  AsShotNeutral=[{asn_r/10000:.4f}, 1.0, {asn_b/10000:.4f}]")


# ------------------------------------------------------------------
# 6. Helpers
# ------------------------------------------------------------------
def parse_filter(s):
    return np.array([float(x.strip()) for x in s.split(',')], dtype=np.float32)


def find_first_exr(script_dir):
    exr_files = sorted(glob.glob(os.path.join(script_dir, "*.exr")))
    return exr_files[0] if exr_files else None


# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) == 1:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_exr = find_first_exr(script_dir)
        if input_exr is None:
            print("No .exr file found in script directory.")
            print("Usage: python exr2dng.py input.exr [output.dng] [options]")
            sys.exit(1)
        output_dng = os.path.splitext(input_exr)[0] + ".dng"
        print(f"Auto-detected input:  {input_exr}")
        print(f"Auto-generated output: {output_dng}")
        
        args = type('Args', (), {
            'input': input_exr,
            'output_dng': output_dng,
            'planes': None,
            'r_filter': ','.join(map(str, DEFAULTS["r_filter"])),
            'g_filter': ','.join(map(str, DEFAULTS["g_filter"])),
            'b_filter': ','.join(map(str, DEFAULTS["b_filter"])),
            'pattern': DEFAULTS["pattern"],
            'min_stop': DEFAULTS["min_stop"],
            'max_stop': DEFAULTS["max_stop"],
            'sensor_black_level': DEFAULTS["sensor_black_level"],
            'middle_gray_dn': DEFAULTS["middle_gray_dn"],
            'white_level': DEFAULTS["white_level"],
            'black_level': DEFAULTS["black_level"],
            'bit_depth': DEFAULTS["bit_depth"],
            'read_noise': DEFAULTS["read_noise"],
            'no_shot_noise': DEFAULTS["no_shot_noise"],
        })()
    else:
        parser = argparse.ArgumentParser(
            description="Convert linear Rec.2020 RGB EXR to synthetic Bayer-raw DNG. "
                        "Gain is set by max_stop only.")
        parser.add_argument("input", help="Input EXR")
        parser.add_argument("output_dng", nargs='?', default=None,
                            help="Output DNG path (default: same name as input with .dng)")
        parser.add_argument("--planes", nargs=4, metavar=("R","G1","G2","B"),
                            help="Use 4 pre-rendered scalar EXRs instead of RGB filtering")
        parser.add_argument("--r-filter", default=','.join(map(str, DEFAULTS["r_filter"])))
        parser.add_argument("--g-filter", default=','.join(map(str, DEFAULTS["g_filter"])))
        parser.add_argument("--b-filter", default=','.join(map(str, DEFAULTS["b_filter"])))
        parser.add_argument("--pattern", default=DEFAULTS["pattern"],
                            choices=["RGGB", "BGGR", "GRBG", "GBRG"])
        
        # Central setting: min and max stops relative to 0.18
        # max_stop sets the gain. min_stop is informational (noise floor).
        parser.add_argument("--min-stop", type=float, default=DEFAULTS["min_stop"],
                            help="Stops below 0.18 where signal hits noise floor. Default: -10.0")
        parser.add_argument("--max-stop", type=float, default=DEFAULTS["max_stop"],
                            help="Stops above 0.18 where sensor saturates. Sets the gain. "
                                 "Default: 4.0 (0.18 -> ~1264 DN, clip at 2.88)")
        
        parser.add_argument("--sensor-black-level", type=int, default=DEFAULTS["sensor_black_level"],
                            help="DN for zero scene-linear signal (sensor black level). Default: 256")
        parser.add_argument("--middle-gray-dn", type=int, default=DEFAULTS["middle_gray_dn"],
                            help="Fixed DN for 0.18 middle gray. Default: auto (18% of 14-bit range above sensor_black_level)")
        
        parser.add_argument("--white-level", type=int, default=DEFAULTS["white_level"],
                            help="DNG WhiteLevel. Default: auto (sensor clip point at max_stop)")
        parser.add_argument("--black-level", type=int, default=DEFAULTS["black_level"],
                            help="DNG BlackLevel. Default: auto (sensor floor point at min_stop)")
        parser.add_argument("--bit-depth", type=int, default=DEFAULTS["bit_depth"],
                            choices=[8, 10, 12, 14, 16, 32])
        parser.add_argument("--read-noise", type=float, default=DEFAULTS["read_noise"],
                            help="Read noise std-dev in DN (0 to disable)")
        parser.add_argument("--no-shot-noise", action="store_true", default=DEFAULTS["no_shot_noise"])
        args = parser.parse_args()

        if args.output_dng is None:
            args.output_dng = os.path.splitext(args.input)[0] + ".dng"

    # Parse filters
    r_filter = parse_filter(args.r_filter)
    g_filter = parse_filter(args.g_filter)
    b_filter = parse_filter(args.b_filter)
    print(f"Filters (Rec.709 in Rec.2020 RGB):")
    print(f"  R: {r_filter}")
    print(f"  G: {g_filter}")
    print(f"  B: {b_filter}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if args.planes:
        print("\nReading 4 scalar EXRs...")
        r = read_exr(args.planes[0], expect_rgb=False)
        g1 = read_exr(args.planes[1], expect_rgb=False)
        g2 = read_exr(args.planes[2], expect_rgb=False)
        b = read_exr(args.planes[3], expect_rgb=False)
    else:
        print(f"\nReading RGB EXR: {args.input}")
        rgb = read_exr(args.input, expect_rgb=True)
        print(f"  Shape: {rgb.shape}, range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        
        print("Applying Bayer filters...")
        r, g1, b_scalar = filter_rgb_to_scalar(rgb, r_filter, g_filter, b_filter)
        g2 = g1.copy()

    print(f"R  shape: {r.shape}, range: [{r.min():.4f}, {r.max():.4f}]")
    print(f"G1 shape: {g1.shape}, range: [{g1.min():.4f}, {g1.max():.4f}]")
    print(f"G2 shape: {g2.shape}, range: [{g2.min():.4f}, {g2.max():.4f}]")
    b = b if args.planes else b_scalar
    print(f"B  shape: {b.shape}, range: [{b.min():.4f}, {b.max():.4f}]")

    mosaic = build_bayer(r, g1, g2, b, args.pattern)

    # ------------------------------------------------------------------
    # Sensor model
    # ------------------------------------------------------------------
    print("\nApplying sensor model...")
    raw, black_level, white_level = quantize_to_sensor(
        mosaic,
        min_stop=args.min_stop,
        max_stop=args.max_stop,
        sensor_black_level=args.sensor_black_level,
        middle_gray_dn=args.middle_gray_dn,
        black_level=args.black_level,
        white_level=args.white_level,
        read_noise=args.read_noise,
        shot_noise=not args.no_shot_noise,
    )

    # ------------------------------------------------------------------
    # Write DNG
    # ------------------------------------------------------------------
    write_dng(
        raw,
        args.output_dng,
        black_level=black_level,
        white_level=white_level,
        bit_depth=args.bit_depth,
        pattern=args.pattern,
        r_filter=r_filter,
        g_filter=g_filter,
        b_filter=b_filter,
    )


if __name__ == "__main__":
    main()