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


# ------------------------------------------------------------------
# 1. EXR Reading
# ------------------------------------------------------------------
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
                       min_stop=-10.0,
                       max_stop=4.0,
                       black_level=256,
                       white_level=16383,
                       read_noise=1.5,
                       shot_noise=True):
    """
    Linear sensor model. Gain is set by max_stop only.
    
    max_stop: stops above 0.18 where the sensor saturates (WhiteLevel)
    min_stop: stops below 0.18 where signal is lost in noise (informational)
    """
    usable_range = white_level - black_level
    actual_dr = np.log2(usable_range / max(read_noise, 1e-6))

    # Scene-linear values
    clip = 0.18 * (2.0 ** max_stop)
    floor = 0.18 * (2.0 ** min_stop)

    # Gain: DN per unit scene-linear. Fixed by max_stop only.
    scale = usable_range / clip

    # Where 0.18 lands
    gray_dn = black_level + 0.18 * scale

    print(f"  max_stop: {max_stop:+.1f}  -> clip = {clip:.4f}")
    print(f"  min_stop: {min_stop:+.1f}  -> floor = {floor:.6f}")
    print(f"  Scale: {scale:.2f} DN per scene-linear unit")
    print(f"  0.18 middle gray -> {gray_dn:.1f} DN ({gray_dn/white_level*100:.1f}% of white)")
    print(f"  Actual sensor DR (hw): {actual_dr:.2f} stops")

    dn = mosaic * scale + black_level

    # Photon shot noise
    if shot_noise:
        electrons = np.maximum(dn - black_level, 0.0)
        noise = np.random.normal(0, np.sqrt(electrons))
        dn = dn + noise

    # Read noise
    if read_noise > 0:
        dn = dn + np.random.normal(0, read_noise, dn.shape)

    # Scalar clip
    dn = np.clip(dn, black_level, white_level)
    return np.rint(dn).astype(np.uint16)


# ------------------------------------------------------------------
# 5. Write DNG
# ------------------------------------------------------------------
def write_dng(mosaic, path,
              black_level=256,
              white_level=16383,
              bit_depth=14,
              pattern='RGGB',
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

    # ColorMatrix1: XYZ to Camera Native
    # | Primary   | x      | y           | Y           |
    # | --------- | ------ | ----------- | ----------- |
    # | **Red**   | 1.0670 | 0.0372      | 0.0335      |
    # | **Green** | 0.0988 | 0.8739      | 0.7349      |
    # | **Blue**  | 0.1215 | -0.0049     | -0.0066      |
    # White point: x = 0.3915, y = 0.2472, Y = 0.7617
    cm1 = (
    4553, 4415, -155, 1382, -239, 1678,
    -422, 9129, 2219, 1625, 51, 3646,
    291, 3524, -351, 9931, 1441, 1730
)


    # CameraCalibration1: identity
    cc1 = (
        1, 1, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 1,
        0, 1, 0, 1, 1, 1
    )

    if r_filter is None:
        # Rec.709 primaries in Rec.2020
        r_filter = np.array([0.627404, 0.069097, 0.016391])
        g_filter = np.array([0.329283, 0.919540, 0.088013])
        b_filter = np.array([0.043313, 0.011362, 0.895595])

    r_sum = float(np.sum(r_filter))
    g_sum = float(np.sum(g_filter))
    b_sum = float(np.sum(b_filter))

    asn_r = int(round((r_sum / g_sum) * 10000))
    asn_b = int(round((b_sum / g_sum) * 10000))

    asn = (asn_r, 10000, 1, 1, asn_b, 10000)
    ab = (1, 1, 1, 1, 1, 1)
    ds = (1, 1, 1, 1)

    extratags = [
        (262, 3, 1, 32803, False),        # PhotometricInterpretation = CFA
        (33421, 3, 2, (2, 2), False),     # CFARepeatPatternDim
        (33422, 1, 4, tuple(cfa), False), # CFAPattern
        (33421, 3, 2, (2, 2), False),               # CFARepeatPatternDim
        (33422, 1, 4, tuple(cfa), False),             # CFAPattern
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
            'r_filter': "0.627404,0.069097,0.016391",
            'g_filter': "0.329283,0.919540,0.088013",
            'b_filter': "0.043313,0.011362,0.895595",
            'pattern': 'RGGB',
            'min_stop': -10.0,
            'max_stop': 4.0,
            'white_level': 16383,
            'black_level': 256,
            'bit_depth': 14,
            'read_noise': 1.5,
            'no_shot_noise': False,
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
        parser.add_argument("--r-filter", default="0.627404,0.069097,0.016391")
        parser.add_argument("--g-filter", default="0.329283,0.919540,0.088013")
        parser.add_argument("--b-filter", default="0.043313,0.011362,0.895595")
        parser.add_argument("--pattern", default="RGGB",
                            choices=["RGGB", "BGGR", "GRBG", "GBRG"])
        
        # Central setting: min and max stops relative to 0.18
        # max_stop sets the gain. min_stop is informational (noise floor).
        parser.add_argument("--min-stop", type=float, default=-10.0,
                            help="Stops below 0.18 where signal hits noise floor. Default: -10.0")
        parser.add_argument("--max-stop", type=float, default=4.0,
                            help="Stops above 0.18 where sensor saturates. Sets the gain. "
                                 "Default: 4.0  (0.18 -> ~1264 DN, clip at 2.88)")
        
        parser.add_argument("--white-level", type=int, default=16383)
        parser.add_argument("--black-level", type=int, default=256)
        parser.add_argument("--bit-depth", type=int, default=14,
                            choices=[8, 10, 12, 14, 16])
        parser.add_argument("--read-noise", type=float, default=1.5,
                            help="Read noise std-dev in DN (0 to disable)")
        parser.add_argument("--no-shot-noise", action="store_true")
        args = parser.parse_args()

        if args.output_dng is None:
            args.output_dng = os.path.splitext(args.input)[0] + ".dng"

    # Parse filters
    r_filter = parse_filter(args.r_filter)
    g_filter = parse_filter(args.g_filter)
    b_filter = parse_filter(args.b_filter)
    print(f"Filters (Rec.2020 RGB):")
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
    raw = quantize_to_sensor(
        mosaic,
        min_stop=args.min_stop,
        max_stop=args.max_stop,
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
        black_level=args.black_level,
        white_level=args.white_level,
        bit_depth=args.bit_depth,
        pattern=args.pattern,
        r_filter=r_filter,
        g_filter=g_filter,
        b_filter=b_filter,
    )


if __name__ == "__main__":
    main()