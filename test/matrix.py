#!/usr/bin/env python3
"""The test matrices: each suite runs one of the check scripts over its cases and counts failures.

    matrix.py SUITE [--clips DIR] [--plugin MVGPUtensils.dll] [--reference EXE] [--only TEXT] [--list]

  super    check_super.py: mvgpu.Super against mvu.Super over formats, pels, filters, pelclips,
           grids and paddings
  analyse  check_reference.py: mvgpu.AnalyseMany and Analyse against the CPU reference
           (test/reference/reference.cpp, --reference) over formats and bit depths, block sizes,
           overlaps, pels, chroma, plevel, radii, deltas, the fallback's and the cost's arguments
           and paddings
  degrain  check_degrain.py: mvgpu.Degrain against mvu.Degrain over formats and bit depths, block
           sizes, pels, radii and every argument; on mvgpu.AnalyseMany's vectors (of the clip, or of
           an 8-bit copy of a high bit depth clip), and on mvu.Analyse's in a few cases
  flow     check_flow.py: mvgpu.FlowInter and mvgpu.FlowFPS against mvu's over formats and bit
           depths, block sizes, pels, frame rates and every argument; on mvgpu.Analyse's vectors (of
           the clip, or of an 8-bit copy of a high bit depth clip), on mvu.Analyse's in a few cases,
           and on constant fields with scene changes in between
  masks    check_masks.py: mvgpu.VectorLengthMask, SADMask, OcclusionMask and SCDetection against
           mvu's over formats and bit depths, grids, pels and every argument, on mvgpu.Analyse's
           vectors, mvu.Analyse's and random ones with scene changes at and past the limits
  motion   check_flow.py: mvgpu.Flow, FlowBlur and Compensate against mvu's over formats and bit
           depths, grids, pels and every argument, on the same kinds of vectors as flow
  recalculate
           check_recalculate.py: mvgpu.Recalculate against mvu.Recalculate over every search type,
           old vectors of other grids, mvu's and random old vectors, formats and bit depths, pels
           and the arguments; float clips at pel 1 against mvu's of the clip quantized to 16 bits
  smoke    a quick cross section of all of them (a few minutes, where all of them take hours)

--clips is the directory of test clips, NAME/noisy.nv12 for the clips in CLIPS below (raw 8-bit
NV12). --reference (or MVGPU_REFERENCE) is the built CPU reference, for the analyse and smoke suites.
--only runs the cases whose label contains TEXT. Each case prints the check's last line.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The raw clips the cases use: name -> (size, frames)
CLIPS = {
    'football_fast_s3': ('1920x1080', 52),
    'objects1080fast_s3': ('1920x1080', 60),
    'c0065_s3': ('3840x2160', 60),
}


def super_cases():
    formats = ['YUV422P8', 'YUV440P8', 'YUV444P8', 'GRAY8', 'YUV420P10', 'YUV420P16', 'YUV444P16', 'GRAY16', 'YUV422P12', 'YUV420PS', 'YUV444PS', 'GRAYS']
    cases = [['--format', f] for f in formats]
    cases += [
        ['--pel', '1'], ['--pel', '1', '--format', 'YUV444P16'], ['--pel', '1', '--format', 'GRAYS'],
        ['--pel', '4'], ['--pel', '4', '--format', 'YUV420PS'], ['--pel', '4', '--format', 'YUV422P10'], ['--pel', '4', '--format', 'GRAY8'],
        ['--pel', '4', '--format', 'YUV444P8'], ['--pel', '4', '--format', 'YUV444P16', '--sharp', '1'], ['--pel', '4', '--format', 'YUV444PS', '--rfilter', '0'],
        ['--pel', '4', '--format', 'YUV444P8', '--sharp', '0', '--crop', '1914x1074'],
        ['--sharp', '0'], ['--sharp', '1'], ['--sharp', '0', '--pel', '4'], ['--sharp', '1', '--pel', '4'],
        ['--sharp', '0', '--format', 'YUV420PS'], ['--sharp', '1', '--format', 'YUV444PS'], ['--sharp', '1', '--pel', '4', '--format', 'YUV420P16'],
        ['--rfilter', '0'], ['--rfilter', '2'], ['--rfilter', '0', '--format', 'YUV444PS'], ['--rfilter', '2', '--format', 'GRAYS'],
        ['--rfilter', '2', '--format', 'YUV422P16'], ['--rfilter', '0', '--format', 'YUV440P10'], ['--rfilter', '2', '--format', 'YUV440P10', '--pel', '4'],
        ['--pelclip'], ['--pelclip', '--pel', '4'], ['--pelclip', '--pel', '4', '--format', 'YUV444PS'], ['--pelclip', '--format', 'GRAY16'],
        ['--pelclip', '--pel', '4', '--format', 'YUV422P10', '--sharp', '0'],
        ['--blksize', '8', '--overlap', '4'], ['--blksize', '4', '--overlap', '2', '--crop', '1914x1074'],
        ['--blksize', '16', '8', '--overlap', '4', '2', '--pad', '16', '8', '--crop', '1910x1078'],
        ['--blksize', '32', '16', '--overlap', '8', '4', '--pad', '8', '12', '--format', 'YUV444P8'],
        ['--blksize', '16', '2', '--overlap', '4', '0', '--crop', '1918x1076', '--pel', '4'],
        ['--blksize', '128', '64', '--overlap', '32', '16', '--format', 'YUV420P10'],
        ['--blksize', '64', '32', '--overlap', '0', '--pad', '7', '9', '--format', 'YUV422P8'],
        ['--blksize', '16', '--overlap', '8', '5', '--format', 'YUV422P8', '--crop', '1906x1070'],
        ['--blksize', '8', '--overlap', '2', '4', '--format', 'YUV440P8', '--pel', '4', '--crop', '1910x1074'],
        ['--pel', '4', '--blksize', '4', '--overlap', '2', '--crop', '1914x1074'], ['--pel', '4', '--format', 'YUV420P16', '--pad', '7', '9'],
        ['--blksize', '32', '--overlap', '16', '--pel', '4', '--format', 'YUV444P8', '--crop', '1906x1070'],
        ['--onelevel', '--format', 'YUV444P16'], ['--pel', '4', '--onelevel', '--format', 'YUV422P8'],
        ['--crop', '180x100'], ['--crop', '180x100', '--format', 'GRAYS', '--pel', '4'], ['--pel', '4', '--crop', '180x100', '--format', 'YUV420P10'],
    ]
    return [(' '.join(c), 'football_fast_s3', ['--frames', '2'] + c) for c in cases]


def analyse_cases():
    ff, ob, k4 = 'football_fast_s3', 'objects1080fast_s3', 'c0065_s3'
    cases = []
    # Every block size and pel, overlap half a block, AnalyseMany with radius 2
    for blk in (8, 16, 32):
        for pel in (2, 4):
            cases.append((f'{blk}/{blk // 2} pel {pel}', ff, ['--frames', '10', '--blksize', str(blk), '--overlap', str(blk // 2), '--pel', str(pel)]))

    def grid(blk, overlap, pel=2):
        return ['--blksize', str(blk), '--overlap', str(overlap), '--pel', str(pel)]

    cases += [
        # chroma=False and plevel, at every level of the search
        ('chroma 0 8/4', ff, ['--frames', '10', '--chroma', '0'] + grid(8, 4)),
        ('chroma 0 16/8 pel 4', ff, ['--frames', '10', '--chroma', '0'] + grid(16, 8, 4)),
        ('chroma 0 32/16', ff, ['--frames', '10', '--chroma', '0'] + grid(32, 16)),
        ('chroma 0 32/16 pel 4 standalone', ob, ['--frames', '10', '--chroma', '0', '--standalone'] + grid(32, 16, 4)),
        ('plevel 0 16/8', ff, ['--frames', '10', '--plevel', '0'] + grid(16, 8)),
        ('plevel 2 16/8', ff, ['--frames', '10', '--plevel', '2'] + grid(16, 8)),
        ('plevel 0 32/16 pel 4', ob, ['--frames', '10', '--plevel', '0'] + grid(32, 16, 4)),
        ('plevel 2 8/4 chroma 0', ff, ['--frames', '10', '--plevel', '2', '--chroma', '0'] + grid(8, 4)),
        # Analyse per delta, without chained or inverted seeds
        ('standalone 16/8', ff, ['--frames', '10', '--standalone'] + grid(16, 8)),
        ('standalone 32/16 pel 4', ff, ['--frames', '10', '--standalone'] + grid(32, 16, 4)),
        ('standalone 8/4 pel 4 chroma 0', ff, ['--frames', '10', '--standalone', '--chroma', '0'] + grid(8, 4, 4)),
        # Other overlaps, and grids that end inside a block
        ('32/0', ff, ['--frames', '10'] + grid(32, 0)),
        ('32/8 pel 4', ff, ['--frames', '10'] + grid(32, 8, 4)),
        ('32/12 1906x1070', ff, ['--frames', '10', '--crop', '1906x1070'] + grid(32, 12)),
        ('32/16 pel 4 1910x1074', ob, ['--frames', '10', '--crop', '1910x1074'] + grid(32, 16, 4)),
        ('16/4 1914x1074', ff, ['--frames', '10', '--crop', '1914x1074'] + grid(16, 4)),
        ('16/0 pel 4', ff, ['--frames', '10'] + grid(16, 0, 4)),
        ('8/2 1910x1078', ff, ['--frames', '10', '--crop', '1910x1078'] + grid(8, 2)),
        ('8/0 pel 4 1914x1074', ff, ['--frames', '10', '--crop', '1914x1074'] + grid(8, 0, 4)),
        # Radii and AnalyseMany's delta
        ('radius 3 16/8', ff, ['--frames', '12', '--radius', '3'] + grid(16, 8)),
        ('radius 3 32/16 pel 4', ob, ['--frames', '12', '--radius', '3'] + grid(32, 16, 4)),
        ('radius 1 8/4', ff, ['--frames', '10', '--radius', '1'] + grid(8, 4)),
        ('delta 2 16/8', ff, ['--frames', '12', '--delta', '2'] + grid(16, 8)),
        ('delta 3 radius 1 32/16 pel 4', ff, ['--frames', '12', '--delta', '3', '--radius', '1'] + grid(32, 16, 4)),
        # The fallback's and the cost's arguments, and the padding
        ('badsad 500 badrange 24 badstep 1 32/16', ff, ['--frames', '10', '--badsad', '500', '--badrange', '24', '--badstep', '1'] + grid(32, 16)),
        ('badsad 2000 badstep 3 32/16 pel 4', ob, ['--frames', '10', '--badsad', '2000', '--badstep', '3'] + grid(32, 16, 4)),
        ('badrange 0 16/8', ff, ['--frames', '10', '--badrange', '0'] + grid(16, 8)),
        ('badrange -40 badstep 4 8/4', ff, ['--frames', '10', '--badrange', '-40', '--badstep', '4'] + grid(8, 4)),
        ('badsad 200 16/8 pel 4', ob, ['--frames', '10', '--badsad', '200'] + grid(16, 8, 4)),
        ('mvlambda 0 32/16', ff, ['--frames', '10', '--mvlambda', '0'] + grid(32, 16)),
        ('mvlambda 4000 lsad 1200 32/16 pel 4', ff, ['--frames', '10', '--mvlambda', '4000', '--lsad', '1200'] + grid(32, 16, 4)),
        ('lsad 0 16/8', ff, ['--frames', '10', '--lsad', '0'] + grid(16, 8)),
        ('mvlambda 300 lsad 3000 8/4 pel 4', ob, ['--frames', '10', '--mvlambda', '300', '--lsad', '3000'] + grid(8, 4, 4)),
        ('pad 8 32/16', ff, ['--frames', '10', '--pad', '8'] + grid(32, 16)),
        ('pad 24 16/8 pel 4', ff, ['--frames', '10', '--pad', '24'] + grid(16, 8, 4)),
        ('pad 6 16/8', ff, ['--frames', '10', '--pad', '6'] + grid(16, 8)),
        # 4:4:4: every block size and pel, and the arguments that touch chroma
        ('444 8/4 pel 2', ff, ['--frames', '10', '--format', 'YUV444P8'] + grid(8, 4)),
        ('444 8/4 pel 4', ff, ['--frames', '10', '--format', 'YUV444P8'] + grid(8, 4, 4)),
        ('444 16/8 pel 2', ff, ['--frames', '10', '--format', 'YUV444P8'] + grid(16, 8)),
        ('444 16/8 pel 4', ob, ['--frames', '10', '--format', 'YUV444P8'] + grid(16, 8, 4)),
        ('444 32/16 pel 2', ob, ['--frames', '10', '--format', 'YUV444P8'] + grid(32, 16)),
        ('444 32/16 pel 4', ff, ['--frames', '10', '--format', 'YUV444P8'] + grid(32, 16, 4)),
        ('444 chroma 0 16/8 pel 4', ff, ['--frames', '10', '--format', 'YUV444P8', '--chroma', '0'] + grid(16, 8, 4)),
        ('444 plevel 2 32/16', ff, ['--frames', '10', '--format', 'YUV444P8', '--plevel', '2'] + grid(32, 16)),
        ('444 standalone 8/4 pel 4', ob, ['--frames', '10', '--format', 'YUV444P8', '--standalone'] + grid(8, 4, 4)),
        ('444 radius 3 16/8', ff, ['--frames', '12', '--format', 'YUV444P8', '--radius', '3'] + grid(16, 8)),
        ('444 delta 2 32/16 pel 4', ob, ['--frames', '12', '--format', 'YUV444P8', '--delta', '2'] + grid(32, 16, 4)),
        ('444 16/6 pel 4 1915x1071', ff, ['--frames', '10', '--format', 'YUV444P8', '--crop', '1915x1071'] + grid(16, 6, 4)),
        ('444 8/3 1913x1077', ff, ['--frames', '10', '--format', 'YUV444P8', '--crop', '1913x1077'] + grid(8, 3)),
        ('444 32/0 pel 4', ff, ['--frames', '10', '--format', 'YUV444P8'] + grid(32, 0, 4)),
        ('444 pad 7 16/8', ff, ['--frames', '10', '--format', 'YUV444P8', '--pad', '7'] + grid(16, 8)),
        ('444 badsad 300 badrange 24 badstep 3 32/16', ob, ['--frames', '10', '--format', 'YUV444P8', '--badsad', '300', '--badrange', '24',
                                                             '--badstep', '3'] + grid(32, 16)),
        ('444 mvlambda 3000 lsad 2000 8/4', ff, ['--frames', '10', '--format', 'YUV444P8', '--mvlambda', '3000', '--lsad', '2000'] + grid(8, 4)),
        # High bit depths: every block size and pel at 16 bits, both formats; the other depths; and
        # the arguments the depth scales (mvlambda, lsad, badsad) or that meet the lane split
        ('YUV420P16 8/4 pel 2', ff, ['--frames', '10', '--format', 'YUV420P16'] + grid(8, 4)),
        ('YUV420P16 8/4 pel 4', ob, ['--frames', '10', '--format', 'YUV420P16'] + grid(8, 4, 4)),
        ('YUV420P16 16/8 pel 2', ff, ['--frames', '10', '--format', 'YUV420P16'] + grid(16, 8)),
        ('YUV420P16 16/8 pel 4', ff, ['--frames', '10', '--format', 'YUV420P16'] + grid(16, 8, 4)),
        ('YUV420P16 32/16 pel 2', ob, ['--frames', '10', '--format', 'YUV420P16'] + grid(32, 16)),
        ('YUV420P16 32/16 pel 4', ff, ['--frames', '10', '--format', 'YUV420P16'] + grid(32, 16, 4)),
        ('YUV444P16 8/4 pel 2', ff, ['--frames', '10', '--format', 'YUV444P16'] + grid(8, 4)),
        ('YUV444P16 8/4 pel 4', ff, ['--frames', '10', '--format', 'YUV444P16'] + grid(8, 4, 4)),
        ('YUV444P16 16/8 pel 2', ob, ['--frames', '10', '--format', 'YUV444P16'] + grid(16, 8)),
        ('YUV444P16 16/8 pel 4', ff, ['--frames', '10', '--format', 'YUV444P16'] + grid(16, 8, 4)),
        ('YUV444P16 32/16 pel 2', ff, ['--frames', '10', '--format', 'YUV444P16'] + grid(32, 16)),
        ('YUV444P16 32/16 pel 4', ob, ['--frames', '10', '--format', 'YUV444P16'] + grid(32, 16, 4)),
        ('YUV420P10 16/8 pel 2', ff, ['--frames', '10', '--format', 'YUV420P10'] + grid(16, 8)),
        ('YUV420P10 32/16 pel 4 chroma 0', ff, ['--frames', '10', '--format', 'YUV420P10', '--chroma', '0'] + grid(32, 16, 4)),
        ('YUV444P10 8/4 pel 4', ob, ['--frames', '10', '--format', 'YUV444P10'] + grid(8, 4, 4)),
        ('YUV420P12 16/8 pel 4 standalone', ff, ['--frames', '10', '--format', 'YUV420P12', '--standalone'] + grid(16, 8, 4)),
        ('YUV444P12 32/16 pel 2 plevel 2', ff, ['--frames', '10', '--format', 'YUV444P12', '--plevel', '2'] + grid(32, 16)),
        ('YUV420P14 8/4 pel 2 radius 3', ff, ['--frames', '12', '--format', 'YUV420P14', '--radius', '3'] + grid(8, 4)),
        ('YUV444P16 chroma 0 16/8 pel 4', ff, ['--frames', '10', '--format', 'YUV444P16', '--chroma', '0'] + grid(16, 8, 4)),
        ('YUV420P16 chroma 0 32/16 plevel 0', ob, ['--frames', '10', '--format', 'YUV420P16', '--chroma', '0', '--plevel', '0'] + grid(32, 16)),
        ('YUV420P16 delta 2 16/8 pel 4', ff, ['--frames', '12', '--format', 'YUV420P16', '--delta', '2'] + grid(16, 8, 4)),
        ('YUV420P16 badsad 300 badrange 24 badstep 1 32/16', ob, ['--frames', '10', '--format', 'YUV420P16', '--badsad', '300', '--badrange', '24',
                                                                  '--badstep', '1'] + grid(32, 16)),
        ('YUV444P16 badsad 2000 badstep 3 16/8', ff, ['--frames', '10', '--format', 'YUV444P16', '--badsad', '2000', '--badstep', '3'] + grid(16, 8)),
        ('YUV420P16 mvlambda 4000 lsad 1200 16/8', ff, ['--frames', '10', '--format', 'YUV420P16', '--mvlambda', '4000', '--lsad', '1200'] + grid(16, 8)),
        ('YUV444P10 mvlambda 0 32/16 pel 4', ff, ['--frames', '10', '--format', 'YUV444P10', '--mvlambda', '0'] + grid(32, 16, 4)),
        ('YUV420P16 lsad 0 8/4', ff, ['--frames', '10', '--format', 'YUV420P16', '--lsad', '0'] + grid(8, 4)),
        ('YUV420P16 32/12 1906x1070', ff, ['--frames', '10', '--format', 'YUV420P16', '--crop', '1906x1070'] + grid(32, 12)),
        ('YUV444P16 16/6 pel 4 1915x1071', ff, ['--frames', '10', '--format', 'YUV444P16', '--crop', '1915x1071'] + grid(16, 6, 4)),
        ('YUV444P16 8/3 1913x1077', ob, ['--frames', '10', '--format', 'YUV444P16', '--crop', '1913x1077'] + grid(8, 3)),
        ('YUV420P16 pad 6 16/8', ff, ['--frames', '10', '--format', 'YUV420P16', '--pad', '6'] + grid(16, 8)),
        ('YUV444P16 pad 7 32/0 pel 4', ff, ['--frames', '10', '--format', 'YUV444P16', '--pad', '7'] + grid(32, 0, 4)),
        # Gray, and float samples (searched as the 16-bit samples they stand for)
        ('GRAY8 16/8 pel 2', ff, ['--frames', '10', '--format', 'GRAY8'] + grid(16, 8)),
        ('GRAY8 8/4 pel 4', ob, ['--frames', '10', '--format', 'GRAY8'] + grid(8, 4, 4)),
        ('GRAY16 32/16 pel 4', ff, ['--frames', '10', '--format', 'GRAY16'] + grid(32, 16, 4)),
        ('GRAY10 16/8 pel 2 standalone', ff, ['--frames', '10', '--format', 'GRAY10', '--standalone'] + grid(16, 8)),
        ('YUV420PS 16/8 pel 2', ff, ['--frames', '10', '--format', 'YUV420PS'] + grid(16, 8)),
        ('YUV420PS 16/8 pel 4', ob, ['--frames', '10', '--format', 'YUV420PS'] + grid(16, 8, 4)),
        ('YUV420PS 32/16 pel 4', ff, ['--frames', '10', '--format', 'YUV420PS'] + grid(32, 16, 4)),
        ('YUV420PS 8/4 pel 2 radius 3', ff, ['--frames', '12', '--format', 'YUV420PS', '--radius', '3'] + grid(8, 4)),
        ('YUV444PS 8/4 pel 2', ob, ['--frames', '10', '--format', 'YUV444PS'] + grid(8, 4)),
        ('YUV444PS 32/16 pel 4', ff, ['--frames', '10', '--format', 'YUV444PS'] + grid(32, 16, 4)),
        ('YUV444PS chroma 0 16/8 pel 4', ff, ['--frames', '10', '--format', 'YUV444PS', '--chroma', '0'] + grid(16, 8, 4)),
        ('YUV420PS badsad 300 badstep 1 32/16 plevel 2', ob, ['--frames', '10', '--format', 'YUV420PS', '--badsad', '300', '--badstep', '1',
                                                              '--plevel', '2'] + grid(32, 16)),
        ('GRAYS 16/8 pel 4', ff, ['--frames', '10', '--format', 'GRAYS'] + grid(16, 8, 4)),
        ('YUV420PS 16/6 pel 4 1914x1074', ff, ['--frames', '10', '--format', 'YUV420PS', '--crop', '1914x1074'] + grid(16, 6, 4)),
        # pel 1
        ('pel 1 16/8', ff, ['--frames', '10'] + grid(16, 8, 1)),
        ('pel 1 8/4 chroma 0', ob, ['--frames', '10', '--chroma', '0'] + grid(8, 4, 1)),
        ('pel 1 444 32/16', ff, ['--frames', '10', '--format', 'YUV444P8'] + grid(32, 16, 1)),
        ('pel 1 YUV420P16 16/8', ff, ['--frames', '10', '--format', 'YUV420P16'] + grid(16, 8, 1)),
        ('pel 1 YUV420PS 32/16 standalone', ob, ['--frames', '10', '--format', 'YUV420PS', '--standalone'] + grid(32, 16, 1)),
        ('pel 1 GRAY8 8/4', ff, ['--frames', '10', '--format', 'GRAY8'] + grid(8, 4, 1)),
        ('pel 1 16/6 1914x1074', ff, ['--frames', '10', '--crop', '1914x1074'] + grid(16, 6, 1)),
        ('pel 1 badsad 300 badstep 1 16/8', ob, ['--frames', '10', '--badsad', '300', '--badstep', '1'] + grid(16, 8, 1)),
        ('pel 1 radius 3 8/4 plevel 2', ff, ['--frames', '12', '--radius', '3', '--plevel', '2'] + grid(8, 4, 1)),
        # 4K
        ('4K 32/16 pel 4', k4, ['--frames', '6'] + grid(32, 16, 4)),
        ('4K YUV420PS 16/8 pel 4', k4, ['--frames', '6', '--format', 'YUV420PS'] + grid(16, 8, 4)),
        ('4K YUV420P16 16/8 pel 4', k4, ['--frames', '6', '--format', 'YUV420P16'] + grid(16, 8, 4)),
        ('4K YUV444P10 32/16 pel 2', k4, ['--frames', '6', '--format', 'YUV444P10'] + grid(32, 16)),
        ('4K 444 16/8 pel 4', k4, ['--frames', '6', '--format', 'YUV444P8'] + grid(16, 8, 4)),
        ('4K 16/8 chroma 0 plevel 2', k4, ['--frames', '6', '--chroma', '0', '--plevel', '2'] + grid(16, 8)),
        ('4K 8/4 pel 2', k4, ['--frames', '6'] + grid(8, 4)),
    ]
    return cases


def degrain_cases():
    ff, ob, k4 = 'football_fast_s3', 'objects1080fast_s3', 'c0065_s3'
    cases = []
    # Every format, block size and pel, overlap half a block; mvgpu's vectors where Analyse searches the grid
    for fmt in ('YUV420P8', 'YUV444P8'):
        for blk in (8, 16, 32):
            for pel in (2, 4):
                vec = 'mvgpu'
                cases.append((f'{fmt} {blk}/{blk // 2} pel {pel} {vec} vectors', ff,
                              ['--frames', '8', '--format', fmt, '--blksize', str(blk), '--overlap', str(blk // 2), '--pel', str(pel), '--vectors', vec]))
    cases += [
        ('8/4 pel 2', ff, ['--frames', '12', '--blksize', '8', '--overlap', '4', '--pel', '2']),
        ('16/4 pel 2 1914x1074', ff, ['--frames', '12', '--blksize', '16', '--overlap', '4', '--pel', '2', '--crop', '1914x1074']),
        ('16/4 pel 4 1914x1074', ff, ['--frames', '12', '--blksize', '16', '--overlap', '4', '--pel', '4', '--crop', '1914x1074']),
        ('8/2 pel 2 1910x1078', ff, ['--frames', '12', '--blksize', '8', '--overlap', '2', '--pel', '2', '--crop', '1910x1078']),
        ('8/2 pel 4 1910x1078', ff, ['--frames', '12', '--blksize', '8', '--overlap', '2', '--pel', '4', '--crop', '1910x1078']),
        ('16/0 pel 2', ff, ['--frames', '12', '--blksize', '16', '--overlap', '0', '--pel', '2']),
        ('8/0 pel 4 1914x1074', ff, ['--frames', '12', '--blksize', '8', '--overlap', '0', '--pel', '4', '--crop', '1914x1074']),
        ('16/2 pel 2 1916x1076', ff, ['--frames', '12', '--blksize', '16', '--overlap', '2', '--pel', '2', '--crop', '1916x1076']),
        ('16/6 pel 4', ff, ['--frames', '12', '--blksize', '16', '--overlap', '6', '--pel', '4']),
        ('32/8 pel 2 1906x1070', ff, ['--frames', '8', '--blksize', '32', '--overlap', '8', '--pel', '2', '--crop', '1906x1070', '--vectors', 'mvu']),
        ('32/0 pel 4', ff, ['--frames', '8', '--blksize', '32', '--overlap', '0', '--pel', '4', '--vectors', 'mvu']),
        ('32/12 pel 4 1910x1074', ff, ['--frames', '8', '--blksize', '32', '--overlap', '12', '--pel', '4', '--crop', '1910x1074', '--vectors', 'mvu']),
        ('444 8/2 pel 4 1913x1077', ff, ['--frames', '8', '--format', 'YUV444P8', '--blksize', '8', '--overlap', '2', '--pel', '4', '--crop', '1913x1077', '--vectors', 'mvu']),
        ('444 16/0 pel 2', ff, ['--frames', '8', '--format', 'YUV444P8', '--blksize', '16', '--overlap', '0', '--pel', '2', '--vectors', 'mvu']),
        ('444 16/6 pel 4 1915x1071', ff, ['--frames', '8', '--format', 'YUV444P8', '--blksize', '16', '--overlap', '6', '--pel', '4', '--crop', '1915x1071', '--vectors', 'mvu']),
        ('444 32/16 pel 2 1906x1070', ff, ['--frames', '8', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--pel', '2', '--crop', '1906x1070', '--vectors', 'mvu']),
        ('radius 1', ff, ['--frames', '12', '--radius', '1']),
        ('radius 3 pel 4', ff, ['--frames', '12', '--radius', '3', '--pel', '4']),
        ('radius 6 8/4', ff, ['--frames', '16', '--radius', '6', '--blksize', '8', '--overlap', '4']),
        ('radius 10', ff, ['--frames', '24', '--radius', '10']),
        ('radius 3 444 32/16 pel 4', ff, ['--frames', '12', '--radius', '3', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--pel', '4', '--vectors', 'mvu']),
        ('radius 6 32/16', ff, ['--frames', '16', '--radius', '6', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu']),
        ('thsad 600 300, thsad2 200', ff, ['--frames', '12', '--radius', '3', '--thsad', '600', '300', '--thsad2', '200']),
        ('444 thsad 600 300, thsad2 200', ff, ['--frames', '12', '--radius', '3', '--thsad', '600', '300', '--thsad2', '200', '--format', 'YUV444P8', '--vectors', 'mvu']),
        ('planes 0', ff, ['--frames', '8', '--planes', '0']),
        ('planes 1 2 pel 4', ff, ['--frames', '8', '--planes', '1', '2', '--pel', '4']),
        ('planes 0 2', ff, ['--frames', '8', '--planes', '0', '2']),
        ('444 planes 1 2 pel 4', ff, ['--frames', '8', '--planes', '1', '2', '--pel', '4', '--format', 'YUV444P8', '--vectors', 'mvu']),
        ('limit 3 2', ff, ['--frames', '8', '--limit', '3', '2']),
        ('limit 1', ff, ['--frames', '8', '--limit', '1']),
        ('limit 255', ff, ['--frames', '8', '--limit', '255']),
        ('444 limit 3 2', ff, ['--frames', '8', '--limit', '3', '2', '--format', 'YUV444P8', '--vectors', 'mvu']),
        ('thscd1 150 thscd2 20', ob, ['--frames', '12', '--thscd1', '150', '--thscd2', '20']),
        ('thscd1 0 thscd2 0', ob, ['--frames', '8', '--thscd1', '0', '--thscd2', '0']),
        ('thscd2 100', ob, ['--frames', '8', '--thscd2', '100']),
        ('thscd1 400 thscd2 30', ob, ['--frames', '12', '--thscd1', '400', '--thscd2', '30']),
        ('thscd1 300 thscd2 50 r3', ob, ['--frames', '12', '--thscd1', '300', '--thscd2', '50', '--radius', '3']),
        ('thscd1 900 thscd2 5 ff', ff, ['--frames', '12', '--thscd1', '900', '--thscd2', '5']),
        ('444 32/16 thscd1 300 thscd2 50', ob, ['--frames', '12', '--thscd1', '300', '--thscd2', '50', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu']),
        ('weights 1 2 3 2 1', ff, ['--frames', '8', '--weights', '1', '2', '3', '2', '1']),
        ('weights 0 1 0 1 0', ff, ['--frames', '8', '--weights', '0', '1', '0', '1', '0']),
        ('weights 7 0 0 0 7 pel 4', ff, ['--frames', '8', '--weights', '7', '0', '0', '0', '7', '--pel', '4']),
        ('weights 9999 1 100000 3 0', ff, ['--frames', '8', '--weights', '9999', '1', '100000', '3', '0']),
        ('centersuper', ff, ['--frames', '8', '--centersuper']),
        ('centersuper pel 4 8/4', ff, ['--frames', '8', '--centersuper', '--pel', '4', '--blksize', '8', '--overlap', '4']),
        ('444 centersuper pel 4 32/16', ff, ['--frames', '8', '--centersuper', '--pel', '4', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu']),
        ('render 8/4 for 16/8 vectors', ff, ['--frames', '8', '--render', '8', '4']),
        ('render 8/4 + centersuper pel 4', ff, ['--frames', '8', '--render', '8', '4', '--centersuper', '--pel', '4']),
        ('444 render 32/16 for 16/8 vectors pel 4 1920x1072', ff, ['--frames', '8', '--render', '32', '16', '--pel', '4', '--format', 'YUV444P8', '--vectors', 'mvu', '--crop', '1920x1072']),
        ('4K 16/8 pel 4', k4, ['--frames', '6', '--pel', '4']),
        ('4K 8/4 pel 2', k4, ['--frames', '6', '--blksize', '8', '--overlap', '4']),
        ('4K 444 32/16 pel 4', k4, ['--frames', '6', '--pel', '4', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu']),
    ]
    # High bit depths: every block size and pel at 16 bits on vectors of an 8-bit copy, then other
    # depths, mvu's vectors analysed at the clip's depth (thresholds scaled to it), and the arguments
    # that meet the samples' range
    a8 = ['--analyse8']
    for fmt in ('YUV420P16', 'YUV444P16'):
        for blk in (8, 16, 32):
            for pel in (2, 4):
                cases.append((f'{fmt} {blk}/{blk // 2} pel {pel} analysed on 8 bits', ff,
                              ['--frames', '8', '--format', fmt, '--blksize', str(blk), '--overlap', str(blk // 2), '--pel', str(pel)] + a8))
    for fmt in ('YUV420P16', 'YUV444P16'):
        for blk in (8, 16, 32):
            cases.append((f'{fmt} {blk}/{blk // 2} pel 4 mvgpu vectors', ff,
                          ['--frames', '8', '--format', fmt, '--blksize', str(blk), '--overlap', str(blk // 2), '--pel', '4']))
    cases += [
        ('YUV420P10 16/8 pel 2 analysed on 8 bits', ff, ['--frames', '8', '--format', 'YUV420P10'] + a8),
        ('YUV420P10 16/8 pel 2 mvgpu vectors', ff, ['--frames', '8', '--format', 'YUV420P10']),
        ('YUV444P12 8/4 pel 2 radius 3 mvgpu vectors', ff, ['--frames', '12', '--format', 'YUV444P12', '--blksize', '8', '--overlap', '4', '--radius', '3']),
        ('YUV420P16 thscd1 150 thscd2 20 mvgpu vectors', ob, ['--frames', '12', '--format', 'YUV420P16', '--thscd1', '150', '--thscd2', '20']),
        ('YUV420P10 8/4 pel 4 mvu vectors', ff, ['--frames', '8', '--format', 'YUV420P10', '--blksize', '8', '--overlap', '4', '--pel', '4', '--vectors', 'mvu']),
        ('YUV444P12 32/16 pel 4 analysed on 8 bits', ff, ['--frames', '8', '--format', 'YUV444P12', '--blksize', '32', '--overlap', '16', '--pel', '4'] + a8),
        ('YUV420P14 16/4 pel 2 1914x1074 mvu vectors', ff, ['--frames', '8', '--format', 'YUV420P14', '--blksize', '16', '--overlap', '4', '--crop', '1914x1074', '--vectors', 'mvu']),
        ('YUV420P16 16/8 pel 2 mvu vectors', ff, ['--frames', '8', '--format', 'YUV420P16', '--vectors', 'mvu']),
        ('YUV444P16 32/16 pel 4 mvu vectors', ff, ['--frames', '8', '--format', 'YUV444P16', '--blksize', '32', '--overlap', '16', '--pel', '4', '--vectors', 'mvu']),
        ('YUV420P16 radius 3 thsad 600 300 thsad2 200', ff, ['--frames', '12', '--format', 'YUV420P16', '--radius', '3', '--thsad', '600', '300', '--thsad2', '200'] + a8),
        ('YUV444P16 32/16 thsad 20000 mvu vectors', ff, ['--frames', '8', '--format', 'YUV444P16', '--blksize', '32', '--overlap', '16', '--thsad', '20000', '--vectors', 'mvu']),
        ('YUV420P16 limit 300 200', ff, ['--frames', '8', '--format', 'YUV420P16', '--limit', '300', '200'] + a8),
        ('YUV420P10 limit 3 1023', ff, ['--frames', '8', '--format', 'YUV420P10', '--limit', '3', '1023'] + a8),
        ('YUV420P16 thscd1 300 thscd2 50 mvu vectors', ob, ['--frames', '12', '--format', 'YUV420P16', '--thscd1', '300', '--thscd2', '50', '--vectors', 'mvu']),
        ('YUV420P16 thscd1 150 thscd2 20', ob, ['--frames', '12', '--format', 'YUV420P16', '--thscd1', '150', '--thscd2', '20'] + a8),
        ('YUV420P16 weights 1 2 3 2 1 planes 0 2', ff, ['--frames', '8', '--format', 'YUV420P16', '--weights', '1', '2', '3', '2', '1', '--planes', '0', '2'] + a8),
        ('YUV444P16 centersuper pel 4', ff, ['--frames', '8', '--format', 'YUV444P16', '--centersuper', '--pel', '4'] + a8),
        ('YUV420P16 render 8/4 for 16/8 vectors', ff, ['--frames', '8', '--format', 'YUV420P16', '--render', '8', '4'] + a8),
        ('YUV420P16 radius 6 8/0 pel 4 1914x1074', ff, ['--frames', '16', '--format', 'YUV420P16', '--radius', '6', '--blksize', '8', '--overlap', '0', '--pel', '4',
                                                        '--crop', '1914x1074'] + a8),
        ('4K YUV420P16 16/8 pel 4', k4, ['--frames', '6', '--format', 'YUV420P16', '--pel', '4'] + a8),
        ('4K YUV444P10 32/16 pel 2 mvu vectors', k4, ['--frames', '6', '--format', 'YUV444P10', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu']),
    ]
    # Gray, and float samples (compared bit for bit)
    cases += [
        ('GRAY8 16/8 pel 2', ff, ['--frames', '8', '--format', 'GRAY8']),
        ('GRAY16 32/16 pel 4', ff, ['--frames', '8', '--format', 'GRAY16', '--blksize', '32', '--overlap', '16', '--pel', '4']),
        ('GRAY8 limit 3 radius 3', ff, ['--frames', '12', '--format', 'GRAY8', '--limit', '3', '--radius', '3']),
        ('GRAY10 8/4 pel 4 mvu vectors', ff, ['--frames', '8', '--format', 'GRAY10', '--blksize', '8', '--overlap', '4', '--pel', '4', '--vectors', 'mvu']),
        ('GRAYS 16/0 pel 2', ff, ['--frames', '8', '--format', 'GRAYS', '--blksize', '16', '--overlap', '0']),
        ('YUV420PS 16/8 pel 2 mvgpu vectors', ff, ['--frames', '8', '--format', 'YUV420PS']),
        ('YUV420PS 8/4 pel 4 mvu vectors', ff, ['--frames', '8', '--format', 'YUV420PS', '--blksize', '8', '--overlap', '4', '--pel', '4', '--vectors', 'mvu']),
        ('YUV444PS 32/16 pel 4 mvgpu vectors', ff, ['--frames', '8', '--format', 'YUV444PS', '--blksize', '32', '--overlap', '16', '--pel', '4']),
        ('YUV444PS 16/8 pel 2 mvu vectors', ff, ['--frames', '8', '--format', 'YUV444PS', '--vectors', 'mvu']),
        ('YUV420PS radius 3 thsad 600 300 thsad2 200', ff, ['--frames', '12', '--format', 'YUV420PS', '--radius', '3', '--thsad', '600', '300', '--thsad2', '200']),
        ('YUV420PS limit 0.01 0.02', ff, ['--frames', '8', '--format', 'YUV420PS', '--limit', '0.01', '0.02']),
        ('YUV444PS centersuper pel 4', ff, ['--frames', '8', '--format', 'YUV444PS', '--centersuper', '--pel', '4']),
        ('YUV420PS planes 1 2 weights 1 2 3 2 1', ff, ['--frames', '8', '--format', 'YUV420PS', '--planes', '1', '2', '--weights', '1', '2', '3', '2', '1']),
        ('YUV420PS 16/6 pel 4 1914x1074', ff, ['--frames', '8', '--format', 'YUV420PS', '--blksize', '16', '--overlap', '6', '--pel', '4', '--crop', '1914x1074']),
        ('YUV420PS analysed on 8 bits', ff, ['--frames', '8', '--format', 'YUV420PS', '--analyse8']),
        ('YUV420PS thscd1 150 thscd2 20', ob, ['--frames', '12', '--format', 'YUV420PS', '--thscd1', '150', '--thscd2', '20']),
        ('4K YUV420PS 16/8 pel 4', k4, ['--frames', '6', '--format', 'YUV420PS', '--pel', '4']),
    ]
    # pel 1
    cases += [
        ('pel 1 16/8', ff, ['--frames', '8', '--pel', '1']),
        ('pel 1 8/4', ff, ['--frames', '8', '--pel', '1', '--blksize', '8', '--overlap', '4']),
        ('pel 1 444 32/16', ff, ['--frames', '8', '--pel', '1', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16']),
        ('pel 1 YUV420P16 16/8', ff, ['--frames', '8', '--pel', '1', '--format', 'YUV420P16']),
        ('pel 1 YUV444PS 16/8', ff, ['--frames', '8', '--pel', '1', '--format', 'YUV444PS']),
        ('pel 1 GRAY16 8/4 mvu vectors', ff, ['--frames', '8', '--pel', '1', '--format', 'GRAY16', '--blksize', '8', '--overlap', '4', '--vectors', 'mvu']),
        ('pel 1 render 8/4 for 16/8 vectors', ff, ['--frames', '8', '--pel', '1', '--render', '8', '4']),
        ('pel 1 radius 3 limit 3 2', ff, ['--frames', '12', '--pel', '1', '--radius', '3', '--limit', '3', '2']),
        ('pel 1 16/0 1916x1076', ff, ['--frames', '8', '--pel', '1', '--blksize', '16', '--overlap', '0', '--crop', '1916x1076']),
    ]
    return cases


def flow_cases():
    ff, ob, k4 = 'football_fast_s3', 'objects1080fast_s3', 'c0065_s3'
    cases = []
    # Every format, block size and pel, overlap half a block; mvgpu's vectors where Analyse searches the grid
    for fmt in ('YUV420P8', 'YUV444P8'):
        for blk in (8, 16, 32):
            for pel in (2, 4):
                vec = 'mvgpu'
                grid = ['--format', fmt, '--blksize', str(blk), '--overlap', str(blk // 2), '--pel', str(pel), '--vectors', vec]
                cases.append((f'inter {fmt} {blk}/{blk // 2} pel {pel} {vec} vectors', ff, ['--frames', '8'] + grid))
                if blk == 16:
                    cases.append((f'fps {fmt} {blk}/{blk // 2} pel {pel} {vec} vectors', ff, ['--frames', '8', '--filter', 'fps', '--num', '60', '--den', '1'] + grid))
    inter, fps = ['--filter', 'inter'], ['--filter', 'fps']
    cases += [
        # Constant fields: scene changes at and past the limits on frames beside interpolated ones
        ('inter const', ff, ['--frames', '15', '--vectors', 'const'] + inter),
        ('inter const blend 0 pel 4', ff, ['--frames', '15', '--vectors', 'const', '--blend', '0', '--pel', '4'] + inter),
        ('inter const 444 32/16 delta 2', ff, ['--frames', '15', '--vectors', 'const', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--delta', '2'] + inter),
        ('inter const thscd1 100 thscd2 10', ff, ['--frames', '15', '--vectors', 'const', '--thscd1', '100', '--thscd2', '10'] + inter),
        ('fps const', ff, ['--frames', '15', '--vectors', 'const', '--num', '60', '--den', '1'] + fps),
        ('fps const extramask 0 blend 0', ff, ['--frames', '15', '--vectors', 'const', '--num', '50', '--den', '1', '--extramask', '0', '--blend', '0'] + fps),
        ('fps const 444 8/4 pel 4 delta 3', ff, ['--frames', '15', '--vectors', 'const', '--format', 'YUV444P8', '--blksize', '8', '--overlap', '4', '--pel', '4', '--delta', '3', '--num', '0', '--den', '0'] + fps),
        # FlowInter's arguments
        ('inter time 0', ff, ['--frames', '8', '--time', '0'] + inter),
        ('inter time 100 pel 4', ff, ['--frames', '8', '--time', '100', '--pel', '4'] + inter),
        ('inter time 33.3', ff, ['--frames', '8', '--time', '33.3'] + inter),
        ('inter time 75 444 32/16', ff, ['--frames', '8', '--time', '75', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu'] + inter),
        ('inter ml 1', ff, ['--frames', '8', '--ml', '1'] + inter),
        ('inter ml 10000 pel 4', ff, ['--frames', '8', '--ml', '10000', '--pel', '4'] + inter),
        ('inter ml 3 444 8/4', ff, ['--frames', '8', '--ml', '3', '--format', 'YUV444P8', '--blksize', '8', '--overlap', '4', '--vectors', 'mvu'] + inter),
        ('inter blend 0', ob, ['--frames', '12', '--blend', '0', '--thscd1', '150', '--thscd2', '20'] + inter),
        ('inter thscd1 150 thscd2 20', ob, ['--frames', '12', '--thscd1', '150', '--thscd2', '20'] + inter),
        ('inter thscd1 0 thscd2 0', ob, ['--frames', '8', '--thscd1', '0', '--thscd2', '0'] + inter),
        ('inter thscd2 100', ob, ['--frames', '8', '--thscd2', '100'] + inter),
        ('inter delta 2', ff, ['--frames', '12', '--delta', '2'] + inter),
        ('inter delta 3 pel 4 8/4', ff, ['--frames', '12', '--delta', '3', '--pel', '4', '--blksize', '8', '--overlap', '4'] + inter),
        # FlowFPS's rates and arguments
        ('fps 24 to 48 (num 0)', ff, ['--frames', '10', '--num', '0', '--den', '0'] + fps),
        ('fps 24 to 25', ff, ['--frames', '10', '--num', '25', '--den', '1'] + fps),
        ('fps 24 to 15', ff, ['--frames', '12', '--num', '15', '--den', '1'] + fps),
        ('fps 30000/1001 to 60000/1001 pel 4', ff, ['--frames', '10', '--fps', '30000/1001', '--num', '60000', '--den', '1001', '--pel', '4'] + fps),
        ('fps 24 to 60 extramask 0', ff, ['--frames', '10', '--num', '60', '--den', '1', '--extramask', '0'] + fps),
        ('fps 24 to 60 blend 0 thscd1 150 thscd2 20', ob, ['--frames', '10', '--num', '60', '--den', '1', '--blend', '0', '--thscd1', '150', '--thscd2', '20'] + fps),
        ('fps 24 to 72 delta 2', ff, ['--frames', '10', '--num', '72', '--den', '1', '--delta', '2'] + fps),
        ('fps 24 to 60 ml 2 444 32/16 pel 4', ff, ['--frames', '8', '--num', '60', '--den', '1', '--ml', '2', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--pel', '4', '--vectors', 'mvu'] + fps),
        # Grids that end inside a block, and without overlap
        ('inter 16/4 1914x1074', ff, ['--frames', '8', '--blksize', '16', '--overlap', '4', '--crop', '1914x1074'] + inter),
        ('inter 8/2 pel 4 1910x1078', ff, ['--frames', '8', '--blksize', '8', '--overlap', '2', '--pel', '4', '--crop', '1910x1078'] + inter),
        ('inter 32/8 1906x1070', ff, ['--frames', '8', '--blksize', '32', '--overlap', '8', '--crop', '1906x1070', '--vectors', 'mvu'] + inter),
        ('inter 444 8/2 pel 4 1913x1077', ff, ['--frames', '8', '--format', 'YUV444P8', '--blksize', '8', '--overlap', '2', '--pel', '4', '--crop', '1913x1077', '--vectors', 'mvu'] + inter),
        ('fps 444 16/6 pel 4 1915x1071', ff, ['--frames', '8', '--format', 'YUV444P8', '--blksize', '16', '--overlap', '6', '--pel', '4', '--crop', '1915x1071', '--vectors', 'mvu', '--num', '60', '--den', '1'] + fps),
        ('inter 16/0', ff, ['--frames', '8', '--blksize', '16', '--overlap', '0'] + inter),
        ('inter 8/0 pel 4 1914x1074', ff, ['--frames', '8', '--blksize', '8', '--overlap', '0', '--pel', '4', '--crop', '1914x1074'] + inter),
        ('fps 32/0 pel 4', ff, ['--frames', '8', '--blksize', '32', '--overlap', '0', '--pel', '4', '--vectors', 'mvu', '--num', '60', '--den', '1'] + fps),
        ('inter 444 16/0', ff, ['--frames', '8', '--format', 'YUV444P8', '--blksize', '16', '--overlap', '0', '--vectors', 'mvu'] + inter),
        ('inter 180x100', ff, ['--frames', '8', '--crop', '180x100', '--vectors', 'mvu'] + inter),
        ('fps 444 182x102 32/16 pel 4', ff, ['--frames', '8', '--crop', '182x102', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--pel', '4', '--vectors', 'mvu', '--num', '60', '--den', '1'] + fps),
        # 4K
        ('4K inter 16/8 pel 4', k4, ['--frames', '6', '--pel', '4'] + inter),
        ('4K fps 8/4', k4, ['--frames', '6', '--blksize', '8', '--overlap', '4', '--num', '60', '--den', '1'] + fps),
        ('4K inter 444 32/16 pel 4', k4, ['--frames', '6', '--pel', '4', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu'] + inter),
    ]
    # High bit depths: every block size and pel at 16 bits on vectors of an 8-bit copy, then other
    # depths, mvu's vectors analysed at the clip's depth, constant fields and the arguments
    a8 = ['--analyse8']
    for fmt in ('YUV420P16', 'YUV444P16'):
        for blk in (8, 16, 32):
            for pel in (2, 4):
                grid = ['--format', fmt, '--blksize', str(blk), '--overlap', str(blk // 2), '--pel', str(pel)] + a8
                cases.append((f'inter {fmt} {blk}/{blk // 2} pel {pel} analysed on 8 bits', ff, ['--frames', '8'] + grid + inter))
                if blk == 16:
                    cases.append((f'fps {fmt} {blk}/{blk // 2} pel {pel} analysed on 8 bits', ff, ['--frames', '8', '--num', '60', '--den', '1'] + grid + fps))
    for fmt in ('YUV420P16', 'YUV444P16'):
        for blk in (8, 32):
            cases.append((f'inter {fmt} {blk}/{blk // 2} pel 4 mvgpu vectors', ff,
                          ['--frames', '8', '--format', fmt, '--blksize', str(blk), '--overlap', str(blk // 2), '--pel', '4'] + inter))
    cases += [
        ('inter YUV420P10 16/8 analysed on 8 bits', ff, ['--frames', '8', '--format', 'YUV420P10'] + a8 + inter),
        ('fps YUV420P10 16/8 mvgpu vectors', ff, ['--frames', '8', '--format', 'YUV420P10', '--num', '60', '--den', '1'] + fps),
        ('inter YUV444P12 16/8 pel 4 delta 2 mvgpu vectors', ff, ['--frames', '12', '--format', 'YUV444P12', '--pel', '4', '--delta', '2'] + inter),
        ('inter YUV444P12 8/4 pel 4 mvu vectors', ff, ['--frames', '8', '--format', 'YUV444P12', '--blksize', '8', '--overlap', '4', '--pel', '4', '--vectors', 'mvu'] + inter),
        ('inter YUV420P16 mvu vectors', ff, ['--frames', '8', '--format', 'YUV420P16', '--vectors', 'mvu'] + inter),
        ('fps YUV444P16 32/16 pel 4 mvu vectors', ff, ['--frames', '8', '--format', 'YUV444P16', '--blksize', '32', '--overlap', '16', '--pel', '4', '--vectors', 'mvu',
                                                       '--num', '60', '--den', '1'] + fps),
        ('inter const YUV420P16', ff, ['--frames', '15', '--vectors', 'const', '--format', 'YUV420P16'] + inter),
        ('fps const YUV444P10 extramask 0 blend 0', ff, ['--frames', '15', '--vectors', 'const', '--format', 'YUV444P10', '--num', '50', '--den', '1', '--extramask', '0',
                                                         '--blend', '0'] + fps),
        ('inter YUV420P16 time 33.3 ml 3', ff, ['--frames', '8', '--format', 'YUV420P16', '--time', '33.3', '--ml', '3'] + a8 + inter),
        ('inter YUV420P16 thscd1 150 thscd2 20 blend 0', ob, ['--frames', '12', '--format', 'YUV420P16', '--thscd1', '150', '--thscd2', '20', '--blend', '0'] + a8 + inter),
        ('fps YUV420P16 24 to 72 delta 2', ff, ['--frames', '10', '--format', 'YUV420P16', '--num', '72', '--den', '1', '--delta', '2'] + a8 + fps),
        ('inter YUV444P16 8/2 pel 4 1913x1077', ff, ['--frames', '8', '--format', 'YUV444P16', '--blksize', '8', '--overlap', '2', '--pel', '4', '--crop', '1913x1077'] + a8 + inter),
        ('4K inter YUV420P16 16/8 pel 4', k4, ['--frames', '6', '--format', 'YUV420P16', '--pel', '4'] + a8 + inter),
        ('4K fps YUV444P10 32/16 mvu vectors', k4, ['--frames', '6', '--format', 'YUV444P10', '--blksize', '32', '--overlap', '16', '--vectors', 'mvu', '--num', '60',
                                                    '--den', '1'] + fps),
    ]
    # Gray, and float samples (compared bit for bit)
    cases += [
        ('inter GRAY8 16/8', ff, ['--frames', '8', '--format', 'GRAY8'] + inter),
        ('fps GRAY16 32/16 pel 4', ff, ['--frames', '8', '--format', 'GRAY16', '--blksize', '32', '--overlap', '16', '--pel', '4', '--num', '60', '--den', '1'] + fps),
        ('inter GRAYS 8/4', ff, ['--frames', '8', '--format', 'GRAYS', '--blksize', '8', '--overlap', '4'] + inter),
        ('inter const GRAY10 blend 0', ff, ['--frames', '15', '--vectors', 'const', '--format', 'GRAY10', '--blend', '0'] + inter),
        ('inter YUV420PS 16/8 pel 2', ff, ['--frames', '8', '--format', 'YUV420PS'] + inter),
        ('inter YUV420PS 16/8 pel 4', ff, ['--frames', '8', '--format', 'YUV420PS', '--pel', '4'] + inter),
        ('fps YUV444PS 32/16 pel 4', ff, ['--frames', '8', '--format', 'YUV444PS', '--blksize', '32', '--overlap', '16', '--pel', '4', '--num', '60', '--den', '1'] + fps),
        ('fps YUV420PS 8/4 extramask 1', ff, ['--frames', '8', '--format', 'YUV420PS', '--blksize', '8', '--overlap', '4', '--num', '60', '--den', '1',
                                              '--extramask', '1'] + fps),
        ('inter const YUV420PS', ff, ['--frames', '15', '--vectors', 'const', '--format', 'YUV420PS'] + inter),
        ('fps const YUV444PS extramask 0 blend 0', ff, ['--frames', '15', '--vectors', 'const', '--format', 'YUV444PS', '--num', '50', '--den', '1', '--extramask', '0',
                                                        '--blend', '0'] + fps),
        ('inter YUV420PS time 33.3 ml 3', ff, ['--frames', '8', '--format', 'YUV420PS', '--time', '33.3', '--ml', '3'] + inter),
        ('inter YUV420PS mvu vectors', ff, ['--frames', '8', '--format', 'YUV420PS', '--vectors', 'mvu'] + inter),
        ('inter YUV444PS analysed on 8 bits', ff, ['--frames', '8', '--format', 'YUV444PS', '--analyse8'] + inter),
        ('inter YUV420PS thscd1 150 thscd2 20 blend 0', ob, ['--frames', '12', '--format', 'YUV420PS', '--thscd1', '150', '--thscd2', '20', '--blend', '0'] + inter),
        ('4K inter YUV420PS 16/8 pel 4', k4, ['--frames', '6', '--format', 'YUV420PS', '--pel', '4'] + inter),
    ]
    # pel 1
    cases += [
        ('inter pel 1 16/8', ff, ['--frames', '8', '--pel', '1'] + inter),
        ('fps pel 1 8/4', ff, ['--frames', '8', '--pel', '1', '--blksize', '8', '--overlap', '4', '--num', '60', '--den', '1'] + fps),
        ('inter pel 1 444 32/16', ff, ['--frames', '8', '--pel', '1', '--format', 'YUV444P8', '--blksize', '32', '--overlap', '16'] + inter),
        ('inter const pel 1', ff, ['--frames', '15', '--pel', '1', '--vectors', 'const'] + inter),
        ('fps pel 1 YUV420P16 16/8', ff, ['--frames', '8', '--pel', '1', '--format', 'YUV420P16', '--num', '60', '--den', '1'] + fps),
        ('inter pel 1 YUV420PS', ff, ['--frames', '8', '--pel', '1', '--format', 'YUV420PS'] + inter),
        ('inter pel 1 GRAY8 time 25', ff, ['--frames', '8', '--pel', '1', '--format', 'GRAY8', '--time', '25'] + inter),
    ]
    return cases


def masks_cases():
    ff, ob, k4 = 'football_fast_s3', 'objects1080fast_s3', 'c0065_s3'
    rnd = ['--vectors', 'random']
    return [
        ('16/8 pel 2', ff, ['--frames', '10']),
        ('8/4 pel 4', ob, ['--frames', '10', '--blksize', '8', '--overlap', '4', '--pel', '4']),
        ('32/16 pel 1 1914x1074', ff, ['--frames', '8', '--blksize', '32', '--overlap', '16', '--pel', '1', '--crop', '1914x1074']),
        ('16/6 pel 4 1914x1074', ff, ['--frames', '8', '--blksize', '16', '--overlap', '6', '--pel', '4', '--crop', '1914x1074']),
        ('16/0', ff, ['--frames', '8', '--overlap', '0']),
        ('444 16/8', ff, ['--frames', '8', '--format', 'YUV444P8']),
        ('random vectors', ff, ['--frames', '10'] + rnd),
        ('random vectors delta -1', ff, ['--frames', '10', '--delta', '-1'] + rnd),
        ('random delta 2 thscd1 300 thscd2 20 scval 77', ff, ['--frames', '12', '--delta', '2', '--thscd1', '300', '--thscd2', '20', '--scval', '77'] + rnd),
        ('mvu vectors 8/4', ff, ['--frames', '8', '--vectors', 'mvu', '--blksize', '8', '--overlap', '4']),
        ('ml 30 time 50', ff, ['--frames', '8', '--ml', '30', '--time', '50'] + rnd),
        ('time 0 ml 1000', ff, ['--frames', '8', '--time', '0', '--ml', '1000']),
        ('gamma 2', ff, ['--frames', '8', '--gamma', '2'] + rnd),
        ('gamma 0', ff, ['--frames', '8', '--gamma', '0'] + rnd),
        ('gamma 0.5', ff, ['--frames', '8', '--gamma', '0.5', '--tolerance', '1'] + rnd),
        ('gamma 1.7 YUV420P10', ff, ['--frames', '8', '--gamma', '1.7', '--format', 'YUV420P10', '--tolerance', '1'] + rnd),
        ('YUV420P16', ff, ['--frames', '8', '--format', 'YUV420P16']),
        ('YUV444P12 random', ff, ['--frames', '8', '--format', 'YUV444P12'] + rnd),
        ('YUV420P10 analysed on 8 bits', ff, ['--frames', '8', '--format', 'YUV420P10', '--analyse8']),
        ('GRAY10 delta -2', ff, ['--frames', '8', '--format', 'GRAY10', '--delta', '-2']),
        ('YUV420PS', ff, ['--frames', '8', '--format', 'YUV420PS']),
        ('GRAYS random time 50 ml 30', ff, ['--frames', '8', '--format', 'GRAYS', '--time', '50', '--ml', '30'] + rnd),
        ('YUV444PS gamma 0.37 ml 20', ff, ['--frames', '8', '--format', 'YUV444PS', '--gamma', '0.37', '--ml', '20', '--tolerance', '2.4e-7'] + rnd),
        ('YUV420PS scval 0.5', ff, ['--frames', '12', '--format', 'YUV420PS', '--scval', '0.5', '--thscd1', '300', '--thscd2', '20'] + rnd),
        ('4K 16/8 pel 4', k4, ['--frames', '6', '--pel', '4']),
    ]


def motion_cases():
    ff, ob, k4 = 'football_fast_s3', 'objects1080fast_s3', 'c0065_s3'
    flow, blur, comp = ['--filter', 'flow'], ['--filter', 'blur'], ['--filter', 'compensate']
    cases = []
    for name, f in (('flow', flow), ('blur', blur), ('compensate', comp)):
        cases += [
            (f'{name} 16/8 pel 2', ff, ['--frames', '8'] + f),
            (f'{name} 8/4 pel 4', ff, ['--frames', '8', '--blksize', '8', '--overlap', '4', '--pel', '4'] + f),
            (f'{name} 32/16 pel 1', ff, ['--frames', '8', '--blksize', '32', '--overlap', '16', '--pel', '1'] + f),
            (f'{name} 444 16/8 pel 4', ff, ['--frames', '8', '--format', 'YUV444P8', '--pel', '4'] + f),
            (f'{name} const', ff, ['--frames', '15', '--vectors', 'const'] + f),
            (f'{name} mvu vectors 444 8/2 pel 4 1913x1077', ff, ['--frames', '8', '--vectors', 'mvu', '--format', 'YUV444P8', '--blksize', '8', '--overlap', '2',
                                                               '--pel', '4', '--crop', '1913x1077'] + f),
            (f'{name} 16/6 1914x1074', ff, ['--frames', '8', '--blksize', '16', '--overlap', '6', '--crop', '1914x1074'] + f),
            (f'{name} 16/0', ff, ['--frames', '8', '--overlap', '0'] + f),
            (f'{name} thscd1 150 thscd2 20', ob, ['--frames', '12', '--thscd1', '150', '--thscd2', '20'] + f),
            (f'{name} YUV420P16 pel 4', ff, ['--frames', '8', '--format', 'YUV420P16', '--pel', '4'] + f),
            (f'{name} YUV444P10 32/16 analysed on 8 bits', ff, ['--frames', '8', '--format', 'YUV444P10', '--blksize', '32', '--overlap', '16', '--analyse8'] + f),
            (f'{name} GRAY12 pel 1 delta 2', ff, ['--frames', '8', '--format', 'GRAY12', '--pel', '1', '--delta', '2'] + f),
            (f'{name} YUV420PS pel 4', ff, ['--frames', '8', '--format', 'YUV420PS', '--pel', '4'] + f),
            (f'{name} YUV444PS 8/4 mvu vectors', ff, ['--frames', '8', '--format', 'YUV444PS', '--blksize', '8', '--overlap', '4', '--vectors', 'mvu'] + f),
            (f'{name} GRAYS const', ff, ['--frames', '15', '--format', 'GRAYS', '--vectors', 'const'] + f),
            (f'4K {name} 16/8 pel 4', k4, ['--frames', '6', '--pel', '4'] + f),
        ]
    cases += [
        # Their arguments
        ('flow delta -1 time 50', ff, ['--frames', '8', '--delta', '-1', '--time', '50'] + flow),
        ('flow time 0', ff, ['--frames', '8', '--time', '0'] + flow),
        ('flow time 33.3 delta 3 pel 4', ff, ['--frames', '10', '--time', '33.3', '--delta', '3', '--pel', '4'] + flow),
        ('flow const delta -2 YUV420P16', ff, ['--frames', '15', '--vectors', 'const', '--delta', '-2', '--format', 'YUV420P16'] + flow),
        ('blur 200 prec 2', ff, ['--frames', '8', '--blur', '200', '--prec', '2'] + blur),
        ('blur 0', ff, ['--frames', '8', '--blur', '0'] + blur),
        ('blur 150 prec 5 pel 4 YUV444P16', ff, ['--frames', '8', '--blur', '150', '--prec', '5', '--pel', '4', '--format', 'YUV444P16'] + blur),
        ('blur 200 const delta 2', ff, ['--frames', '15', '--blur', '200', '--vectors', 'const', '--delta', '2'] + blur),
        ('blur 120 YUV420PS', ff, ['--frames', '8', '--blur', '120', '--format', 'YUV420PS'] + blur),
        ('compensate delta -1 time 50 thsad 300', ff, ['--frames', '8', '--delta', '-1', '--time', '50', '--thsad', '300'] + comp),
        ('compensate thsad 0', ff, ['--frames', '8', '--thsad', '0'] + comp),
        ('compensate thsad 100000 pel 4', ff, ['--frames', '8', '--thsad', '100000', '--pel', '4'] + comp),
        ('compensate time 25 delta 2 8/4', ff, ['--frames', '10', '--time', '25', '--delta', '2', '--blksize', '8', '--overlap', '4'] + comp),
        ('compensate thsad 400 YUV420PS time 75', ff, ['--frames', '8', '--thsad', '400', '--format', 'YUV420PS', '--time', '75'] + comp),
        ('compensate const delta -3 444 32/16', ff, ['--frames', '15', '--vectors', 'const', '--delta', '-3', '--format', 'YUV444P8', '--blksize', '32',
                                                     '--overlap', '16'] + comp),
    ]
    return cases


def recalculate_cases():
    ff, ob, k4 = 'football_fast_s3', 'objects1080fast_s3', 'c0065_s3'
    t0 = ['--thsad', '0']  # every block searched
    return [
        ('16/8 pel 2', ff, ['--frames', '6']),
        ('thsad 50', ff, ['--frames', '6', '--thsad', '50']),
        ('search 0 searchparam 4', ff, ['--frames', '6', '--search', '0', '--searchparam', '4'] + t0),
        ('search 1 searchparam 3', ob, ['--frames', '6', '--search', '1', '--searchparam', '3'] + t0),
        ('search 2 searchparam 1', ff, ['--frames', '6', '--search', '2', '--searchparam', '1'] + t0),
        ('search 2 searchparam 8', ff, ['--frames', '6', '--search', '2', '--searchparam', '8'] + t0),
        ('search 3 searchparam 8', ff, ['--frames', '6', '--search', '3', '--searchparam', '8'] + t0),
        ('search 4 searchparam 3', ff, ['--frames', '6', '--search', '4', '--searchparam', '3'] + t0),
        ('search 5', ob, ['--frames', '6', '--search', '5'] + t0),
        ('smooth 0 delta -2', ff, ['--frames', '6', '--smooth', '0', '--delta', '-2'] + t0),
        ('from 32/16 thsad 100', ff, ['--frames', '6', '--old-blksize', '32', '--old-overlap', '16', '--thsad', '100']),
        ('from 8/4 smooth 0', ff, ['--frames', '6', '--old-blksize', '8', '--old-overlap', '4', '--smooth', '0']),
        ('8/4 from 32/16 delta 2', ff, ['--frames', '8', '--blksize', '8', '--overlap', '4', '--old-blksize', '32', '--old-overlap', '16', '--delta', '2'] + t0),
        ('random vectors', ff, ['--frames', '6', '--vectors', 'random'] + t0),
        ('random 444 pel 4 search 3', ff, ['--frames', '6', '--vectors', 'random', '--format', 'YUV444P8', '--pel', '4', '--search', '3'] + t0),
        ('mvu from 32/16 search 3', ff, ['--frames', '6', '--vectors', 'mvu', '--old-blksize', '32', '--old-overlap', '16', '--search', '3']),
        ('chroma 0 mvlambda 3000 pnew 100', ff, ['--frames', '6', '--chroma', '0', '--mvlambda', '3000', '--pnew', '100'] + t0),
        ('mvlambda 0 pnew 0 pel 4', ff, ['--frames', '6', '--mvlambda', '0', '--pnew', '0', '--pel', '4'] + t0),
        ('32/16 pel 4 search 0', ff, ['--frames', '6', '--blksize', '32', '--overlap', '16', '--pel', '4', '--search', '0', '--searchparam', '8'] + t0),
        ('16/6 pel 4 1914x1074', ff, ['--frames', '6', '--blksize', '16', '--overlap', '6', '--pel', '4', '--crop', '1914x1074'] + t0),
        ('16/0 pel 1', ff, ['--frames', '6', '--overlap', '0', '--pel', '1'] + t0),
        ('YUV420P16 pel 4', ff, ['--frames', '6', '--format', 'YUV420P16', '--pel', '4'] + t0),
        ('YUV444P10 32/16 search 0', ff, ['--frames', '6', '--format', 'YUV444P10', '--blksize', '32', '--overlap', '16', '--search', '0', '--searchparam', '8'] + t0),
        ('YUV420P12 16/6 pel 4 1914x1074', ff, ['--frames', '6', '--format', 'YUV420P12', '--blksize', '16', '--overlap', '6', '--pel', '4', '--crop', '1914x1074'] + t0),
        ('GRAY8 8/4 pel 1', ff, ['--frames', '6', '--format', 'GRAY8', '--blksize', '8', '--overlap', '4', '--pel', '1'] + t0),
        ('GRAY16 mvu vectors', ff, ['--frames', '6', '--format', 'GRAY16', '--vectors', 'mvu'] + t0),
        ('YUV420PS pel 1', ff, ['--frames', '6', '--format', 'YUV420PS', '--pel', '1'] + t0),
        ('YUV444PS pel 1 search 3', ff, ['--frames', '6', '--format', 'YUV444PS', '--pel', '1', '--search', '3'] + t0),
        ('GRAYS 32/16 pel 1', ff, ['--frames', '6', '--format', 'GRAYS', '--blksize', '32', '--overlap', '16', '--pel', '1'] + t0),
        ('4K 16/8 pel 4', k4, ['--frames', '4', '--pel', '4'] + t0),
    ]


# The smoke suite: these cases of the others
SMOKE = {
    'super': ['--pel 4', '--format YUV420PS', '--pel 1 --format YUV444P16', '--pelclip'],
    'analyse': ['16/8 pel 2', '8/4 pel 4', '444 32/16 pel 4', 'YUV420P16 16/8 pel 4', 'YUV420PS 16/8 pel 2', 'GRAY8 16/8 pel 2', 'pel 1 16/8'],
    'degrain': ['YUV420P8 16/8 pel 2 mvgpu vectors', 'radius 3 444 32/16 pel 4', 'YUV420P16 16/8 pel 4 analysed on 8 bits', 'YUV420PS 16/8 pel 2 mvgpu vectors',
                'thscd1 150 thscd2 20', 'pel 1 16/8'],
    'flow': ['inter YUV420P8 16/8 pel 2 mvgpu vectors', 'fps YUV444P8 16/8 pel 4 mvgpu vectors', 'inter const', 'inter YUV420PS 16/8 pel 4', 'fps pel 1 8/4'],
    'masks': ['16/8 pel 2', 'random vectors delta -1', 'YUV420PS'],
    'motion': ['flow 16/8 pel 2', 'blur 444 16/8 pel 4', 'compensate const', 'compensate YUV420PS pel 4'],
    'recalculate': ['16/8 pel 2', 'search 3 searchparam 8', 'YUV420PS pel 1'],
}


def smoke_cases():
    cases = []
    for suite, labels in SMOKE.items():
        script, make = SUITES[suite]
        found = {c[0]: c for c in make()}
        for label in labels:
            label_, clip, extra = found[label]
            cases.append((f'{suite}: {label_}', clip, extra, script))
    return cases


SUITES = {'super': ('check_super.py', super_cases), 'analyse': ('check_reference.py', analyse_cases), 'degrain': ('check_degrain.py', degrain_cases),
          'flow': ('check_flow.py', flow_cases), 'masks': ('check_masks.py', masks_cases), 'motion': ('check_flow.py', motion_cases),
          'recalculate': ('check_recalculate.py', recalculate_cases), 'smoke': (None, smoke_cases)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('suite', choices=sorted(SUITES))
    ap.add_argument('--clips', default=os.environ.get('MVGPU_TEST_CLIPS', ''), help='the test clips directory (or MVGPU_TEST_CLIPS)')
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    ap.add_argument('--reference', default=os.environ.get('MVGPU_REFERENCE', ''), help='the CPU reference executable (or MVGPU_REFERENCE), for the analyse suite')
    ap.add_argument('--only', help='run only the cases whose label contains this')
    ap.add_argument('--list', action='store_true', help='list the cases')
    args = ap.parse_args()

    script, make = SUITES[args.suite]
    cases = [c for c in make() if not args.only or args.only in c[0]]
    if args.list:
        for case in cases:
            print(f'{case[0]}  [{case[1]}]')
        return
    if not args.clips:
        ap.error('--clips (or MVGPU_TEST_CLIPS) is needed')
    if args.suite in ('analyse', 'smoke') and not args.reference:
        ap.error(f'the {args.suite} suite needs --reference (or MVGPU_REFERENCE)')
    failed = 0
    for case in cases:
        label, clip, extra = case[:3]
        case_script = case[3] if len(case) > 3 else script
        size, _ = CLIPS[clip]
        cmd = [sys.executable, os.path.join(HERE, case_script), '--src', os.path.join(args.clips, clip, 'noisy.nv12'), '--size', size] + extra
        if args.plugin:
            cmd += ['--plugin', args.plugin]
        if case_script == 'check_reference.py':
            cmd += ['--reference', args.reference]
        r = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip() and 'API 3' not in ln and 'Version mismatch' not in ln]
        if r.returncode != 0:
            failed += 1
        print(f'{"ok  " if r.returncode == 0 else "FAIL"} {label:44} {lines[-1] if lines else ""}', flush=True)
    print(f'{args.suite}: {len(cases)} cases, {failed} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
