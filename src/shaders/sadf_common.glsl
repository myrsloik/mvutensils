// sadf_common.glsl
//
// mvu's SAD and SATD of float blocks, in exactly the order of float operations mvu's compiled kernels take
// on an AVX-512 CPU (SADFunctions_Float.h and the compiler's vectorization of its C, read from mvu's DLL),
// so that the SADs are mvu's bit for bit. Included by the kernels that measure float SADs, after they
// define
//
//     float DiffF(int p, int x, int y)
//     vec2 DiffUV(int x, int y)
//
// the difference of sample (x, y) of the current block's luma (p 0) and the reference's at the vector,
// current minus reference, and U's and V's differences at (x, y) of its chroma. Every block takes one
// lane: its sums run in the lane in the order mvu's lanes took them, the order of every addition kept
// (precise). U's and V's SADs are each mvu's, but taken in one pass (SadUV), their sums side by side in
// vec2s, component by component exactly the scalar ones: the super holds their samples side by side
// (SuperLayout.h), and a pass per plane read twice the cache lines, which made the float search 10%
// slower.
//
// A plane's SAD (SadRawF), the sum of |DiffF|: blocks 16 or more wide in 16 lanes (sadf_zmm), 8 wide in 8
// (sadf_ymm): lane l sums column l of every chunk of the row, the rows in 4 accumulators by their row
// modulo 4 (blocks of fewer than 4 rows all in the first), (a0 + a1) + (a2 + a3), then the lanes halving,
// lane i with lane i + L/2 first (_mm512_reduce_add_ps, sadf_reduce256); blocks 4 or 2 wide as the
// fast-math C vectorized: each column's rows halving, row i with row i + h/2 first, then the columns
// (c0 + c2) + (c1 + c3). Scaled to the 16-bit range as scale_f32_sad does (ScaledF), each plane's alone.
//
// Luma's SATD (SatdRawF), satd_f32_c: each 4x4 tile's Hadamard transform as it writes it, and the 16
// absolute values of each tile summed as the compiler vectorized the tiles: blocks up to 8 wide tile after
// tile in their own order (T4F); wider ones a row of tiles at a time, 4 lanes each folding every fourth
// tile into its running sum (T16F), the first lane's starting from the rows before, the lanes then
// (l0 + l2) + (l1 + l3). Halved and scaled by one multiply, 32767.5, as the compiler folded them.
//
// Each sample is read at one place in the code, in loops the compiler is asked to keep (dont_unroll): the
// driver inlines DiffF everywhere it is called, and with the lanes, accumulators and tiles written out
// the float kernels took minutes to compile.

float AbsDiffF(int p, int x, int y) {
    precise float d = DiffF(p, x, y);
    return abs(d);
}

// Lane l's sum of a w x h block of plane p (L lanes): column l of every chunk of every row into the
// accumulator of the row modulo 4 (fewer than 4 rows all into the first), then (a0 + a1) + (a2 + a3)
float LaneF(int p, int w, int h, int L, int l) {
    precise float a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    if (h < 4) {
        [[dont_unroll]] for (int y = 0; y < h; ++y)
            for (int c = l; c < w; c += L)
                a0 += AbsDiffF(p, c, y);
    } else {
        [[dont_unroll]] for (int y = 0; y < h; y += 4)
            for (int c = l; c < w; c += L)
                [[dont_unroll]] for (int k = 0; k < 4; ++k) {
                    precise float d = AbsDiffF(p, c, y + k);
                    if (k == 0)
                        a0 += d;
                    else if (k == 1)
                        a1 += d;
                    else if (k == 2)
                        a2 += d;
                    else
                        a3 += d;
                }
    }
    precise float s01 = a0 + a1;
    precise float s23 = a2 + a3;
    precise float t = s01 + s23;
    return t;
}

// Row i of column x of a block 4 or 2 wide, its h rows (2, 4 or 8) halving: the order the rows pair in,
// row i with row i + h/2 first
int ColumnRow(int i, int h) {
    // h 8: 0 4 2 6 1 5 3 7; h 4: 0 2 1 3; h 2: 0 1
    return h == 8 ? ((i & 1) << 2) | (i & 2) | (i >> 2) : h == 4 ? ((i & 1) << 1) | (i >> 1) : i;
}

// Column x of a block 4 or 2 wide: its rows in pairs, then the pairs' sums in pairs, and so on
float ColumnF(int p, int x, int h) {
    float s[8];
    [[dont_unroll]] for (int i = 0; i < h; ++i)
        s[i] = AbsDiffF(p, x, ColumnRow(i, h));
    for (int n = h >> 1; n >= 1; n >>= 1)
        for (int i = 0; i < n; ++i) {
            precise float t = s[2 * i] + s[2 * i + 1];
            s[i] = t;
        }
    return s[0];
}

// The raw float SAD of a w x h block of plane p
float SadRawF(int p, int w, int h) {
    if (w <= 4) {
        // the columns (c0 + c2) + (c1 + c3), or c0 + c1
        float col[4];
        [[dont_unroll]] for (int x = 0; x < w; ++x)
            col[x] = ColumnF(p, x, h);
        if (w == 2) {
            precise float s = col[0] + col[1];
            return s;
        }
        precise float a = col[0] + col[2];
        precise float b = col[1] + col[3];
        precise float s = a + b;
        return s;
    }
    // 8 lanes for blocks 8 wide, 16 for wider ones, then the lanes halving, lane i with lane i + L/2
    int L = w == 8 ? 8 : 16;
    float t[16];
    [[dont_unroll]] for (int l = 0; l < L; ++l)
        t[l] = LaneF(p, w, h, L, l);
    for (int n = L >> 1; n >= 1; n >>= 1)
        for (int i = 0; i < n; ++i) {
            precise float s = t[i] + t[i + n];
            t[i] = s;
        }
    return t[0];
}

// scale_f32_sad's mapping of a raw sum to the 16-bit range (the SATD's halving folded into its scale)
uint ScaledF(float s, float scale) {
    precise float v = s * scale;
    if (!(v > 0.0))
        return 0u;
    if (v >= 4294967040.0)
        return 0xFFFFFFFFu;
    precise float r = v + 0.5;
    return uint(r);
}

uint SadF(int p, int w, int h) {
    return ScaledF(SadRawF(p, w, h), 65535.0);
}

// The same for U and V at once, component by component: LaneF, ColumnF and SadRawF of DiffUV
vec2 AbsDiffUV(int x, int y) {
    precise vec2 d = DiffUV(x, y);
    return abs(d);
}

vec2 LaneUV(int w, int h, int L, int l) {
    precise vec2 a0 = vec2(0.0), a1 = vec2(0.0), a2 = vec2(0.0), a3 = vec2(0.0);
    if (h < 4) {
        [[dont_unroll]] for (int y = 0; y < h; ++y)
            for (int c = l; c < w; c += L)
                a0 += AbsDiffUV(c, y);
    } else {
        [[dont_unroll]] for (int y = 0; y < h; y += 4)
            for (int c = l; c < w; c += L)
                [[dont_unroll]] for (int k = 0; k < 4; ++k) {
                    precise vec2 d = AbsDiffUV(c, y + k);
                    if (k == 0)
                        a0 += d;
                    else if (k == 1)
                        a1 += d;
                    else if (k == 2)
                        a2 += d;
                    else
                        a3 += d;
                }
    }
    precise vec2 s01 = a0 + a1;
    precise vec2 s23 = a2 + a3;
    precise vec2 t = s01 + s23;
    return t;
}

vec2 ColumnUV(int x, int h) {
    vec2 s[8];
    [[dont_unroll]] for (int i = 0; i < h; ++i)
        s[i] = AbsDiffUV(x, ColumnRow(i, h));
    for (int n = h >> 1; n >= 1; n >>= 1)
        for (int i = 0; i < n; ++i) {
            precise vec2 t = s[2 * i] + s[2 * i + 1];
            s[i] = t;
        }
    return s[0];
}

vec2 SadRawUV(int w, int h) {
    if (w <= 4) {
        vec2 col[4];
        [[dont_unroll]] for (int x = 0; x < w; ++x)
            col[x] = ColumnUV(x, h);
        if (w == 2) {
            precise vec2 s = col[0] + col[1];
            return s;
        }
        precise vec2 a = col[0] + col[2];
        precise vec2 b = col[1] + col[3];
        precise vec2 s = a + b;
        return s;
    }
    int L = w == 8 ? 8 : 16;
    vec2 t[16];
    [[dont_unroll]] for (int l = 0; l < L; ++l)
        t[l] = LaneUV(w, h, L, l);
    for (int n = L >> 1; n >= 1; n >>= 1)
        for (int i = 0; i < n; ++i) {
            precise vec2 s = t[i] + t[i + n];
            t[i] = s;
        }
    return t[0];
}

// U's and V's SADs of a w x h block of chroma, each scaled alone
uvec2 SadUV(int w, int h) {
    vec2 s = SadRawUV(w, h);
    return uvec2(ScaledF(s.x, 65535.0), ScaledF(s.y, 65535.0));
}

// The 16 absolute values of luma's tile (tx, ty), c[4 * j + k] the k-th of column j, satd_f32_c's
// |a0 + a2|, |a1 + a3|, |a0 - a2|, |a1 - a3|
void TileF(int tx, int ty, out float c[16]) {
    float t[16]; // t[4 * i + j], row i
    float e[4];
    [[dont_unroll]] for (int i = 0; i < 4; ++i) {
        [[dont_unroll]] for (int j = 0; j < 4; ++j)
            e[j] = DiffF(0, 4 * tx + j, 4 * ty + i);
        precise float a0 = e[0] + e[1], a1 = e[0] - e[1], a2 = e[2] + e[3], a3 = e[2] - e[3];
        precise float t0 = a0 + a2, t1 = a1 + a3, t2 = a0 - a2, t3 = a1 - a3;
        t[4 * i] = t0;
        t[4 * i + 1] = t1;
        t[4 * i + 2] = t2;
        t[4 * i + 3] = t3;
    }
    for (int j = 0; j < 4; ++j) {
        precise float a0 = t[j] + t[4 + j], a1 = t[j] - t[4 + j], a2 = t[8 + j] + t[12 + j], a3 = t[8 + j] - t[12 + j];
        precise float k0 = a0 + a2, k1 = a1 + a3, k2 = a0 - a2, k3 = a1 - a3;
        c[4 * j] = abs(k0);
        c[4 * j + 1] = abs(k1);
        c[4 * j + 2] = abs(k2);
        c[4 * j + 3] = abs(k3);
    }
}

// The tile sums, cjk = c[4 * j + k]
float T4F(float c[16]) {
    precise float a = (c[5] + c[14]) + (c[15] + c[12]);
    precise float b = (c[13] + c[6]) + (c[7] + c[4]);
    precise float d = (c[1] + c[10]) + (c[11] + c[8]);
    precise float e = (c[9] + c[2]) + (c[3] + c[0]);
    precise float ab = a + b;
    precise float de = d + e;
    precise float s = ab + de;
    return s;
}

float T16F(float c[16], float acc) {
    precise float x = c[1] + acc;
    precise float y = c[2] + c[0];
    precise float z = y + x;
    precise float q = c[5] + c[3];
    precise float r = c[4] + q;
    precise float s = r + z;
    precise float m = c[6] + c[7];
    precise float n = c[9] + m;
    precise float o = c[8] + n;
    precise float u = o + s;
    precise float f = c[10] + c[11];
    precise float g = c[13] + f;
    precise float h = c[12] + g;
    precise float i = c[14] + h;
    precise float v = i + u;
    precise float w = c[15] + v;
    return w;
}

// The raw float SATD of luma's w x h block: up to 8 wide tile after tile, wider in 4 lanes, tile tx into
// lane tx % 4
float SatdRawF(int w, int h) {
    precise float total = 0.0;
    float c[16];
    [[dont_unroll]] for (int ty = 0; ty < h / 4; ++ty) {
        float l[4];
        l[0] = total;
        l[1] = 0.0;
        l[2] = 0.0;
        l[3] = 0.0;
        [[dont_unroll]] for (int tx = 0; tx < w / 4; ++tx) {
            TileF(tx, ty, c);
            if (w <= 8)
                total += T4F(c);
            else
                l[tx & 3] = T16F(c, l[tx & 3]);
        }
        if (w > 8) {
            precise float a = l[0] + l[2];
            precise float b = l[1] + l[3];
            total = a + b;
        }
    }
    return total;
}

uint SatdF(int w, int h) {
    return ScaledF(SatdRawF(w, h), 32767.5);
}
