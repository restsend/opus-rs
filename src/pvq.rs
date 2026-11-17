use crate::compat::with_scratch;
use crate::range_coder::RangeCoder;

#[cfg(not(feature = "std"))]
use crate::compat::Math;

pub const CELT_PVQ_U_DATA: [u32; 1272] = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
    51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97,
    99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135,
    137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173,
    175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211,
    213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249,
    251, 253, 255, 257, 259, 261, 263, 265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285, 287,
    289, 291, 293, 295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315, 317, 319, 321, 323, 325,
    327, 329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351, 13, 25, 41, 61, 85, 113, 145,
    181, 221, 265, 313, 365, 421, 481, 545, 613, 685, 761, 841, 925, 1013, 1105, 1201, 1301, 1405,
    1513, 1625, 1741, 1861, 1985, 2113, 2245, 2381, 2521, 2665, 2813, 2965, 3121, 3281, 3445, 3613,
    3785, 3961, 4141, 4325, 4513, 4705, 4901, 5101, 5305, 5513, 5725, 5941, 6161, 6385, 6613, 6845,
    7081, 7321, 7565, 7813, 8065, 8321, 8581, 8845, 9113, 9385, 9661, 9941, 10225, 10513, 10805,
    11101, 11401, 11705, 12013, 12325, 12641, 12961, 13285, 13613, 13945, 14281, 14621, 14965,
    15313, 15665, 16021, 16381, 16745, 17113, 17485, 17861, 18241, 18625, 19013, 19405, 19801,
    20201, 20605, 21013, 21425, 21841, 22261, 22685, 23113, 23545, 23981, 24421, 24865, 25313,
    25765, 26221, 26681, 27145, 27613, 28085, 28561, 29041, 29525, 30013, 30505, 31001, 31501,
    32005, 32513, 33025, 33541, 34061, 34585, 35113, 35645, 36181, 36721, 37265, 37813, 38365,
    38921, 39481, 40045, 40613, 41185, 41761, 42341, 42925, 43513, 44105, 44701, 45301, 45905,
    46513, 47125, 47741, 48361, 48985, 49613, 50245, 50881, 51521, 52165, 52813, 53465, 54121,
    54781, 55445, 56113, 56785, 57461, 58141, 58825, 59513, 60205, 60901, 61601, 63, 129, 231, 377,
    575, 833, 1159, 1561, 2047, 2625, 3303, 4089, 4991, 6017, 7175, 8473, 9919, 11521, 13287,
    15225, 17343, 19649, 22151, 24857, 27775, 30913, 34279, 37881, 41727, 45825, 50183, 54809,
    59711, 64897, 70375, 76153, 82239, 88641, 95367, 102425, 109823, 117569, 125671, 134137,
    142975, 152193, 161799, 171801, 182207, 193025, 204263, 215929, 228031, 240577, 253575, 267033,
    280959, 295361, 310247, 325625, 341503, 357889, 374791, 392217, 410175, 428673, 447719, 467321,
    487487, 508225, 529543, 551449, 573951, 597057, 620775, 645113, 670079, 695681, 721927, 748825,
    776383, 804609, 833511, 863097, 893375, 924353, 956039, 988441, 1021567, 1055425, 1090023,
    1125369, 1161471, 1198337, 1235975, 1274393, 1313599, 1353601, 1394407, 1436025, 1478463,
    1521729, 1565831, 1610777, 1656575, 1703233, 1750759, 1799161, 1848447, 1898625, 1949703,
    2001689, 2054591, 2108417, 2163175, 2218873, 2275519, 2333121, 2391687, 2451225, 2511743,
    2573249, 2635751, 2699257, 2763775, 2829313, 2895879, 2963481, 3032127, 3101825, 3172583,
    3244409, 3317311, 3391297, 3466375, 3542553, 3619839, 3698241, 3777767, 3858425, 3940223,
    4023169, 4107271, 4192537, 4278975, 4366593, 4455399, 4545401, 4636607, 4729025, 4822663,
    4917529, 5013631, 5110977, 5209575, 5309433, 5410559, 5512961, 5616647, 5721625, 5827903,
    5935489, 6044391, 6154617, 6266175, 6379073, 6493319, 6608921, 6725887, 6844225, 6963943,
    7085049, 7207551, 321, 681, 1289, 2241, 3649, 5641, 8361, 11969, 16641, 22569, 29961, 39041,
    50049, 63241, 78889, 97281, 118721, 143529, 172041, 204609, 241601, 283401, 330409, 383041,
    441729, 506921, 579081, 658689, 746241, 842249, 947241, 1061761, 1186369, 1321641, 1468169,
    1626561, 1797441, 1981449, 2179241, 2391489, 2618881, 2862121, 3121929, 3399041, 3694209,
    4008201, 4341801, 4695809, 5071041, 5468329, 5888521, 6332481, 6801089, 7295241, 7815849,
    8363841, 8940161, 9545769, 10181641, 10848769, 11548161, 12280841, 13047849, 13850241,
    14689089, 15565481, 16480521, 17435329, 18431041, 19468809, 20549801, 21675201, 22846209,
    24064041, 25329929, 26645121, 28010881, 29428489, 30899241, 32424449, 34005441, 35643561,
    37340169, 39096641, 40914369, 42794761, 44739241, 46749249, 48826241, 50971689, 53187081,
    55473921, 57833729, 60268041, 62778409, 65366401, 68033601, 70781609, 73612041, 76526529,
    79526721, 82614281, 85790889, 89058241, 92418049, 95872041, 99421961, 103069569, 106816641,
    110664969, 114616361, 118672641, 122835649, 127107241, 131489289, 135983681, 140592321,
    145317129, 150160041, 155123009, 160208001, 165417001, 170752009, 176215041, 181808129,
    187533321, 193392681, 199388289, 205522241, 211796649, 218213641, 224775361, 231483969,
    238341641, 245350569, 252512961, 259831041, 267307049, 274943241, 282741889, 290705281,
    298835721, 307135529, 315607041, 324252609, 333074601, 342075401, 351257409, 360623041,
    370174729, 379914921, 389846081, 399970689, 410291241, 420810249, 431530241, 442453761,
    453583369, 464921641, 476471169, 488234561, 500214441, 512413449, 524834241, 537479489,
    550351881, 563454121, 576788929, 590359041, 604167209, 618216201, 632508801, 1683, 3653, 7183,
    13073, 22363, 36365, 56695, 85305, 124515, 177045, 246047, 335137, 448427, 590557, 766727,
    982729, 1244979, 1560549, 1937199, 2383409, 2908411, 3522221, 4235671, 5060441, 6009091,
    7095093, 8332863, 9737793, 11326283, 13115773, 15124775, 17372905, 19880915, 22670725,
    25765455, 29189457, 32968347, 37129037, 41699767, 46710137, 52191139, 58175189, 64696159,
    71789409, 79491819, 87841821, 96879431, 106646281, 117185651, 128542501, 140763503, 153897073,
    167993403, 183104493, 199284183, 216588185, 235074115, 254801525, 275831935, 298228865,
    322057867, 347386557, 374284647, 402823977, 433078547, 465124549, 499040399, 534906769,
    572806619, 612825229, 655050231, 699571641, 746481891, 795875861, 847850911, 902506913,
    959946283, 1020274013, 1083597703, 1150027593, 1219676595, 1292660325, 1369097135, 1449108145,
    1532817275, 1620351277, 1711839767, 1807415257, 1907213187, 2011371957, 2120032959, 8989,
    19825, 40081, 75517, 134245, 227305, 369305, 579125, 880685, 1303777, 1884961, 2668525,
    3707509, 5064793, 6814249, 9041957, 11847485, 15345233, 19665841, 24957661, 31388293, 39146185,
    48442297, 59511829, 72616013, 88043969, 106114625, 127178701, 151620757, 179861305, 212358985,
    249612805, 292164445, 340600625, 395555537, 457713341, 527810725, 606639529, 695049433,
    793950709, 904317037, 1027188385, 1163673953, 1314955181, 1482288821, 1667010073, 1870535785,
    2094367717, 48639, 108545, 224143, 433905, 795455, 1392065, 2340495, 3800305, 5984767, 9173505,
    13726991, 20103025, 28875327, 40754369, 56610575, 77500017, 104692735, 139703809, 184327311,
    240673265, 311207743, 398796225, 506750351, 638878193, 799538175, 993696769, 1226990095,
    1505789553, 1837271615, 2229491905, 265729, 598417, 1256465, 2485825, 4673345, 8405905,
    14546705, 24331777, 39490049, 62390545, 96220561, 145198913, 214828609, 312193553, 446304145,
    628496897, 872893441, 1196924561, 1621925137, 2173806145, 1462563, 3317445, 7059735, 14218905,
    27298155, 50250765, 89129247, 152951073, 254831667, 413442773, 654862247, 1014889769,
    1541911931, 2300409629, 3375210671, 8097453, 18474633, 39753273, 81270333, 158819253,
    298199265, 540279585, 948062325, 1616336765, 45046719, 103274625, 224298231, 464387817,
    921406335, 1759885185, 3248227095, 251595969, 579168825, 1267854873, 2653649025, 1409933619,
];

const CELT_PVQ_U_ROW: [u32; 15] = [
    0, 176, 351, 525, 698, 870, 1041, 1131, 1178, 1207, 1226, 1240, 1248, 1254, 1257,
];

/// Largest column (max(n, k)) covered by each row of `CELT_PVQ_U_DATA`. Row `r`
/// holds `U(min, max)` for `max` in `[r, CMAX[r]]`; a lookup with a larger
/// `max` would alias the next row's block, so it must be computed instead.
const CELT_PVQ_U_CMAX: [usize; 15] = [
    176, 176, 176, 176, 176, 176, 96, 54, 37, 28, 24, 19, 18, 16, 14,
];

/// Maximum number of pulses per band (`CELT_MAX_PULSES` in libopus).
pub const MAX_PVQ_K: usize = 128;
const MAX_PVQ_U: usize = MAX_PVQ_K + 2;
pub const MAX_PVQ_N: usize = 352;

/// `saturating_add(a, b, c)`, clamped to `u32::MAX` so an out-of-domain
/// `(n, k)` can never silently wrap to a plausible-but-wrong codebook size.
#[inline(always)]
fn saturating_add3(a: u32, b: u32, c: u32) -> u32 {
    let s = a as u64 + b as u64 + c as u64;
    s.min(u32::MAX as u64) as u32
}

#[inline(always)]
pub fn celt_pvq_u_lookup(n: u32, k: u32) -> u32 {
    let r = n.min(k) as usize;
    let c = n.max(k) as usize;

    if r >= CELT_PVQ_U_ROW.len() {
        return compute_u(n, k);
    }
    if c > CELT_PVQ_U_CMAX[r] {
        return compute_u(n, k);
    }
    CELT_PVQ_U_DATA[CELT_PVQ_U_ROW[r] as usize + c]
}

pub fn ncwrs(n: u32, k: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return if k > 0 { 2 } else { 1 };
    }
    if k > MAX_PVQ_K as u32 {
        return u32::MAX;
    }
    let mut u = [0u32; MAX_PVQ_U];
    u[0] = 0;
    u[1] = 1;
    for ki in 2..=(k + 1) as usize {
        u[ki] = ((ki as u32 * 2) as u64).saturating_sub(1) as u32;
    }
    let mut curr_n = n;
    while curr_n > 2 {
        unext(&mut u[1..], (k + 1) as usize, 1);
        curr_n -= 1;
    }
    u[k as usize].saturating_add(u[k as usize + 1])
}

fn compute_u(n: u32, k: u32) -> u32 {
    if n == 0 {
        return if k == 0 { 1 } else { 0 };
    }
    if n == 1 {
        return 1;
    }
    if k > MAX_PVQ_K as u32 {
        return u32::MAX;
    }
    let mut u = [0u32; MAX_PVQ_U];
    u[0] = 0;
    u[1] = 1;
    for ki in 2..=(k + 1) as usize {
        u[ki] = ((ki as u32 * 2) as u64).saturating_sub(1) as u32;
    }
    let mut curr_n = n;
    while curr_n > 2 {
        unext(&mut u[1..], (k + 1) as usize, 1);
        curr_n -= 1;
    }
    u[k as usize]
}

#[inline(always)]
pub fn celt_pvq_u(n: u32, k: u32) -> u32 {
    celt_pvq_u_lookup(n, k)
}

#[inline(always)]
pub fn celt_pvq_v(n: u32, k: u32) -> u32 {
    celt_pvq_u_lookup(n, k).saturating_add(celt_pvq_u_lookup(n, k + 1))
}

fn unext(u: &mut [u32], len: usize, mut u0: u32) {
    let mut j = 1;
    while j < len {
        let u1 = saturating_add3(u[j], u[j - 1], u0);
        u[j - 1] = u0;
        u0 = u1;
        j += 1;
    }
    u[j - 1] = u0;
}

#[inline(always)]
pub fn icwrs(n: u32, _k: u32, y: &[i32]) -> u32 {
    if n == 1 {
        return if y[0] < 0 { 1 } else { 0 };
    }
    debug_assert!(n >= 2, "icwrs: n must be >= 2");
    let mut j = (n - 1) as usize;

    let mut i: u32 = if y[j] < 0 { 1 } else { 0 };
    let mut k = y[j].unsigned_abs();

    while j > 0 {
        j -= 1;
        let yj = y[j];
        let m = n - j as u32;
        i = i.wrapping_add(celt_pvq_u_lookup(m, k));
        k += yj.unsigned_abs();

        let sign_mask = yj >> 31;
        let lookup = (sign_mask as u32) & celt_pvq_u_lookup(m, k + 1);
        i = i.wrapping_add(lookup);
    }
    i
}

/// Whether `U(n', k')` for every `n' <= n`, `k' <= k` is a plain lookup in
/// `CELT_PVQ_U_DATA` (`CELT_PVQ_U_CMAX` does not grow with the row).
#[inline(always)]
fn in_table(n: u32, k: u32) -> bool {
    let r = n.min(k) as usize;
    r < CELT_PVQ_U_ROW.len() && n.max(k) as usize <= CELT_PVQ_U_CMAX[r]
}

/// The pulses of codeword `i` of `V(n, k)` (cwrs.c cwrsi).
#[inline(always)]
pub fn cwrsi(n: u32, k: u32, i: u32, y: &mut [i32]) {
    debug_assert!(k > 0, "cwrsi: k must be > 0");
    if n == 1 {
        let s = -(i as i32);
        y[0] = ((k as i32) + s) ^ s;
    } else if in_table(n, k + 1) {
        // `U(r, c)` with `r <= c`, as libopus's `CELT_PVQ_U_ROW[r][c]`.
        cwrsi_rows(n, k, i, y, |r, c| {
            CELT_PVQ_U_DATA[CELT_PVQ_U_ROW[r as usize] as usize + c as usize]
        });
    } else {
        cwrsi_rows(n, k, i, y, celt_pvq_u_lookup);
    }
}

/// Lowers `k` to the largest value below it with `U(k, n) <= i` and returns
/// that `U(k, n)` (`do p=CELT_PVQ_U_ROW[--_k][_n]; while(p>_i);`).
#[inline(always)]
fn down(k: &mut u32, n: u32, i: u32, u: &impl Fn(u32, u32) -> u32) -> u32 {
    // U(0, n) = 0 ends the search at k = 0.
    while *k > 0 {
        *k -= 1;
        let p = u(*k, n);
        if p <= i {
            return p;
        }
    }
    0
}

/// cwrsi for `n >= 2`, with `u(r, c)` = `U(r, c)` for `r <= c`.
#[inline(always)]
fn cwrsi_rows(mut n: u32, mut k: u32, mut i: u32, y: &mut [i32], u: impl Fn(u32, u32) -> u32) {
    let mut j = 0;
    while n > 2 {
        if k >= n {
            // Lots of pulses: are the pulses in this dimension negative?
            let p = u(n, k + 1);
            let s = -((i >= p) as i32);
            i -= p & s as u32;
            // Count how many pulses were placed in this dimension.
            let k0 = k;
            let mut p = u(n, n);
            if p > i {
                k = n;
                p = down(&mut k, n, i, &u);
            } else {
                loop {
                    p = u(n, k);
                    if p <= i {
                        break;
                    }
                    k -= 1;
                }
            }
            i -= p;
            y[j] = (k0 as i32 - k as i32 + s) ^ s;
        } else {
            // Lots of dimensions: are there any pulses in this dimension?
            let p = u(k, n);
            let q = u(k + 1, n);
            if p <= i && i < q {
                i -= p;
                y[j] = 0;
            } else {
                let s = -((i >= q) as i32);
                i -= q & s as u32;
                let k0 = k;
                let p = down(&mut k, n, i, &u);
                i -= p;
                y[j] = (k0 as i32 - k as i32 + s) ^ s;
            }
        }
        j += 1;
        n -= 1;
    }
    // n == 2
    let p = k.wrapping_mul(2).wrapping_add(1);
    let s = -((i >= p) as i32);
    i -= p & s as u32;
    let k0 = k;
    k = (i + 1) >> 1;
    if k > 0 {
        i -= 2 * k - 1;
    }
    y[j] = (k0 as i32 - k as i32 + s) ^ s;
    // n == 1
    let s = -(i as i32);
    y[j + 1] = (k as i32 + s) ^ s;
}

#[inline(always)]
pub fn encode_pulses(y: &[i32], n: u32, k: u32, rc: &mut RangeCoder) {
    if k == 0 {
        return;
    }
    let fl = icwrs(n, k, y);
    let ft = celt_pvq_v(n, k);
    debug_assert!(fl < ft, "encode_pulses: fl={fl} >= ft={ft}, n={n}, k={k}");
    rc.enc_uint(fl, ft);
}

#[inline(always)]
pub fn decode_pulses(y: &mut [i32], n: u32, k: u32, rc: &mut RangeCoder) {
    if k == 0 {
        for i in 0..n as usize {
            y[i] = 0;
        }
        return;
    }
    let ft = celt_pvq_v(n, k);
    let fl = rc.dec_uint(ft).min(ft.saturating_sub(1));
    cwrsi(n, k, fl, y);
}

#[allow(non_snake_case)]
fn op_pvq_refine(
    Xn: &[f32],
    iy: &mut [i32],
    iy0: Option<&[i32]>,
    K: i32,
    up: i32,
    margin: i32,
    N: usize,
) -> bool {
    let K8 = (K as f32) * 256.0;

    let mut iysum = 0i32;
    for i in 0..N {
        let tmp = K8 * Xn[i];
        iy[i] = (tmp + 128.0) as i32 >> 8;
        iysum += iy[i];
    }

    if let Some(iy0_ref) = iy0 {
        for i in 0..N {
            let min_val = up * iy0_ref[i] - (margin - 1);
            let max_val = up * iy0_ref[i] + (margin - 1);
            iy[i] = iy[i].clamp(min_val, max_val);
        }
        iysum = iy.iter().sum();
    }

    if (iysum - K).abs() > 32 {
        return true;
    }

    let dir = if iysum < K { 1 } else { -1 };
    let mut remaining = (K - iysum).abs();

    let mut rounding: [f32; 32] = [0.0; 32];
    for i in 0..N {
        rounding[i] = K8 * Xn[i] - ((iy[i] as f32) * 256.0);
    }

    while remaining > 0 {
        let mut best_i = 0;
        let mut best_round = if dir == 1 { -1e30f32 } else { 1e30f32 };

        for i in 0..N {
            let can_adjust = if dir == 1 {
                iy0.is_none_or(|iy0_ref| (iy[i] - up * iy0_ref[i]).abs() < (margin - 1))
            } else {
                iy[i] != 0
                    && iy0.is_none_or(|iy0_ref| (iy[i] - up * iy0_ref[i]).abs() < (margin - 1))
            };

            if can_adjust
                && ((dir == 1 && rounding[i] > best_round)
                    || (dir == -1 && rounding[i] < best_round && iy[i] != 0))
            {
                best_round = rounding[i];
                best_i = i;
            }
        }

        iy[best_i] += dir;
        rounding[best_i] -= dir as f32 * 256.0;
        remaining -= 1;
    }

    false
}

pub fn pvq_search_qext(
    x: &[f32],
    y: &mut [i32],
    up_y: &mut [i32],
    refine: &mut [i32],
    k: i32,
    extra_bits: i32,
    n: usize,
) -> f32 {
    debug_assert!(n <= 32);
    debug_assert!(extra_bits >= 2);

    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x[i].abs();
    }

    if sum < 1e-15 {
        y[0] = k;
        up_y[0] = ((1 << extra_bits) - 1) * k;
        for i in 1..n {
            y[i] = 0;
            up_y[i] = 0;
            refine[i] = 0;
        }
        refine[0] = 0;
        return (up_y[0] as f32) * (up_y[0] as f32);
    }

    #[allow(non_snake_case)]
    let mut Xn: [f32; 32] = [0.0; 32];
    let rcp_sum = 1.0 / sum;
    for i in 0..n {
        Xn[i] = x[i].abs() * rcp_sum;
    }

    let failed1 = op_pvq_refine(&Xn, y, None, k, 1, k + 1, n);

    let up = (1 << extra_bits) - 1;
    let up_k = up * k;
    let margin = up;
    let failed2 = op_pvq_refine(&Xn, up_y, Some(y), up_k, up, margin, n);

    if failed1 || failed2 {
        y[0] = k;
        up_y[0] = up_k;
        for i in 1..n {
            y[i] = 0;
            up_y[i] = 0;
        }
    }

    for i in 0..n {
        refine[i] = up_y[i] - up * y[i];
    }

    let mut yy = 0.0f32;
    for i in 0..n {
        yy += (up_y[i] as f32) * (up_y[i] as f32);
    }

    for i in 0..n {
        if x[i] < 0.0 {
            y[i] = -y[i];
            up_y[i] = -up_y[i];
            refine[i] = -refine[i];
        }
    }

    yy
}

#[inline(always)]
fn pvq_search_n2(x: &[f32], y: &mut [i32], k: i32) {
    debug_assert!(x.len() >= 2 && y.len() >= 2);

    let abs_x0 = x[0].abs();
    let abs_x1 = x[1].abs();
    let sum = abs_x0 + abs_x1;

    if sum < 1e-15 {
        y[0] = k;
        y[1] = 0;
        return;
    }

    let rcp_sum = 1.0 / sum;
    let y0 = (k as f32 * abs_x0 * rcp_sum + 0.5).floor() as i32;
    let y0 = y0.clamp(0, k);
    let y1 = k - y0;

    y[0] = if x[0] >= 0.0 { y0 } else { -y0 };
    y[1] = if x[1] >= 0.0 { y1 } else { -y1 };
}

/// The position the next pulse goes to in libopus `op_pvq_search_c`: the
/// scan keeps the first position whose score beats every earlier one, where
/// `j` beats the best so far when `best_den * Rxy^2 > Ryy * best_num`
/// (maximising Rxy/sqrt(Ryy) without a division).
///
/// It scores eight positions at a time and compares them all with the best
/// so far; the lowest one that beats it is the next position the sequential
/// scan would pick, and the later ones are compared again with that one, so
/// both take the same decisions. `abs_x` and `y2` hold a multiple of 8
/// values; padding with `abs_x = 0` and `y2 = inf` never wins.
#[inline(always)]
fn best_pulse_position(abs_x: &[f32], y2: &[f32], xy: f32, yy: f32) -> usize {
    // (numerator, denominator, position) of the best score so far.
    let rxy = xy + abs_x[0];
    let mut best = (rxy * rxy, yy + y2[0], 0);
    for (b, (a, y)) in abs_x.chunks_exact(8).zip(y2.chunks_exact(8)).enumerate() {
        let a: &[f32; 8] = a.try_into().unwrap();
        let y: &[f32; 8] = y.try_into().unwrap();
        let mut num = [0.0f32; 8];
        let mut den = [0.0f32; 8];
        for l in 0..8 {
            let rxy = xy + a[l];
            num[l] = rxy * rxy;
            den[l] = yy + y[l];
        }
        let mut from = if b == 0 { 1 } else { 0 };
        loop {
            let mut beats = 0u32;
            for l in 0..8 {
                beats |= ((best.1 * num[l] > den[l] * best.0) as u32) << l;
            }
            beats &= !0u32 << from;
            if beats == 0 {
                break;
            }
            let l = beats.trailing_zeros() as usize;
            best = (num[l], den[l], 8 * b + l);
            from = l + 1;
        }
    }
    best.2
}

pub fn pvq_search(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    // Hard contract: n must fit the scratch buffers and the input slices must
    // hold n samples (issue #27 deep scan).
    assert!(
        n <= MAX_PVQ_N && x.len() >= n && y.len() >= n,
        "pvq_search: n={} exceeds MAX_PVQ_N={} or slices too short",
        n,
        MAX_PVQ_N
    );
    if n == 0 {
        return;
    }
    if n <= 4 {
        let (mut abs_x, mut y2) = ([0.0; 4], [0.0; 4]);
        op_pvq_search(
            &x[..n],
            &mut y[..n],
            k,
            &mut abs_x[..n],
            &mut y2[..n],
            |a, y, xy, yy| best_pulse_position_in_order(a, y, xy, yy),
        );
        return;
    }
    // Padded to whole blocks for best_pulse_position.
    let padded = n.next_multiple_of(8);
    with_scratch(padded, |abs_x: &mut [f32]| {
        with_scratch(padded, |y2: &mut [f32]| {
            y2[n..].fill(f32::INFINITY);
            op_pvq_search(&x[..n], &mut y[..n], k, abs_x, y2, best_pulse_position)
        })
    });
}

/// [`best_pulse_position`] one position at a time, for short bands.
#[inline(always)]
fn best_pulse_position_in_order(abs_x: &[f32], y2: &[f32], xy: f32, yy: f32) -> usize {
    let rxy = xy + abs_x[0];
    let mut best = (rxy * rxy, yy + y2[0], 0);
    for (j, (&a, &y)) in abs_x.iter().zip(y2).enumerate().skip(1) {
        scan_position(&mut best, j, a, y, xy, yy);
    }
    best.2
}

/// One step of the scan: position `j` replaces `best` (numerator,
/// denominator, position) when its score beats it.
#[inline(always)]
fn scan_position(best: &mut (f32, f32, usize), j: usize, a: f32, y: f32, xy: f32, yy: f32) {
    let rxy = xy + a;
    let ryy = yy + y;
    let rxy = rxy * rxy;
    if best.1 * rxy > ryy * best.0 {
        *best = (rxy, ryy, j);
    }
}

/// libopus `op_pvq_search_c` (float build): the `k` pulses `iy` whose
/// direction best matches `x`, returning their squared norm. `abs_x` and
/// `y2` are zeroed scratch buffers of at least `x.len()` values, padded as
/// `best_position` needs (`abs_x_padded[..n]` and `y2_padded[..n]` hold
/// the values, the rest stays as it is).
#[inline(always)]
fn op_pvq_search(
    x: &[f32],
    iy: &mut [i32],
    k: i32,
    abs_x_padded: &mut [f32],
    y2_padded: &mut [f32],
    best_position: impl Fn(&[f32], &[f32], f32, f32) -> usize,
) -> f32 {
    const EPSILON: f32 = 1e-15;
    let n = x.len();
    let iy = &mut iy[..n];
    let abs_x = &mut abs_x_padded[..n];
    let y2 = &mut y2_padded[..n];

    // Get rid of the sign.
    for ((a, i), &v) in abs_x.iter_mut().zip(iy.iter_mut()).zip(x) {
        *a = v.abs();
        *i = 0;
    }

    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    let mut pulses_left = k;

    // Do a pre-search by projecting on the pyramid.
    if k > (n >> 1) as i32 {
        let mut sum = 0.0f32;
        for &a in abs_x.iter() {
            sum += a;
        }
        // If X is too small, just replace it with a pulse at 0. This also
        // keeps infinities and NaNs from allocating too many pulses.
        if !(sum > EPSILON && sum < 64.0) {
            abs_x[0] = 1.0;
            abs_x[1..].fill(0.0);
            sum = 1.0;
        }
        // Using K+e with e < 1 guarantees we cannot get more than K pulses.
        let rcp = (k as f32 + 0.8) * (1.0 / sum);
        for ((i, y), &a) in iy.iter_mut().zip(y2.iter_mut()).zip(abs_x.iter()) {
            // floor() of the non-negative product, as a truncating cast.
            *i = (rcp * a) as i32;
            *y = *i as f32;
            yy += *y * *y;
            xy += a * *y;
            *y *= 2.0;
            pulses_left -= *i;
        }
    }

    // This should never happen, but just in case it does (e.g. on silence)
    // we fill the first bin with pulses.
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy += tmp * tmp;
        yy += tmp * y2[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    for _ in 0..pulses_left {
        // The squared magnitude term gets added anyway, so add it outside the loop.
        yy += 1.0;
        let best_id = best_position(abs_x_padded, y2_padded, xy, yy);
        let (abs_x, y2) = (&abs_x_padded[..n], &mut y2_padded[..n]);
        xy += abs_x[best_id];
        yy += y2[best_id];
        y2[best_id] += 2.0;
        iy[best_id] += 1;
    }

    // Put the original sign back.
    for (i, &v) in iy.iter_mut().zip(x) {
        let s = (v < 0.0) as i32;
        *i = (*i ^ -s) + s;
    }
    yy
}

#[inline]
fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    let ms = -s;
    if stride == 1 && len >= 2 {
        // The same rotations with the value each step passes to the next
        // kept in a register.
        let x = &mut x[..len];
        let mut x1 = x[0];
        for i in 0..len - 1 {
            let x2 = x[i + 1];
            x[i] = c * x1 + ms * x2;
            x1 = c * x2 + s * x1;
        }
        x[len - 1] = x1;
        let mut x2 = x[len - 2];
        for i in (0..len - 2).rev() {
            let x1 = x[i];
            x[i + 1] = c * x2 + s * x1;
            x2 = c * x1 + ms * x2;
        }
        x[0] = x2;
        return;
    }
    for i in 0..(len - stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }
    if len >= 2 * stride {
        for i in (0..(len - 2 * stride)).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = c * x2 + s * x1;
            x[i] = c * x1 + ms * x2;
        }
    }
}

#[inline(always)]
pub fn exp_rotation(x: &mut [f32], length: usize, dir: i32, stride: usize, k: i32, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    // stride == 0 would divide by zero in the block-size computation below
    // (issue #27 deep scan).
    if stride == 0 || 2 * k >= length as i32 || spread <= 0 || spread > 3 {
        return;
    }
    let factor = SPREAD_FACTOR[spread as usize - 1];
    let gain = (length as f32) / (length as f32 + factor as f32 * k as f32);
    let theta = 0.5 * gain * gain;
    // libopus's float celt_cos_norm, cos((pi/2) x) in double precision; s is
    // the cosine of 1 - theta.
    let cos_norm = |x: f32| (0.5 * core::f64::consts::PI * x as f64).cos() as f32;
    let c = cos_norm(theta);
    let s = cos_norm(1.0 - theta);

    let mut stride2 = 0;
    if length >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < length {
            stride2 += 1;
        }
    }

    let block_len = length / stride;
    for i in 0..stride {
        let x_offset = i * block_len;
        let x_subset = &mut x[x_offset..x_offset + block_len];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(x_subset, block_len, stride2, s, c);
            }
            exp_rotation1(x_subset, block_len, 1, c, s);
        } else {
            exp_rotation1(x_subset, block_len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(x_subset, block_len, stride2, s, -c);
            }
        }
    }
}

#[inline(always)]
pub fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = n / b;
    if n0 == 0 {
        return 0;
    }
    let mut collapse_mask = 0u32;
    for (i, block) in iy[..n0 * b].chunks_exact(n0).enumerate() {
        if block.iter().fold(0, |t, &v| t | v) != 0 {
            collapse_mask |= 1 << i;
        }
    }
    collapse_mask
}

pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let x = &mut x[..n];
    // EPSILON after the sum of squares (vq.c:704).
    let e = 1e-15 + crate::pitch::inner_prod(x, x, n);
    let g = gain * (1.0 / e.sqrt());
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// `x[..n]` = the pulses `y` scaled to length `gain`.
#[inline(always)]
fn alg_quant_resynth(y: &[i32], x: &mut [f32], n: usize, gain: f32) {
    let (y, x) = (&y[..n], &mut x[..n]);
    // Below 2^24 every partial sum of the integer squares is exact in f32,
    // so the in-order float sum equals the integer sum.
    let isum = y.iter().fold(0i64, |s, &v| s + v as i64 * v as i64);
    let ryy = if isum < 1 << 24 {
        isum as f32
    } else {
        y.iter().fold(0.0f32, |s, &v| s + v as f32 * v as f32)
    };
    // normalise_residual (vq.c:150-166): the reciprocal square root
    // times the gain.
    let g = (1.0 / ryy.sqrt()) * gain;
    for (xv, &v) in x.iter_mut().zip(y) {
        *xv = v as f32 * g;
    }
}

fn ec_enc_refine(rc: &mut RangeCoder, refine: i32, up: i32, extra_bits: i32) {
    let half_up = up / 2;
    let large = refine.abs() > half_up;

    rc.encode_bit_logp(large, 1);

    if large {
        rc.enc_bits((refine < 0) as u32, 1);
        rc.enc_bits((refine.abs() - half_up - 1) as u32, (extra_bits - 1) as u32);
    } else {
        rc.enc_bits((refine + half_up) as u32, extra_bits as u32);
    }
}

#[inline]
pub fn alg_quant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    stride: usize,
    rc: &mut RangeCoder,
    gain: f32,
    resynth: bool,
) -> u32 {
    with_scratch(n, |y: &mut [i32]| {
        exp_rotation(x, n, 1, stride, k, spread);
        pvq_search(x, y, k, n);
        let mask = extract_collapse_mask(y, n, stride);
        encode_pulses(y, n as u32, k as u32, rc);
        if resynth {
            alg_quant_resynth(y, x, n, gain);
            exp_rotation(x, n, -1, stride, k, spread);
        }
        mask
    })
}

pub fn alg_quant_qext(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    stride: usize,
    rc: &mut RangeCoder,
    gain: f32,
    resynth: bool,
    extra_bits: Option<i32>,
) -> u32 {
    if n <= 32 {
        let mut y_buf = [0i32; 32];
        let y = &mut y_buf[..n];

        exp_rotation(x, n, 1, stride, k, spread);

        let use_qext = extra_bits.is_some_and(|eb| eb >= 2);

        if use_qext && n == 2 {
            let eb = extra_bits.unwrap();
            pvq_search_n2(x, y, k);
            let mask = extract_collapse_mask(y, n, stride);
            encode_pulses(y, n as u32, k as u32, rc);

            let up = (1 << eb) - 1;
            let abs_x0 = x[0].abs();
            let abs_x1 = x[1].abs();
            let sum = abs_x0 + abs_x1;
            if sum >= 1e-15 {
                let rcp_sum = 1.0 / sum;
                let ideal_y0 = k as f32 * abs_x0 * rcp_sum;
                let actual_y0 = y[0].abs() as f32;
                let refine = ((ideal_y0 - actual_y0) * up as f32).round() as i32;
                ec_enc_refine(rc, refine, up, eb);
            }

            if resynth {
                alg_quant_resynth(y, x, n, gain);
                exp_rotation(x, n, -1, stride, k, spread);
            }
            return mask;
        }

        if use_qext && n > 2 && n <= 32 {
            let eb = extra_bits.unwrap();
            let mut up_y = [0i32; 32];
            let mut refine = [0i32; 32];
            let _yy = pvq_search_qext(x, y, &mut up_y, &mut refine, k, eb, n);
            let mask = extract_collapse_mask(&up_y, n, stride);
            encode_pulses(y, n as u32, k as u32, rc);

            let up = (1 << eb) - 1;
            for i in 0..n - 1 {
                ec_enc_refine(rc, refine[i], up, eb);
            }

            if y[n - 1] == 0 {
                rc.enc_bits((up_y[n - 1] < 0) as u32, 1);
            }

            if resynth {
                alg_quant_resynth(&up_y, x, n, gain);
                exp_rotation(x, n, -1, stride, k, spread);
            }
            return mask;
        }

        pvq_search(x, y, k, n);
        let mask = extract_collapse_mask(y, n, stride);
        encode_pulses(y, n as u32, k as u32, rc);

        if resynth {
            alg_quant_resynth(y, x, n, gain);
            exp_rotation(x, n, -1, stride, k, spread);
        }
        mask
    } else {
        alg_quant(x, n, k, spread, stride, rc, gain, resynth)
    }
}

#[inline]
pub fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    stride: usize,
    rc: &mut RangeCoder,
    gain: f32,
) -> u32 {
    with_scratch(n, |y: &mut [i32]| {
        decode_pulses(y, n as u32, k as u32, rc);
        let mask = extract_collapse_mask(y, n, stride);
        alg_quant_resynth(y, x, n, gain);
        exp_rotation(x, n, -1, stride, k, spread);
        mask
    })
}
