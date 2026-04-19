use crate::range_coder::RangeCoder;
use std::mem::MaybeUninit;

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

#[inline(always)]
pub fn celt_pvq_u_lookup(n: u32, k: u32) -> u32 {
    let r = n.min(k) as usize;
    let c = n.max(k) as usize;

    if r >= CELT_PVQ_U_ROW.len() {
        return compute_u(n, k);
    }
    unsafe {
        let row_base = *CELT_PVQ_U_ROW.get_unchecked(r);
        let idx = row_base as usize + c;
        if idx >= CELT_PVQ_U_DATA.len() {
            return compute_u(n, k);
        }
        *CELT_PVQ_U_DATA.get_unchecked(idx)
    }
}

const MAX_PVQ_K: usize = 128;
const MAX_PVQ_U: usize = MAX_PVQ_K + 2;
pub const MAX_PVQ_N: usize = 352;

pub fn ncwrs(n: u32, k: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return if k > 0 { 2 } else { 1 };
    }
    let mut u = [0u32; MAX_PVQ_U];
    u[0] = 0;
    u[1] = 1;
    for ki in 2..=(k + 1) as usize {
        u[ki] = (ki as u32 * 2).wrapping_sub(1);
    }
    let mut curr_n = n;
    while curr_n > 2 {
        unext(&mut u[1..], (k + 1) as usize, 1);
        curr_n -= 1;
    }
    u[k as usize].wrapping_add(u[k as usize + 1])
}

fn compute_u(n: u32, k: u32) -> u32 {
    if n == 0 {
        return if k == 0 { 1 } else { 0 };
    }
    if n == 1 {
        return 1;
    }
    let mut u = [0u32; MAX_PVQ_U];
    u[0] = 0;
    u[1] = 1;
    for ki in 2..=(k + 1) as usize {
        u[ki] = (ki as u32 * 2).wrapping_sub(1);
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
    celt_pvq_u_lookup(n, k).wrapping_add(celt_pvq_u_lookup(n, k + 1))
}

fn unext(u: &mut [u32], len: usize, mut u0: u32) {
    let mut j = 1;
    while j < len {
        let u1 = u[j].wrapping_add(u[j - 1]).wrapping_add(u0);
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

#[inline(always)]
pub fn cwrsi(n: u32, k: u32, mut i: u32, y: &mut [i32]) {
    debug_assert!(k > 0, "cwrsi: k must be > 0");

    if n == 1 {
        let s = -(i as i32);
        y[0] = ((k as i32) + s) ^ s;
        return;
    }

    let mut curr_n = n;

    let mut curr_k = k as i32;
    let mut j = 0usize;

    while curr_n > 2 {
        if curr_k >= curr_n as i32 {
            let p_kp1 = celt_pvq_u_lookup(curr_n, (curr_k + 1) as u32);
            let s: i32 = if i >= p_kp1 {
                i -= p_kp1;
                -1
            } else {
                0
            };
            let k0 = curr_k;
            let q = celt_pvq_u_lookup(curr_n, curr_n);
            let mut p;
            if q > i {
                curr_k = curr_n as i32;
                loop {
                    curr_k -= 1;
                    p = celt_pvq_u_lookup(curr_n, curr_k.max(0) as u32);
                    if p <= i || curr_k <= 0 {
                        break;
                    }
                }
            } else {
                p = celt_pvq_u_lookup(curr_n, curr_k as u32);
                while p > i && curr_k > 0 {
                    curr_k -= 1;
                    p = celt_pvq_u_lookup(curr_n, curr_k as u32);
                }
            }
            i -= p;
            let val = k0 - curr_k;
            y[j] = (val + s) ^ s;
        } else {
            let p_k = celt_pvq_u_lookup(curr_k as u32, curr_n);
            let p_kp1 = celt_pvq_u_lookup((curr_k + 1) as u32, curr_n);
            if p_k <= i && i < p_kp1 {
                i -= p_k;
                y[j] = 0;
                j += 1;
                curr_n -= 1;
                continue;
            }
            let s: i32 = if i >= p_kp1 {
                i -= p_kp1;
                -1
            } else {
                0
            };
            let k0 = curr_k;

            let mut p;
            loop {
                curr_k -= 1;
                p = celt_pvq_u_lookup(curr_k.max(0) as u32, curr_n);
                if p <= i || curr_k <= 0 {
                    break;
                }
            }
            i -= p;
            let val = k0 - curr_k;
            y[j] = (val + s) ^ s;
        }
        j += 1;
        curr_n -= 1;
    }

    let p2 = (2u32).wrapping_mul(curr_k as u32).wrapping_add(1);
    let s2: i32 = if i >= p2 {
        i -= p2;
        -1
    } else {
        0
    };
    let k0 = curr_k;
    curr_k = ((i + 1) >> 1) as i32;
    if curr_k > 0 {
        i -= 2 * curr_k as u32 - 1;
    }
    y[j] = ((k0 - curr_k) + s2) ^ s2;
    j += 1;

    let s1 = -(i as i32);
    y[j] = (curr_k + s1) ^ s1;
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

#[inline]
fn pvq_search_n4(x: &[f32], y: &mut [i32], k: i32) {
    debug_assert!(x.len() >= 4 && y.len() >= 4);

    if k == 0 {
        y[0] = 0;
        y[1] = 0;
        y[2] = 0;
        y[3] = 0;
        return;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;

        let sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFFu32 as i32));
        let vx = _mm_loadu_ps(x.as_ptr());
        let vabs = _mm_and_ps(vx, sign_mask);

        let vzero_f = _mm_setzero_ps();

        let vneg_mask = _mm_cmplt_ps(vx, vzero_f);

        let vsigns = _mm_and_si128(_mm_castps_si128(vneg_mask), _mm_set1_epi32(1));

        let vabs_x = vabs;
        let mut vy2f = _mm_setzero_ps();
        let mut vy = _mm_setzero_si128();
        let mut xy = 0.0f32;
        let mut yy = 0.0f32;

        let vtwo = _mm_set1_ps(2.0);

        let vone_i = _mm_set1_epi32(1);

        for _ in 0..k {
            let vxy = _mm_set1_ps(xy);
            let vrxy = _mm_add_ps(vabs_x, vxy);
            let vyy1 = _mm_add_ps(vy2f, _mm_set1_ps(yy + 1.0));

            let vscore = _mm_mul_ps(vrxy, _mm_rsqrt_ps(vyy1));

            let s0 = _mm_cvtss_f32(vscore);
            let s1 = _mm_cvtss_f32(_mm_shuffle_ps(vscore, vscore, 0b01_01_01_01));
            let s2 = _mm_cvtss_f32(_mm_shuffle_ps(vscore, vscore, 0b10_10_10_10));
            let s3 = _mm_cvtss_f32(_mm_shuffle_ps(vscore, vscore, 0b11_11_11_11));
            let mut best_score = s0;
            let mut best_i: u32 = 0;
            if s1 > best_score {
                best_score = s1;
                best_i = 1;
            }
            if s2 > best_score {
                best_score = s2;
                best_i = 2;
            }
            if s3 > best_score {
                best_i = 3;
            }
            let _ = best_score;

            let vbest = _mm_set1_epi32(best_i as i32);
            let vlane = _mm_setr_epi32(0, 1, 2, 3);
            let vmask = _mm_castsi128_ps(_mm_cmpeq_epi32(vlane, vbest));

            let vpick_ax = _mm_and_ps(vabs_x, vmask);

            let vpick_ax_hi = _mm_movehl_ps(vpick_ax, vpick_ax);
            let vpick_ax2 = _mm_add_ps(vpick_ax, vpick_ax_hi);
            let vpick_ax3 = _mm_add_ss(vpick_ax2, _mm_shuffle_ps(vpick_ax2, vpick_ax2, 1));
            xy += _mm_cvtss_f32(vpick_ax3);

            let vpick_ryy = _mm_and_ps(vyy1, vmask);
            let vpick_ryy_hi = _mm_movehl_ps(vpick_ryy, vpick_ryy);
            let vpick_ryy2 = _mm_add_ps(vpick_ryy, vpick_ryy_hi);
            let vpick_ryy3 = _mm_add_ss(vpick_ryy2, _mm_shuffle_ps(vpick_ryy2, vpick_ryy2, 1));
            yy = _mm_cvtss_f32(vpick_ryy3);

            let vadd2 = _mm_and_ps(vtwo, vmask);
            vy2f = _mm_add_ps(vy2f, vadd2);

            let vadd1 = _mm_and_si128(vone_i, _mm_castps_si128(vmask));
            vy = _mm_add_epi32(vy, vadd1);
        }

        let vneg_s = _mm_sub_epi32(_mm_setzero_si128(), vsigns);
        let vy_xor = _mm_xor_si128(vy, vneg_s);
        let vy_out = _mm_add_epi32(vy_xor, vsigns);
        _mm_storeu_si128(y.as_mut_ptr() as *mut __m128i, vy_out);
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let ax0 = x[0].abs();
        let ax1 = x[1].abs();
        let ax2 = x[2].abs();
        let ax3 = x[3].abs();
        let s0 = (x[0] < 0.0) as i32;
        let s1 = (x[1] < 0.0) as i32;
        let s2 = (x[2] < 0.0) as i32;
        let s3 = (x[3] < 0.0) as i32;
        let mut xy = 0.0f32;
        let mut yy = 0.0f32;
        let mut y2f0 = 0.0f32;
        let mut y2f1 = 0.0f32;
        let mut y2f2 = 0.0f32;
        let mut y2f3 = 0.0f32;
        let mut y0 = 0i32;
        let mut y1 = 0i32;
        let mut y2 = 0i32;
        let mut y3 = 0i32;
        for _ in 0..k {
            let rxy0 = xy + ax0;
            let sq0 = rxy0 * rxy0;
            let ryy0 = yy + y2f0 + 1.0;
            let rxy1 = xy + ax1;
            let sq1 = rxy1 * rxy1;
            let ryy1 = yy + y2f1 + 1.0;
            let rxy2 = xy + ax2;
            let sq2 = rxy2 * rxy2;
            let ryy2 = yy + y2f2 + 1.0;
            let rxy3 = xy + ax3;
            let sq3 = rxy3 * rxy3;
            let ryy3 = yy + y2f3 + 1.0;
            let mut bsq = sq0;
            let mut bden = ryy0;
            let mut best_i: u32 = 0;
            if bden * sq1 > ryy1 * bsq {
                bsq = sq1;
                bden = ryy1;
                best_i = 1;
            }
            if bden * sq2 > ryy2 * bsq {
                bsq = sq2;
                bden = ryy2;
                best_i = 2;
            }
            if bden * sq3 > ryy3 * bsq {
                best_i = 3;
            }
            let _ = bsq;
            match best_i {
                0 => {
                    xy += ax0;
                    yy = ryy0;
                    y2f0 += 2.0;
                    y0 += 1;
                }
                1 => {
                    xy += ax1;
                    yy = ryy1;
                    y2f1 += 2.0;
                    y1 += 1;
                }
                2 => {
                    xy += ax2;
                    yy = ryy2;
                    y2f2 += 2.0;
                    y2 += 1;
                }
                _ => {
                    xy += ax3;
                    yy = ryy3;
                    y2f3 += 2.0;
                    y3 += 1;
                }
            }
        }
        y[0] = (y0 ^ -s0) + s0;
        y[1] = (y1 ^ -s1) + s1;
        y[2] = (y2 ^ -s2) + s2;
        y[3] = (y3 ^ -s3) + s3;
    }
}

#[inline(always)]
pub fn pvq_search(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    if k == 1 {
        let mut best_i = 0;
        let mut best_abs = x[0].abs();
        for i in 1..n {
            let abs_xi = x[i].abs();
            if abs_xi > best_abs {
                best_abs = abs_xi;
                best_i = i;
            }
        }
        for j in 0..n {
            y[j] = 0;
        }
        let sign: i32 = if x[best_i] >= 0.0 { 1 } else { -1 };
        y[best_i] = sign;
        return;
    }

    if n == 2 {
        pvq_search_n2(x, y, k);
        return;
    }

    if n == 4 {
        pvq_search_n4(x, y, k);
        return;
    }

    if n >= 32 {
        pvq_search_fast_select(x, y, k, n);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if n <= 16 {
        pvq_search_neon(x, y, k, n);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if k > 4 && std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            pvq_search_avx2(x, y, k, n);
        }
        return;
    }

    pvq_search_scalar(x, y, k, n);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pvq_fast_select_init_neon(
    x: &[f32],
    n: usize,
    abs_x: &mut [MaybeUninit<f32>; MAX_PVQ_N],
    signs: &mut [MaybeUninit<i32>; MAX_PVQ_N],
) -> f32 {
    use std::arch::aarch64::*;

    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 16 <= n {
        let vx0 = vld1q_f32(x.as_ptr().add(i));
        let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
        let vx2 = vld1q_f32(x.as_ptr().add(i + 8));
        let vx3 = vld1q_f32(x.as_ptr().add(i + 12));

        let vabs0 = vabsq_f32(vx0);
        let vabs1 = vabsq_f32(vx1);
        let vabs2 = vabsq_f32(vx2);
        let vabs3 = vabsq_f32(vx3);

        vst1q_f32(abs_x.as_mut_ptr().add(i) as *mut f32, vabs0);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 4) as *mut f32, vabs1);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 8) as *mut f32, vabs2);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 12) as *mut f32, vabs3);

        sum_vec = vaddq_f32(sum_vec, vabs0);
        sum_vec = vaddq_f32(sum_vec, vabs1);
        sum_vec = vaddq_f32(sum_vec, vabs2);
        sum_vec = vaddq_f32(sum_vec, vabs3);

        for j in 0..16 {
            signs[i + j].write(if x[i + j] < 0.0 { -1i32 } else { 1i32 });
        }

        i += 16;
    }

    while i + 8 <= n {
        let vx0 = vld1q_f32(x.as_ptr().add(i));
        let vx1 = vld1q_f32(x.as_ptr().add(i + 4));

        let vabs0 = vabsq_f32(vx0);
        let vabs1 = vabsq_f32(vx1);

        vst1q_f32(abs_x.as_mut_ptr().add(i) as *mut f32, vabs0);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 4) as *mut f32, vabs1);

        sum_vec = vaddq_f32(sum_vec, vabs0);
        sum_vec = vaddq_f32(sum_vec, vabs1);

        for j in 0..8 {
            signs[i + j].write(if x[i + j] < 0.0 { -1i32 } else { 1i32 });
        }

        i += 8;
    }

    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vabs = vabsq_f32(vx);
        vst1q_f32(abs_x.as_mut_ptr().add(i) as *mut f32, vabs);
        sum_vec = vaddq_f32(sum_vec, vabs);

        for j in 0..4 {
            signs[i + j].write(if x[i + j] < 0.0 { -1i32 } else { 1i32 });
        }

        i += 4;
    }

    let mut sum = vaddvq_f32(sum_vec);

    for j in i..n {
        let abs_xi = x[j].abs();
        abs_x[j].write(abs_xi);
        sum += abs_xi;
        signs[j].write(if x[j] < 0.0 { -1i32 } else { 1i32 });
    }

    sum
}

#[inline]
pub fn pvq_search_fast_select(x: &[f32], y: &mut [i32], k: i32, n: usize) -> f32 {
    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    y[..n].fill(0);

    if k <= 0 {
        return 0.0;
    }

    let mut abs_x_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let mut signs_mu = [MaybeUninit::<i32>::uninit(); MAX_PVQ_N];

    #[cfg(target_arch = "aarch64")]
    let sum = unsafe { pvq_fast_select_init_neon(x, n, &mut abs_x_mu, &mut signs_mu) };
    #[cfg(not(target_arch = "aarch64"))]
    let sum = {
        let mut s = 0.0f32;
        for i in 0..n {
            abs_x_mu[i].write(x[i].abs());
            signs_mu[i].write(if x[i] < 0.0 { -1i32 } else { 1i32 });
            s += unsafe { abs_x_mu[i].assume_init() };
        }
        s
    };

    let abs_x = unsafe { std::slice::from_raw_parts(abs_x_mu.as_ptr() as *const f32, n) };
    let signs = unsafe { std::slice::from_raw_parts(signs_mu.as_ptr() as *const i32, n) };

    if k > (n >> 1) as i32 && sum > 1e-15 {
        let rcp = (k as f32 + 0.8) / sum;

        let abs_x_ptr = abs_x.as_ptr();
        let y_ptr = y.as_mut_ptr();
        unsafe {
            for i in 0..n {
                let yi = (*abs_x_ptr.add(i) * rcp) as i32;
                *y_ptr.add(i) = yi;
                let yf = yi as f32;
                yy += yf * yf;
                xy += yf * *abs_x_ptr.add(i);
                k -= yi;
            }
        }

        if k > n as i32 + 3 {
            let tmp = k as f32;
            unsafe {
                yy += tmp * tmp + tmp * *y_ptr as f32;
                *y_ptr += k;
            }
            k = 0;
        }
    }

    const BATCH_SIZE: i32 = 4;

    if k < BATCH_SIZE * 2 || n < 16 {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let mut y2f_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
            for i in 0..n {
                y2f_mu[i].write(2.0 * y[i] as f32);
            }
            let y2f = unsafe {
                std::slice::from_raw_parts_mut(y2f_mu.as_mut_ptr() as *mut f32, MAX_PVQ_N)
            };

            let abs_x_ptr = abs_x.as_ptr();
            let y2f_ptr = y2f.as_mut_ptr();
            let y_ptr = y.as_mut_ptr();
            unsafe {
                let n4 = n & !3;
                while k > 0 {
                    yy += 1.0;
                    let vxy = vdupq_n_f32(xy);
                    let vyy = vdupq_n_f32(yy);
                    let mut vmax = vdupq_n_f32(0.0);
                    let mut best_id: usize = 0;

                    let mut i = 0;
                    while i < n4 {
                        let vx = vld1q_f32(abs_x_ptr.add(i));
                        let vy = vld1q_f32(y2f_ptr.add(i));
                        let rxy = vaddq_f32(vx, vxy);
                        let ryy = vaddq_f32(vy, vyy);
                        let inv_sqrt = vrsqrteq_f32(ryy);
                        let score = vmulq_f32(rxy, inv_sqrt);
                        vmax = vmaxq_f32(vmax, score);
                        let sc = std::slice::from_raw_parts(
                            &score as *const float32x4_t as *const f32,
                            4,
                        );
                        let mx = vmaxvq_f32(vmax);
                        for lane in 0..4 {
                            if sc[lane] == mx {
                                best_id = i + lane;
                            }
                        }
                        i += 4;
                    }

                    while i < n {
                        let rxy = xy + *abs_x_ptr.add(i);
                        let ryy = yy + *y2f_ptr.add(i);
                        let score = rxy * (1.0 / ryy.sqrt());
                        let current_max = vmaxvq_f32(vmax);
                        if score > current_max {
                            best_id = i;
                            vmax = vsetq_lane_f32(score, vmax, 0);
                        }
                        i += 1;
                    }

                    xy += *abs_x_ptr.add(best_id);
                    yy += *y2f_ptr.add(best_id);
                    *y2f_ptr.add(best_id) += 2.0;
                    *y_ptr.add(best_id) += 1;
                    k -= 1;
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut y2f = [0.0f32; MAX_PVQ_N];
            let abs_x_ptr = abs_x.as_ptr();
            let y2f_ptr = y2f.as_mut_ptr();
            let y_ptr = y.as_mut_ptr();
            unsafe {
                while k > 0 {
                    yy += 1.0;
                    let rxy0 = xy + *abs_x_ptr;
                    let mut best_id = 0;
                    let mut best_num = rxy0 * rxy0;
                    let mut best_den = yy + *y2f_ptr;
                    let mut i = 1;
                    while i + 1 < n {
                        let rxy1 = xy + *abs_x_ptr.add(i);
                        let ryy1 = yy + *y2f_ptr.add(i);
                        let rxy1_sq = rxy1 * rxy1;
                        if best_den * rxy1_sq > ryy1 * best_num {
                            best_id = i;
                            best_num = rxy1_sq;
                            best_den = ryy1;
                        }
                        let rxy2 = xy + *abs_x_ptr.add(i + 1);
                        let ryy2 = yy + *y2f_ptr.add(i + 1);
                        let rxy2_sq = rxy2 * rxy2;
                        if best_den * rxy2_sq > ryy2 * best_num {
                            best_id = i + 1;
                            best_num = rxy2_sq;
                            best_den = ryy2;
                        }
                        i += 2;
                    }
                    if i < n {
                        let rxy = xy + *abs_x_ptr.add(i);
                        let ryy = yy + *y2f_ptr.add(i);
                        let rxy_sq = rxy * rxy;
                        if best_den * rxy_sq > ryy * best_num {
                            best_id = i;
                        }
                    }
                    xy += *abs_x_ptr.add(best_id);
                    yy += *y2f_ptr.add(best_id);
                    *y2f_ptr.add(best_id) += 2.0;
                    *y_ptr.add(best_id) += 1;
                    k -= 1;
                }
            }
        }
    } else {
        let mut y2f_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];

        let y_ptr = y.as_mut_ptr();
        for i in 0..n {
            unsafe {
                y2f_mu[i].write(2.0 * *y_ptr.add(i) as f32);
            }
        }
        let y2f =
            unsafe { std::slice::from_raw_parts_mut(y2f_mu.as_mut_ptr() as *mut f32, MAX_PVQ_N) };
        let mut scores_mu = [MaybeUninit::<(f32, usize)>::uninit(); MAX_PVQ_N];

        let abs_x_ptr = abs_x.as_ptr();
        let y2f_ptr = y2f.as_mut_ptr();
        while k > 0 {
            let batch = BATCH_SIZE.min(k);

            unsafe {
                for i in 0..n {
                    let rxy = xy + *abs_x_ptr.add(i);
                    let ryy = yy + *y2f_ptr.add(i) + 1.0;
                    let score = rxy * rxy / ryy;
                    scores_mu[i].write((score, i));
                }
            }

            let scores = unsafe {
                std::slice::from_raw_parts_mut(scores_mu.as_mut_ptr() as *mut (f32, usize), n)
            };

            let pos = batch as usize;

            scores.select_nth_unstable_by(pos, |a, b| {
                if a.0 > b.0 {
                    std::cmp::Ordering::Less
                } else if a.0 < b.0 {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });

            unsafe {
                for b in 0..batch as usize {
                    let idx = scores[b].1;
                    xy += *abs_x_ptr.add(idx);
                    yy += *y2f_ptr.add(idx) + 1.0;
                    *y2f_ptr.add(idx) += 2.0;
                    *y_ptr.add(idx) += 1;
                }
            }

            k -= batch;
        }
    }

    unsafe {
        let y_ptr = y.as_mut_ptr();
        for i in 0..n {
            *y_ptr.add(i) *= *signs.as_ptr().add(i);
        }
    }

    yy
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pvq_search_scalar_init_neon(
    x: &[f32],
    n: usize,
    abs_x: &mut [f32; 32],
    sign_x: &mut [i32; 32],
) -> f32 {
    use std::arch::aarch64::*;

    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 16 <= n {
        let vx0 = vld1q_f32(x.as_ptr().add(i));
        let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
        let vx2 = vld1q_f32(x.as_ptr().add(i + 8));
        let vx3 = vld1q_f32(x.as_ptr().add(i + 12));

        let vabs0 = vabsq_f32(vx0);
        let vabs1 = vabsq_f32(vx1);
        let vabs2 = vabsq_f32(vx2);
        let vabs3 = vabsq_f32(vx3);

        vst1q_f32(abs_x.as_mut_ptr().add(i), vabs0);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 4), vabs1);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 8), vabs2);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 12), vabs3);

        sum_vec = vaddq_f32(sum_vec, vabs0);
        sum_vec = vaddq_f32(sum_vec, vabs1);
        sum_vec = vaddq_f32(sum_vec, vabs2);
        sum_vec = vaddq_f32(sum_vec, vabs3);

        for j in 0..16 {
            sign_x[i + j] = (x[i + j] < 0.0) as i32;
        }

        i += 16;
    }

    while i + 8 <= n {
        let vx0 = vld1q_f32(x.as_ptr().add(i));
        let vx1 = vld1q_f32(x.as_ptr().add(i + 4));

        let vabs0 = vabsq_f32(vx0);
        let vabs1 = vabsq_f32(vx1);

        vst1q_f32(abs_x.as_mut_ptr().add(i), vabs0);
        vst1q_f32(abs_x.as_mut_ptr().add(i + 4), vabs1);

        sum_vec = vaddq_f32(sum_vec, vabs0);
        sum_vec = vaddq_f32(sum_vec, vabs1);

        for j in 0..8 {
            sign_x[i + j] = (x[i + j] < 0.0) as i32;
        }

        i += 8;
    }

    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vabs = vabsq_f32(vx);
        vst1q_f32(abs_x.as_mut_ptr().add(i), vabs);
        sum_vec = vaddq_f32(sum_vec, vabs);

        for j in 0..4 {
            sign_x[i + j] = (x[i + j] < 0.0) as i32;
        }

        i += 4;
    }

    let mut sum = vaddvq_f32(sum_vec);

    for j in i..n {
        let xi = x[j];
        let abs_xi = xi.abs();
        abs_x[j] = abs_xi;
        sum += abs_xi;
        sign_x[j] = (xi < 0.0) as i32;
    }

    sum
}

#[inline(always)]
fn pvq_search_small_k(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    debug_assert!(k <= 4 && k > 0);
    debug_assert!(n <= 31);

    let mut abs_x = [0.0f32; 32];
    let mut y2f = [0.0f32; 32];
    let mut sign_x = [0i32; 32];

    unsafe {
        let x_ptr = x.as_ptr();
        let abs_x_ptr = abs_x.as_mut_ptr();
        let sign_ptr = sign_x.as_mut_ptr();
        for i in 0..n {
            let xi = *x_ptr.add(i);
            *abs_x_ptr.add(i) = xi.abs();
            *sign_ptr.add(i) = (xi < 0.0) as i32;
        }
    }

    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    let abs_x_ptr = abs_x.as_ptr();
    let y2f_ptr = y2f.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();
    unsafe {
        for _ in 0..k {
            yy += 1.0;

            let rxy0 = xy + *abs_x_ptr;
            let mut best_id = 0usize;
            let mut best_num = rxy0 * rxy0;
            let mut best_den = yy + *y2f_ptr;

            let mut i = 1;
            while i + 1 < n {
                let rxy1 = xy + *abs_x_ptr.add(i);
                let rxy2 = xy + *abs_x_ptr.add(i + 1);
                let den1 = yy + *y2f_ptr.add(i);
                let den2 = yy + *y2f_ptr.add(i + 1);
                let rxy1_sq = rxy1 * rxy1;
                let rxy2_sq = rxy2 * rxy2;

                if best_den * rxy1_sq > den1 * best_num {
                    best_id = i;
                    best_num = rxy1_sq;
                    best_den = den1;
                }
                if best_den * rxy2_sq > den2 * best_num {
                    best_id = i + 1;
                    best_num = rxy2_sq;
                    best_den = den2;
                }
                i += 2;
            }
            if i < n {
                let rxy = xy + *abs_x_ptr.add(i);
                let rxy_sq = rxy * rxy;
                let den = yy + *y2f_ptr.add(i);
                if best_den * rxy_sq > den * best_num {
                    best_id = i;
                }
            }

            xy += *abs_x_ptr.add(best_id);
            yy += *y2f_ptr.add(best_id);
            *y2f_ptr.add(best_id) += 2.0;
            *y_ptr.add(best_id) += 1;
        }
    }

    unsafe {
        let y_ptr = y.as_mut_ptr();
        let sign_ptr = sign_x.as_ptr();
        for i in 0..n {
            let s = *sign_ptr.add(i);
            *y_ptr.add(i) = (*y_ptr.add(i) ^ -s) + s;
        }
    }
}
#[inline]
fn pvq_search_scalar(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    debug_assert!(n <= 31);
    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    y[..n].fill(0);

    if k <= 0 {
        return;
    }

    if k <= 4 {
        pvq_search_small_k(x, y, k, n);
        return;
    }

    let mut abs_x = [0.0f32; 32];
    let mut y2f = [0.0f32; 32];
    let mut sign_x = [0i32; 32];

    #[cfg(target_arch = "aarch64")]
    let sum = unsafe { pvq_search_scalar_init_neon(x, n, &mut abs_x, &mut sign_x) };
    #[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
    let sum = unsafe {
        if std::arch::is_x86_feature_detected!("avx2") {
            pvq_search_scalar_init_avx2(x, n, &mut abs_x, &mut sign_x)
        } else {
            let mut s = 0.0f32;
            for i in 0..n {
                let xi = x[i];
                let abs_xi = xi.abs();
                abs_x[i] = abs_xi;
                s += abs_xi;
                sign_x[i] = (xi < 0.0) as i32;
            }
            s
        }
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let sum = {
        let mut s = 0.0f32;
        for i in 0..n {
            let xi = x[i];
            let abs_xi = xi.abs();
            abs_x[i] = abs_xi;
            s += abs_xi;
            sign_x[i] = (xi < 0.0) as i32;
        }
        s
    };

    if k > (n >> 1) as i32 && sum > 1e-15 {
        let rcp = (k as f32 + 0.8) / sum;

        let abs_x_ptr = abs_x.as_ptr();
        let y2f_ptr = y2f.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();
        unsafe {
            for i in 0..n {
                let yi = (*abs_x_ptr.add(i) * rcp) as i32;
                *y_ptr.add(i) = yi;
                let yf = yi as f32;
                yy += yf * yf;
                xy += yf * *abs_x_ptr.add(i);
                *y2f_ptr.add(i) = 2.0 * yf;
                k -= yi;
            }

            if k > n as i32 + 3 {
                let tmp = k as f32;
                yy += tmp * tmp;
                yy += tmp * *y_ptr as f32;
                *y_ptr += k;
                *y2f_ptr = 2.0 * *y_ptr as f32;
                k = 0;
            }
        }
    }

    let abs_x_ptr = abs_x.as_ptr();
    let y2f_ptr = y2f.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();
    unsafe {
        while k > 0 {
            yy += 1.0;

            let rxy0 = xy + *abs_x_ptr;
            let mut best_id = 0usize;
            let mut best_num = rxy0 * rxy0;
            let mut best_den = yy + *y2f_ptr;

            let mut i = 1;
            while i < n {
                let rxy = xy + *abs_x_ptr.add(i);
                let ryy = yy + *y2f_ptr.add(i);
                let rxy_sq = rxy * rxy;

                if best_den * rxy_sq > ryy * best_num {
                    best_num = rxy_sq;
                    best_den = ryy;
                    best_id = i;
                }
                i += 1;
            }

            xy += *abs_x_ptr.add(best_id);
            yy += *y2f_ptr.add(best_id);
            *y2f_ptr.add(best_id) += 2.0;
            *y_ptr.add(best_id) += 1;
            k -= 1;
        }
    }

    unsafe {
        let y_ptr = y.as_mut_ptr();
        let sign_ptr = sign_x.as_ptr();
        for i in 0..n {
            let s = *sign_ptr.add(i);
            *y_ptr.add(i) = (*y_ptr.add(i) ^ -s) + s;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn pvq_search_neon(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    use std::arch::aarch64::*;

    debug_assert!(n <= 16);
    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    y[..n].fill(0);

    if k <= 0 {
        return;
    }

    if k <= 4 {
        let mut abs_x_arr = [0.0f32; 16];
        let mut y2f_arr = [0.0f32; 16];
        let mut sign_x_arr = [0i32; 16];

        unsafe {
            let vzero = vdupq_n_f32(0.0);
            let n4 = n & !3;
            for i in (0..n4).step_by(4) {
                let vx = vld1q_f32(x.as_ptr().add(i));
                let vabs = vabsq_f32(vx);
                vst1q_f32(abs_x_arr.as_mut_ptr().add(i), vabs);
                let vneg = vcltq_f32(vx, vzero);
                let vsign = vandq_u32(vneg, vdupq_n_u32(1));
                vst1q_s32(sign_x_arr.as_mut_ptr().add(i), vreinterpretq_s32_u32(vsign));
            }
            for i in n4..n {
                let xi = x[i];
                abs_x_arr[i] = xi.abs();
                sign_x_arr[i] = (xi < 0.0) as i32;
            }
        }

        let mut yy_local = 0.0f32;
        let mut xy_local = 0.0f32;

        let abs_x_ptr = abs_x_arr.as_ptr();
        let y2f_ptr = y2f_arr.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();
        unsafe {
            for _ in 0..k {
                yy_local += 1.0;
                let mut best_num = xy_local + *abs_x_ptr;
                let mut best_den = yy_local + *y2f_ptr;
                let mut best_id = 0;

                let mut j = 1;
                while j < n {
                    let rxy = xy_local + *abs_x_ptr.add(j);
                    let ryy = yy_local + *y2f_ptr.add(j);
                    if best_den * rxy > best_num * ryy {
                        best_den = ryy;
                        best_num = rxy;
                        best_id = j;
                    }
                    j += 1;
                }

                xy_local += *abs_x_ptr.add(best_id);
                yy_local += *y2f_ptr.add(best_id);
                *y2f_ptr.add(best_id) += 2.0;
                *y_ptr.add(best_id) += 1;
            }
        }

        unsafe {
            let sign_ptr = sign_x_arr.as_ptr();
            for i in 0..n {
                let s = *sign_ptr.add(i);
                *y_ptr.add(i) = (*y_ptr.add(i) ^ -s) + s;
            }
        }
        return;
    }

    let mut abs_x_mu = [MaybeUninit::<f32>::uninit(); 20];
    let mut y2f_mu = [MaybeUninit::<f32>::uninit(); 20];
    let mut sign_x_mu = [MaybeUninit::<i32>::uninit(); 16];
    let mut sum;

    let n4 = n & !3;
    unsafe {
        let mut vsum = vdupq_n_f32(0.0);
        let vzero = vdupq_n_f32(0.0);
        for i in (0..n4).step_by(4) {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let vabs = vabsq_f32(vx);
            vst1q_f32(abs_x_mu.as_mut_ptr().add(i) as *mut f32, vabs);
            vsum = vaddq_f32(vsum, vabs);
            let vneg = vcltq_f32(vx, vzero);
            let vsign = vandq_u32(vneg, vdupq_n_u32(1));
            vst1q_s32(
                sign_x_mu.as_mut_ptr().add(i) as *mut i32,
                vreinterpretq_s32_u32(vsign),
            );
        }
        sum = vaddvq_f32(vsum);
    }

    for i in n4..n {
        let xi = x[i];
        let abs_xi = xi.abs();
        abs_x_mu[i].write(abs_xi);
        sum += abs_xi;
        sign_x_mu[i].write((xi < 0.0) as i32);
    }

    let abs_x = unsafe { std::slice::from_raw_parts_mut(abs_x_mu.as_mut_ptr() as *mut f32, 20) };
    let y2f = unsafe { std::slice::from_raw_parts_mut(y2f_mu.as_mut_ptr() as *mut f32, 20) };
    let sign_x = unsafe { std::slice::from_raw_parts_mut(sign_x_mu.as_mut_ptr() as *mut i32, 16) };

    let ran_presearch = k > (n >> 1) as i32 && sum > 1e-15;
    if ran_presearch {
        let rcp = (k as f32 + 0.8) / sum;

        unsafe {
            let vrcp = vdupq_n_f32(rcp);
            let mut vyy = vdupq_n_f32(0.0);
            let mut vxy = vdupq_n_f32(0.0);
            let mut vk_sum = vdupq_n_s32(0);

            for i in (0..n4).step_by(4) {
                let vabs = vld1q_f32(abs_x.as_ptr().add(i));
                let vyi_f = vmulq_f32(vabs, vrcp);
                let vyi = vcvtq_s32_f32(vyi_f);
                vst1q_s32(y.as_mut_ptr().add(i), vyi);

                let vyi_f = vcvtq_f32_s32(vyi);
                vyy = vfmaq_f32(vyy, vyi_f, vyi_f);
                vxy = vfmaq_f32(vxy, vyi_f, vabs);

                let vy2f = vaddq_f32(vyi_f, vyi_f);
                vst1q_f32(y2f.as_mut_ptr().add(i), vy2f);

                vk_sum = vaddq_s32(vk_sum, vyi);
            }

            yy = vaddvq_f32(vyy);
            xy = vaddvq_f32(vxy);
            k -= vaddvq_s32(vk_sum);
        }

        for i in n4..n {
            let yi = (abs_x[i] * rcp) as i32;
            y[i] = yi;
            let yf = yi as f32;
            yy += yf * yf;
            xy += yf * abs_x[i];
            y2f[i] = 2.0 * yf;
            k -= yi;
        }

        if k > n as i32 + 3 {
            let tmp = k as f32;
            yy += tmp * tmp + tmp * y[0] as f32;
            y[0] += k;
            y2f[0] = 2.0 * y[0] as f32;
            k = 0;
        }
    } else {
        for i in 0..n {
            y2f[i] = 0.0;
        }
    }

    unsafe {
        let abs_x_ptr = abs_x.as_ptr();
        let y2f_ptr = y2f.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();
        let n4 = n & !3;

        for _ in 0..k {
            yy += 1.0;

            let vxy = vdupq_n_f32(xy);
            let vyy = vdupq_n_f32(yy);
            let mut vmax = vdupq_n_f32(0.0);
            let mut best_id: usize = 0;

            let mut j = 0;
            while j < n4 {
                let vx = vld1q_f32(abs_x_ptr.add(j));
                let vy = vld1q_f32(y2f_ptr.add(j));
                let rxy = vaddq_f32(vx, vxy);
                let ryy = vaddq_f32(vy, vyy);
                let inv_sqrt = vrsqrteq_f32(ryy);
                let score = vmulq_f32(rxy, inv_sqrt);
                vmax = vmaxq_f32(vmax, score);
                let sc = std::slice::from_raw_parts(&score as *const float32x4_t as *const f32, 4);
                let mx = vmaxvq_f32(vmax);
                for lane in 0..4 {
                    if sc[lane] == mx {
                        best_id = j + lane;
                    }
                }
                j += 4;
            }

            while j < n {
                let rxy = xy + *abs_x_ptr.add(j);
                let ryy = yy + *y2f_ptr.add(j);
                let score = rxy * (1.0 / ryy.sqrt());
                let current_max = vmaxvq_f32(vmax);
                if score > current_max {
                    best_id = j;
                    vmax = vsetq_lane_f32(score, vmax, 0);
                }
                j += 1;
            }

            xy += *abs_x_ptr.add(best_id);
            yy += *y2f_ptr.add(best_id);
            *y2f_ptr.add(best_id) += 2.0;
            *y_ptr.add(best_id) += 1;
        }
    }

    unsafe {
        let y_ptr = y.as_mut_ptr();
        let sign_ptr = sign_x.as_ptr();
        for i in 0..n {
            let s = *sign_ptr.add(i);
            *y_ptr.add(i) = (*y_ptr.add(i) ^ -s) + s;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pvq_search_avx2(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    use std::arch::x86_64::*;

    debug_assert!(n <= 31);
    debug_assert!(k > 4);

    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;

    y[..n].fill(0);

    let mut abs_x = [0.0f32; 32];
    let mut y2f = [0.0f32; 32];
    let mut sign_x = [0i32; 32];

    let sign_mask = _mm256_set1_ps(-0.0f32);
    let vzero_ps = _mm256_setzero_ps();
    let vone_i = _mm256_set1_epi32(1);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let a = _mm256_andnot_ps(sign_mask, v);
        _mm256_storeu_ps(abs_x.as_mut_ptr().add(i), a);
        acc = _mm256_add_ps(acc, a);

        let neg_mask = _mm256_cmp_ps(v, vzero_ps, _CMP_LT_OS);
        let sign_i = _mm256_and_si256(_mm256_castps_si256(neg_mask), vone_i);
        _mm256_storeu_si256(sign_x.as_mut_ptr().add(i) as *mut __m256i, sign_i);
        i += 8;
    }

    let lo4 = _mm256_castps256_ps128(acc);
    let hi4 = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(lo4, hi4);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);
    for j in i..n {
        let xi = x[j];
        let a = xi.abs();
        abs_x[j] = a;
        sum += a;
        sign_x[j] = (xi < 0.0) as i32;
    }

    if k > (n >> 1) as i32 && sum > 1e-15 {
        let rcp = (k as f32 + 0.8) / sum;
        let vrcp = _mm256_set1_ps(rcp);
        let mut vyy_acc = _mm256_setzero_ps();
        let mut vxy_acc = _mm256_setzero_ps();
        let mut vk_acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let vabs = _mm256_loadu_ps(abs_x.as_ptr().add(i));
            let vyi_f32 = _mm256_mul_ps(vabs, vrcp);

            let vyi_i = _mm256_cvttps_epi32(vyi_f32);
            _mm256_storeu_si256(y.as_mut_ptr().add(i) as *mut __m256i, vyi_i);
            let vyi_f = _mm256_cvtepi32_ps(vyi_i);

            vyy_acc = _mm256_fmadd_ps(vyi_f, vyi_f, vyy_acc);
            vxy_acc = _mm256_fmadd_ps(vyi_f, vabs, vxy_acc);
            vk_acc = _mm256_add_ps(vk_acc, vyi_f);

            let vy2f = _mm256_add_ps(vyi_f, vyi_f);
            _mm256_storeu_ps(y2f.as_mut_ptr().add(i), vy2f);
            i += 8;
        }

        let hsum = |v: __m256| -> f32 {
            let lo = _mm256_castps256_ps128(v);
            let hi = _mm256_extractf128_ps(v, 1);
            let s4 = _mm_add_ps(lo, hi);
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            _mm_cvtss_f32(s1)
        };
        yy += hsum(vyy_acc);
        xy += hsum(vxy_acc);
        k -= hsum(vk_acc) as i32;

        while i < n {
            let yi = (abs_x[i] * rcp) as i32;
            y[i] = yi;
            let yf = yi as f32;
            yy += yf * yf;
            xy += yf * abs_x[i];
            y2f[i] = 2.0 * yf;
            k -= yi;
            i += 1;
        }
        if k > n as i32 + 3 {
            let tmp = k as f32;
            yy += tmp * tmp + tmp * y[0] as f32;
            y[0] += k;
            y2f[0] = 2.0 * y[0] as f32;
            k = 0;
        }
    }

    let abs_x_ptr = abs_x.as_ptr();
    let y2f_ptr = y2f.as_mut_ptr();
    let y_ptr = y.as_mut_ptr();
    let n8 = n & !7;
    let n_ceil8 = (n + 7) & !7;
    let mut scores = [0.0f32; 32];

    while k > 0 {
        yy += 1.0;
        let vxy = _mm256_set1_ps(xy);
        let vyy = _mm256_set1_ps(yy);

        let mut vmax = _mm256_setzero_ps();
        let mut j = 0;
        while j < n8 {
            let vabs = _mm256_loadu_ps(abs_x_ptr.add(j));
            let vy2f = _mm256_loadu_ps(y2f_ptr.add(j));
            let rxy = _mm256_add_ps(vabs, vxy);
            let ryy = _mm256_add_ps(vy2f, vyy);
            let score = _mm256_mul_ps(rxy, _mm256_rsqrt_ps(ryy));
            _mm256_storeu_ps(scores.as_mut_ptr().add(j), score);
            vmax = _mm256_max_ps(vmax, score);
            j += 8;
        }

        while j < n {
            let rxy = xy + *abs_x_ptr.add(j);
            let ryy = yy + *y2f_ptr.add(j);
            scores[j] = rxy * (1.0 / ryy.sqrt());
            j += 1;
        }

        let global_max = {
            let hi = _mm256_extractf128_ps(vmax, 1);
            let lo = _mm256_castps256_ps128(vmax);
            let m4 = _mm_max_ps(lo, hi);
            let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            let m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
            _mm_cvtss_f32(m1)
        };

        let mut gmax = global_max;
        for j in n8..n {
            if scores[j] > gmax {
                gmax = scores[j];
            }
        }

        let vgmax = _mm256_set1_ps(gmax);
        let mut best_id: usize = 0;
        let mut j = 0;
        while j < n_ceil8 {
            let vs = _mm256_loadu_ps(scores.as_ptr().add(j));
            let mask = _mm256_movemask_ps(_mm256_cmp_ps(vs, vgmax, _CMP_EQ_OQ)) as u32;
            if mask != 0 {
                best_id = j + mask.trailing_zeros() as usize;
                break;
            }
            j += 8;
        }

        xy += *abs_x_ptr.add(best_id);
        yy += *y2f_ptr.add(best_id);
        *y2f_ptr.add(best_id) += 2.0;
        *y_ptr.add(best_id) += 1;
        k -= 1;
    }

    for i in 0..n {
        let s = sign_x[i];
        y[i] = (y[i] ^ -s) + s;
    }
}

#[inline]
fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        exp_rotation1_neon(x, len, stride, c, s);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        exp_rotation1_scalar(x, len, stride, c, s);
    }
}

#[inline]
fn exp_rotation1_scalar(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    let ms = -s;
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn exp_rotation1_neon(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    if stride == 1 {
        exp_rotation1_scalar(x, len, stride, c, s);
        return;
    }

    use std::arch::aarch64::*;

    let vc = vdupq_n_f32(c);
    let vs = vdupq_n_f32(s);

    // Forward pass: SIMD when we can load 4 contiguous elements
    let mut i = 0;
    while i + 4 <= len - stride {
        let vx1 = vld1q_f32(x.as_ptr().add(i));
        let vx2 = vld1q_f32(x.as_ptr().add(i + stride));

        // y1 = c*x1 - s*x2
        let vy1 = vfmsq_f32(vmulq_f32(vx1, vc), vs, vx2);
        // y2 = c*x2 + s*x1
        let vy2 = vfmaq_f32(vmulq_f32(vx2, vc), vs, vx1);

        vst1q_f32(x.as_mut_ptr().add(i), vy1);
        vst1q_f32(x.as_mut_ptr().add(i + stride), vy2);

        i += 4;
    }
    for j in i..(len - stride) {
        let x1 = x[j];
        let x2 = x[j + stride];
        x[j + stride] = c * x2 + s * x1;
        x[j] = c * x1 - s * x2;
    }

    if len >= 2 * stride {
        for j in (0..(len - 2 * stride)).rev() {
            let x1 = x[j];
            let x2 = x[j + stride];
            x[j + stride] = c * x2 + s * x1;
            x[j] = c * x1 - s * x2;
        }
    }
}

#[inline(always)]
pub fn exp_rotation(x: &mut [f32], length: usize, dir: i32, stride: usize, k: i32, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    if 2 * k >= length as i32 || spread <= 0 || spread > 3 {
        return;
    }
    let factor = SPREAD_FACTOR[spread as usize - 1];
    let gain = (length as f32) / (length as f32 + factor as f32 * k as f32);
    let theta = 0.5 * gain * gain;
    let c = (0.5 * std::f32::consts::PI * theta).cos();
    let s = (0.5 * std::f32::consts::PI * theta).sin();

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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn extract_collapse_mask_neon(iy: &[i32], n: usize, b: usize) -> u32 {
    use std::arch::aarch64::*;

    if b <= 1 {
        return 1;
    }
    let n0 = n / b;
    let mut collapse_mask = 0u32;

    for i in 0..b {
        let base = i * n0;
        let slice = &iy[base..base + n0];

        let mut any_nonzero = false;
        let n4 = n0 & !3;
        let mut j = 0;

        while j < n4 {
            let v = vld1q_s32(slice.as_ptr().add(j));

            let or_val = vorrq_s32(v, vextq_s32(v, v, 2));
            let or_val = vorrq_s32(or_val, vextq_s32(or_val, or_val, 1));
            if vgetq_lane_s32(or_val, 0) != 0 {
                any_nonzero = true;
                break;
            }
            j += 4;
        }

        if !any_nonzero {
            for j in j..n0 {
                if slice[j] != 0 {
                    any_nonzero = true;
                    break;
                }
            }
        }

        if any_nonzero {
            collapse_mask |= 1 << i;
        }
    }
    collapse_mask
}

#[inline(always)]
pub fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        extract_collapse_mask_neon(iy, n, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let n0 = n / b;
        let mut collapse_mask = 0u32;
        for i in 0..b {
            let mut tmp = 0i32;
            let base = i * n0;
            for j in 0..n0 {
                tmp |= iy[base + j];
            }
            if tmp != 0 {
                collapse_mask |= 1 << i;
            }
        }
        collapse_mask
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn renormalise_vector_avx2(x: &mut [f32], n: usize, gain: f32) {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        i += 8;
    }
    let acc = _mm256_add_ps(acc0, acc1);

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut e = 1e-15f32 + _mm_cvtss_f32(s1);
    for j in i..n {
        e += x[j] * x[j];
    }

    let g = gain * (1.0 / e.sqrt());
    let vnorm = _mm256_set1_ps(g);
    i = 0;
    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v0, vnorm));
        _mm256_storeu_ps(x.as_mut_ptr().add(i + 8), _mm256_mul_ps(v1, vnorm));
        i += 16;
    }
    while i + 8 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v0, vnorm));
        i += 8;
    }
    for j in i..n {
        x[j] *= g;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn alg_quant_resynth_avx2(y: &[i32], x: &mut [f32], n: usize, gain: f32) {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let yi = _mm256_loadu_si256(y.as_ptr().add(i) as *const __m256i);
        let yf = _mm256_cvtepi32_ps(yi);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), yf);
        acc0 = _mm256_fmadd_ps(yf, yf, acc0);
        i += 8;
    }

    let lo = _mm256_castps256_ps128(acc0);
    let hi = _mm256_extractf128_ps(acc0, 1);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut ryy = _mm_cvtss_f32(s1);

    for j in i..n {
        let v = y[j] as f32;
        x[j] = v;
        ryy += v * v;
    }

    let g = gain / (1e-15f32 + ryy).sqrt();
    let vg = _mm256_set1_ps(g);

    i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v, vg));
        i += 8;
    }
    for j in i..n {
        x[j] *= g;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pvq_search_scalar_init_avx2(
    x: &[f32],
    n: usize,
    abs_x: &mut [f32; 32],
    sign_x: &mut [i32; 32],
) -> f32 {
    use std::arch::x86_64::*;
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let a = _mm256_andnot_ps(sign_mask, v);
        _mm256_storeu_ps(abs_x.as_mut_ptr().add(i), a);
        acc = _mm256_add_ps(acc, a);

        for j in 0..8 {
            sign_x[i + j] = (x[i + j] < 0.0) as i32;
        }
        i += 8;
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);

    for j in i..n {
        let abs_xi = x[j].abs();
        abs_x[j] = abs_xi;
        sum += abs_xi;
        sign_x[j] = (x[j] < 0.0) as i32;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn renormalise_vector_neon(x: &mut [f32], n: usize, gain: f32) {
    use std::arch::aarch64::*;

    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 16 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let x2 = vld1q_f32(x.as_ptr().add(i + 8));
        let x3 = vld1q_f32(x.as_ptr().add(i + 12));
        sum_vec = vfmaq_f32(sum_vec, x0, x0);
        sum_vec = vfmaq_f32(sum_vec, x1, x1);
        sum_vec = vfmaq_f32(sum_vec, x2, x2);
        sum_vec = vfmaq_f32(sum_vec, x3, x3);
        i += 16;
    }

    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        sum_vec = vfmaq_f32(sum_vec, x0, x0);
        sum_vec = vfmaq_f32(sum_vec, x1, x1);
        i += 8;
    }

    while i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        sum_vec = vfmaq_f32(sum_vec, x0, x0);
        i += 4;
    }

    let mut e = 1e-15f32 + vaddvq_f32(sum_vec);
    for j in i..n {
        e += x[j] * x[j];
    }

    let g = gain * (1.0 / e.sqrt());
    let vg = vdupq_n_f32(g);

    i = 0;
    while i + 16 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let x2 = vld1q_f32(x.as_ptr().add(i + 8));
        let x3 = vld1q_f32(x.as_ptr().add(i + 12));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(x0, vg));
        vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(x1, vg));
        vst1q_f32(x.as_mut_ptr().add(i + 8), vmulq_f32(x2, vg));
        vst1q_f32(x.as_mut_ptr().add(i + 12), vmulq_f32(x3, vg));
        i += 16;
    }

    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(x0, vg));
        vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(x1, vg));
        i += 8;
    }

    while i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(x0, vg));
        i += 4;
    }

    for j in i..n {
        x[j] *= g;
    }
}

pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        renormalise_vector_neon(x, n, gain);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if n >= 8 && std::arch::is_x86_feature_detected!("avx2") {
            renormalise_vector_avx2(x, n, gain);
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut e = 1e-15f32;
        for i in 0..n {
            e += x[i] * x[i];
        }
        let g = gain * (1.0 / e.sqrt());
        for i in 0..n {
            x[i] *= g;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn alg_quant_resynth_neon(y: &[i32], x: &mut [f32], n: usize, gain: f32) {
    use std::arch::aarch64::*;

    let mut sum_vec = vdupq_n_f32(0.0);
    let n8 = n & !7;
    let mut i = 0;

    while i < n8 {
        let yi0 = vld1q_s32(y.as_ptr().add(i));
        let yi1 = vld1q_s32(y.as_ptr().add(i + 4));

        let yf0 = vcvtq_f32_s32(yi0);
        let yf1 = vcvtq_f32_s32(yi1);

        vst1q_f32(x.as_mut_ptr().add(i), yf0);
        vst1q_f32(x.as_mut_ptr().add(i + 4), yf1);

        sum_vec = vfmaq_f32(sum_vec, yf0, yf0);
        sum_vec = vfmaq_f32(sum_vec, yf1, yf1);

        i += 8;
    }

    let mut ryy = vaddvq_f32(sum_vec);
    for j in i..n {
        let v = y[j] as f32;
        x[j] = v;
        ryy += v * v;
    }

    let g = gain / (1e-15 + ryy).sqrt();
    let vg = vdupq_n_f32(g);

    i = 0;
    while i < n8 {
        let vx0 = vld1q_f32(x.as_ptr().add(i));
        let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
        let vr0 = vmulq_f32(vx0, vg);
        let vr1 = vmulq_f32(vx1, vg);
        vst1q_f32(x.as_mut_ptr().add(i), vr0);
        vst1q_f32(x.as_mut_ptr().add(i + 4), vr1);
        i += 8;
    }

    for j in i..n {
        x[j] *= g;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn alg_quant_resynth_scalar(y: &[i32], x: &mut [f32], n: usize, gain: f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if std::arch::is_x86_feature_detected!("avx2") {
            alg_quant_resynth_avx2(y, x, n, gain);
            return;
        }
    }
    let mut ryy = 0.0f32;
    for i in 0..n {
        let v = y[i] as f32;
        x[i] = v;
        ryy += v * v;
    }
    let g = gain / (1e-15 + ryy).sqrt();
    for i in 0..n {
        x[i] *= g;
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
    if n <= 32 {
        let mut y_buf = [MaybeUninit::<i32>::uninit(); 32];
        let y = unsafe { std::slice::from_raw_parts_mut(y_buf.as_mut_ptr() as *mut i32, n) };

        exp_rotation(x, n, 1, stride, k, spread);
        pvq_search(x, y, k, n);
        let mask = extract_collapse_mask(y, n, stride);

        encode_pulses(y, n as u32, k as u32, rc);

        if resynth {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                alg_quant_resynth_neon(y, x, n, gain);
            }
            #[cfg(not(target_arch = "aarch64"))]
            alg_quant_resynth_scalar(y, x, n, gain);
            exp_rotation(x, n, -1, stride, k, spread);
        }
        mask
    } else {
        let mut y_mu = [MaybeUninit::<i32>::uninit(); MAX_PVQ_N];
        let y = unsafe { std::slice::from_raw_parts_mut(y_mu.as_mut_ptr() as *mut i32, MAX_PVQ_N) };

        exp_rotation(x, n, 1, stride, k, spread);
        pvq_search(x, &mut y[..n], k, n);
        let mask = extract_collapse_mask(&y[..n], n, stride);
        encode_pulses(&y[..n], n as u32, k as u32, rc);

        if resynth {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                alg_quant_resynth_neon(y, x, n, gain);
            }
            #[cfg(not(target_arch = "aarch64"))]
            alg_quant_resynth_scalar(y, x, n, gain);
            exp_rotation(x, n, -1, stride, k, spread);
        }
        mask
    }
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
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    alg_quant_resynth_neon(y, x, n, gain);
                }
                #[cfg(not(target_arch = "aarch64"))]
                alg_quant_resynth_scalar(y, x, n, gain);
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
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    alg_quant_resynth_neon(&up_y, x, n, gain);
                }
                #[cfg(not(target_arch = "aarch64"))]
                alg_quant_resynth_scalar(&up_y, x, n, gain);
                exp_rotation(x, n, -1, stride, k, spread);
            }
            return mask;
        }

        pvq_search(x, y, k, n);
        let mask = extract_collapse_mask(y, n, stride);
        encode_pulses(y, n as u32, k as u32, rc);

        if resynth {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                alg_quant_resynth_neon(y, x, n, gain);
            }
            #[cfg(not(target_arch = "aarch64"))]
            alg_quant_resynth_scalar(y, x, n, gain);
            exp_rotation(x, n, -1, stride, k, spread);
        }
        mask
    } else {
        let mut y_mu = [MaybeUninit::<i32>::uninit(); MAX_PVQ_N];
        let y = unsafe { std::slice::from_raw_parts_mut(y_mu.as_mut_ptr() as *mut i32, MAX_PVQ_N) };

        exp_rotation(x, n, 1, stride, k, spread);
        pvq_search(x, &mut y[..n], k, n);
        let mask = extract_collapse_mask(&y[..n], n, stride);
        encode_pulses(&y[..n], n as u32, k as u32, rc);

        if resynth {
            let mut ryy = 0.0f32;
            for i in 0..n {
                let v = y[i] as f32;
                x[i] = v;
                ryy += v * v;
            }
            let g = gain / (1e-15 + ryy).sqrt();
            for i in 0..n {
                x[i] *= g;
            }
            exp_rotation(x, n, -1, stride, k, spread);
        }
        mask
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
    let mut y_mu = [MaybeUninit::<i32>::uninit(); MAX_PVQ_N];
    let y = unsafe { std::slice::from_raw_parts_mut(y_mu.as_mut_ptr() as *mut i32, MAX_PVQ_N) };
    decode_pulses(&mut y[..n], n as u32, k as u32, rc);

    let mask = extract_collapse_mask(&y[..n], n, stride);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        alg_quant_resynth_neon(&y[..n], x, n, gain);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        alg_quant_resynth_scalar(&y[..n], x, n, gain);
    }

    exp_rotation(x, n, -1, stride, k, spread);

    mask
}
