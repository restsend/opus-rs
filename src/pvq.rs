use crate::range_coder::RangeCoder;

// CELT_PVQ_U_DATA: precomputed U(N,K) table, indexed as DATA[ROW_OFFSETS[min(N,K)] + max(N,K)].
// Equivalent to C's non-CWRS_EXTRA_ROWS CELT_PVQ_U_DATA (1272 elements).
// Row offsets: {0,176,351,525,698,870,1041,1131,1178,1207,1226,1240,1248,1254,1257}
// Row r stores U(r, K) for K = r..max_K; access via DATA[OFFSETS[r] + K].
// CELT_PVQ_U_DATA: precomputed U(N,K) table.
// Access: CELT_PVQ_U(n,k) = DATA[CELT_PVQ_U_ROW[min(n,k)] + max(n,k)].
// Ported from C opus (non-CWRS_EXTRA_ROWS, 1272 elements).
// Row offsets: {0,176,351,525,698,870,1041,1131,1178,1207,1226,1240,1248,1254,1257}
// Row r stores U(r, K) for K=r..max: DATA[OFFSETS[r]+K] = U(r, K).
pub const CELT_PVQ_U_DATA: [u32; 1272] = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 3, 5, 7, 9, 11, 13, 15,
    17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
    41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
    65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87,
    89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111,
    113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135,
    137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159,
    161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183,
    185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207,
    209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231,
    233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255,
    257, 259, 261, 263, 265, 267, 269, 271, 273, 275, 277, 279,
    281, 283, 285, 287, 289, 291, 293, 295, 297, 299, 301, 303,
    305, 307, 309, 311, 313, 315, 317, 319, 321, 323, 325, 327,
    329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351,
    13, 25, 41, 61, 85, 113, 145, 181, 221, 265, 313, 365,
    421, 481, 545, 613, 685, 761, 841, 925, 1013, 1105, 1201, 1301,
    1405, 1513, 1625, 1741, 1861, 1985, 2113, 2245, 2381, 2521, 2665, 2813,
    2965, 3121, 3281, 3445, 3613, 3785, 3961, 4141, 4325, 4513, 4705, 4901,
    5101, 5305, 5513, 5725, 5941, 6161, 6385, 6613, 6845, 7081, 7321, 7565,
    7813, 8065, 8321, 8581, 8845, 9113, 9385, 9661, 9941, 10225, 10513, 10805,
    11101, 11401, 11705, 12013, 12325, 12641, 12961, 13285, 13613, 13945, 14281, 14621,
    14965, 15313, 15665, 16021, 16381, 16745, 17113, 17485, 17861, 18241, 18625, 19013,
    19405, 19801, 20201, 20605, 21013, 21425, 21841, 22261, 22685, 23113, 23545, 23981,
    24421, 24865, 25313, 25765, 26221, 26681, 27145, 27613, 28085, 28561, 29041, 29525,
    30013, 30505, 31001, 31501, 32005, 32513, 33025, 33541, 34061, 34585, 35113, 35645,
    36181, 36721, 37265, 37813, 38365, 38921, 39481, 40045, 40613, 41185, 41761, 42341,
    42925, 43513, 44105, 44701, 45301, 45905, 46513, 47125, 47741, 48361, 48985, 49613,
    50245, 50881, 51521, 52165, 52813, 53465, 54121, 54781, 55445, 56113, 56785, 57461,
    58141, 58825, 59513, 60205, 60901, 61601, 63, 129, 231, 377, 575, 833,
    1159, 1561, 2047, 2625, 3303, 4089, 4991, 6017, 7175, 8473, 9919, 11521,
    13287, 15225, 17343, 19649, 22151, 24857, 27775, 30913, 34279, 37881, 41727, 45825,
    50183, 54809, 59711, 64897, 70375, 76153, 82239, 88641, 95367, 102425, 109823, 117569,
    125671, 134137, 142975, 152193, 161799, 171801, 182207, 193025, 204263, 215929, 228031, 240577,
    253575, 267033, 280959, 295361, 310247, 325625, 341503, 357889, 374791, 392217, 410175, 428673,
    447719, 467321, 487487, 508225, 529543, 551449, 573951, 597057, 620775, 645113, 670079, 695681,
    721927, 748825, 776383, 804609, 833511, 863097, 893375, 924353, 956039, 988441, 1021567, 1055425,
    1090023, 1125369, 1161471, 1198337, 1235975, 1274393, 1313599, 1353601, 1394407, 1436025, 1478463, 1521729,
    1565831, 1610777, 1656575, 1703233, 1750759, 1799161, 1848447, 1898625, 1949703, 2001689, 2054591, 2108417,
    2163175, 2218873, 2275519, 2333121, 2391687, 2451225, 2511743, 2573249, 2635751, 2699257, 2763775, 2829313,
    2895879, 2963481, 3032127, 3101825, 3172583, 3244409, 3317311, 3391297, 3466375, 3542553, 3619839, 3698241,
    3777767, 3858425, 3940223, 4023169, 4107271, 4192537, 4278975, 4366593, 4455399, 4545401, 4636607, 4729025,
    4822663, 4917529, 5013631, 5110977, 5209575, 5309433, 5410559, 5512961, 5616647, 5721625, 5827903, 5935489,
    6044391, 6154617, 6266175, 6379073, 6493319, 6608921, 6725887, 6844225, 6963943, 7085049, 7207551, 321,
    681, 1289, 2241, 3649, 5641, 8361, 11969, 16641, 22569, 29961, 39041, 50049,
    63241, 78889, 97281, 118721, 143529, 172041, 204609, 241601, 283401, 330409, 383041, 441729,
    506921, 579081, 658689, 746241, 842249, 947241, 1061761, 1186369, 1321641, 1468169, 1626561, 1797441,
    1981449, 2179241, 2391489, 2618881, 2862121, 3121929, 3399041, 3694209, 4008201, 4341801, 4695809, 5071041,
    5468329, 5888521, 6332481, 6801089, 7295241, 7815849, 8363841, 8940161, 9545769, 10181641, 10848769, 11548161,
    12280841, 13047849, 13850241, 14689089, 15565481, 16480521, 17435329, 18431041, 19468809, 20549801, 21675201, 22846209,
    24064041, 25329929, 26645121, 28010881, 29428489, 30899241, 32424449, 34005441, 35643561, 37340169, 39096641, 40914369,
    42794761, 44739241, 46749249, 48826241, 50971689, 53187081, 55473921, 57833729, 60268041, 62778409, 65366401, 68033601,
    70781609, 73612041, 76526529, 79526721, 82614281, 85790889, 89058241, 92418049, 95872041, 99421961, 103069569, 106816641,
    110664969, 114616361, 118672641, 122835649, 127107241, 131489289, 135983681, 140592321, 145317129, 150160041, 155123009, 160208001,
    165417001, 170752009, 176215041, 181808129, 187533321, 193392681, 199388289, 205522241, 211796649, 218213641, 224775361, 231483969,
    238341641, 245350569, 252512961, 259831041, 267307049, 274943241, 282741889, 290705281, 298835721, 307135529, 315607041, 324252609,
    333074601, 342075401, 351257409, 360623041, 370174729, 379914921, 389846081, 399970689, 410291241, 420810249, 431530241, 442453761,
    453583369, 464921641, 476471169, 488234561, 500214441, 512413449, 524834241, 537479489, 550351881, 563454121, 576788929, 590359041,
    604167209, 618216201, 632508801, 1683, 3653, 7183, 13073, 22363, 36365, 56695, 85305, 124515,
    177045, 246047, 335137, 448427, 590557, 766727, 982729, 1244979, 1560549, 1937199, 2383409, 2908411,
    3522221, 4235671, 5060441, 6009091, 7095093, 8332863, 9737793, 11326283, 13115773, 15124775, 17372905, 19880915,
    22670725, 25765455, 29189457, 32968347, 37129037, 41699767, 46710137, 52191139, 58175189, 64696159, 71789409, 79491819,
    87841821, 96879431, 106646281, 117185651, 128542501, 140763503, 153897073, 167993403, 183104493, 199284183, 216588185, 235074115,
    254801525, 275831935, 298228865, 322057867, 347386557, 374284647, 402823977, 433078547, 465124549, 499040399, 534906769, 572806619,
    612825229, 655050231, 699571641, 746481891, 795875861, 847850911, 902506913, 959946283, 1020274013, 1083597703, 1150027593, 1219676595,
    1292660325, 1369097135, 1449108145, 1532817275, 1620351277, 1711839767, 1807415257, 1907213187, 2011371957, 2120032959, 8989, 19825,
    40081, 75517, 134245, 227305, 369305, 579125, 880685, 1303777, 1884961, 2668525, 3707509, 5064793,
    6814249, 9041957, 11847485, 15345233, 19665841, 24957661, 31388293, 39146185, 48442297, 59511829, 72616013, 88043969,
    106114625, 127178701, 151620757, 179861305, 212358985, 249612805, 292164445, 340600625, 395555537, 457713341, 527810725, 606639529,
    695049433, 793950709, 904317037, 1027188385, 1163673953, 1314955181, 1482288821, 1667010073, 1870535785, 2094367717, 48639, 108545,
    224143, 433905, 795455, 1392065, 2340495, 3800305, 5984767, 9173505, 13726991, 20103025, 28875327, 40754369,
    56610575, 77500017, 104692735, 139703809, 184327311, 240673265, 311207743, 398796225, 506750351, 638878193, 799538175, 993696769,
    1226990095, 1505789553, 1837271615, 2229491905, 265729, 598417, 1256465, 2485825, 4673345, 8405905, 14546705, 24331777,
    39490049, 62390545, 96220561, 145198913, 214828609, 312193553, 446304145, 628496897, 872893441, 1196924561, 1621925137, 2173806145,
    1462563, 3317445, 7059735, 14218905, 27298155, 50250765, 89129247, 152951073, 254831667, 413442773, 654862247, 1014889769,
    1541911931, 2300409629, 3375210671, 8097453, 18474633, 39753273, 81270333, 158819253, 298199265, 540279585, 948062325, 1616336765,
    45046719, 103274625, 224298231, 464387817, 921406335, 1759885185, 3248227095, 251595969, 579168825, 1267854873, 2653649025, 1409933619,
];

// Row offsets into CELT_PVQ_U_DATA. Row r starts at CELT_PVQ_U_ROW[r].
// CELT_PVQ_U(n, k) = CELT_PVQ_U_DATA[CELT_PVQ_U_ROW[min(n,k)] + max(n,k)]
const CELT_PVQ_U_ROW: [u32; 15] = [0, 176, 351, 525, 698, 870, 1041, 1131, 1178, 1207, 1226, 1240, 1248, 1254, 1257];

/// O(1) table lookup for U(n,k) = U(k,n).
/// Fast path: valid for all (n,k) where min(n,k) <= 14 (covers all standard CELT use).
/// Fallback (rare/non-CELT): dynamic O(n*k) computation via ncwrs.
#[inline(always)]
pub fn celt_pvq_u_lookup(n: u32, k: u32) -> u32 {
    let r = n.min(k) as usize;
    let c = n.max(k) as usize;
    if r < CELT_PVQ_U_ROW.len() {
        let row_base = CELT_PVQ_U_ROW[r] as usize;
        // Also bounds-check column within the data array
        if row_base + c < CELT_PVQ_U_DATA.len() {
            return CELT_PVQ_U_DATA[row_base + c];
        }
    }
    // Fallback for out-of-table (n,k) pairs (not used in standard CELT)
    ncwrs(n, k)
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

/// V(n, k) = U(n, k) + U(n, k+1): total PVQ codewords for band of size n with k pulses.
#[inline(always)]
pub fn celt_pvq_u(n: u32, k: u32) -> u32 {
    celt_pvq_u_lookup(n, k)
}

/// V(n, k) = U(n, k) + U(n, k+1).
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

/// Encode a PVQ pulse vector y[0..n] into a codeword index.
/// O(n) algorithm using precomputed U(n,k) table lookup.
/// Ported from C opus non-SMALL_FOOTPRINT icwrs().
pub fn icwrs(n: u32, _k: u32, y: &[i32]) -> u32 {
    if n == 1 {
        // Special case: single dimension, codeword = sign bit
        return if y[0] < 0 { 1 } else { 0 };
    }
    debug_assert!(n >= 2, "icwrs: n must be >= 2");
    let mut j = (n - 1) as usize;
    // Start with sign bit of last element
    let mut i: u32 = if y[j] < 0 { 1 } else { 0 };
    let mut k = y[j].unsigned_abs() as u32;
    // Process remaining elements (j = n-2 down to 0)
    // n - j goes from 2 to n; use table: U(n-j, k)
    while j > 0 {
        j -= 1;
        let m = (n - j as u32) as u32; // m = n - j, ranges 2..n
        i = i.wrapping_add(celt_pvq_u_lookup(m, k));
        k += y[j].unsigned_abs() as u32;
        if y[j] < 0 {
            i = i.wrapping_add(celt_pvq_u_lookup(m, k + 1));
        }
    }
    i
}

/// Decode a PVQ codeword index i into pulse vector y[0..n].
/// O(n) algorithm using precomputed U(n,k) table lookup.
/// Ported from C opus non-SMALL_FOOTPRINT cwrsi().
pub fn cwrsi(n: u32, k: u32, mut i: u32, y: &mut [i32]) {
    debug_assert!(k > 0, "cwrsi: k must be > 0");

    if n == 1 {
        let s = -(i as i32);
        y[0] = ((k as i32) + s) ^ s;
        return;
    }

    let mut curr_n = n;
    let mut curr_k = k;
    let mut j = 0usize;

    // Main loop: process dimensions n down to 3
    while curr_n > 2 {
        if curr_k >= curr_n {
            // "Lots of pulses" case: curr_k >= curr_n.
            // Row index = curr_n, which is guaranteed <= 14 here.
            let p_kp1 = celt_pvq_u_lookup(curr_n, curr_k + 1);
            let s: i32 = if i >= p_kp1 { i -= p_kp1; -1 } else { 0 };
            let k0 = curr_k;
            let q = celt_pvq_u_lookup(curr_n, curr_n);
            let mut p;
            if q > i {
                // Backtrack from curr_n downward
                curr_k = curr_n;
                loop {
                    curr_k -= 1;
                    p = celt_pvq_u_lookup(curr_n, curr_k);
                    if p <= i { break; }
                }
            } else {
                // Backtrack from curr_k downward
                p = celt_pvq_u_lookup(curr_n, curr_k);
                while p > i {
                    curr_k -= 1;
                    p = celt_pvq_u_lookup(curr_n, curr_k);
                }
            }
            i -= p;
            let val = (k0 - curr_k) as i32;
            y[j] = (val + s) ^ s;
        } else {
            // "Lots of dimensions" case: curr_k < curr_n.
            // Row index = curr_k, which is < curr_n, so curr_k <= 14 if orig min(n,k)<=14.
            let p_k  = celt_pvq_u_lookup(curr_k, curr_n);
            let p_kp1 = celt_pvq_u_lookup(curr_k + 1, curr_n);
            if p_k <= i && i < p_kp1 {
                i -= p_k;
                y[j] = 0;
                j += 1;
                curr_n -= 1;
                continue;
            }
            let s: i32 = if i >= p_kp1 { i -= p_kp1; -1 } else { 0 };
            let k0 = curr_k;
            // Backtrack curr_k downward
            let mut p;
            loop {
                curr_k -= 1;
                p = celt_pvq_u_lookup(curr_k, curr_n);
                if p <= i { break; }
            }
            i -= p;
            let val = (k0 - curr_k) as i32;
            y[j] = (val + s) ^ s;
        }
        j += 1;
        curr_n -= 1;
    }

    // curr_n == 2: closed-form
    let p2 = 2 * curr_k + 1;
    let s2: i32 = if i >= p2 { i -= p2; -1 } else { 0 };
    let k0 = curr_k;
    curr_k = ((i + 1) >> 1) as u32;
    if curr_k > 0 { i -= 2 * curr_k - 1; }
    y[j] = ((k0 - curr_k) as i32 + s2) ^ s2;
    j += 1;

    // curr_n == 1: last element
    let s1 = -(i as i32);
    y[j] = (curr_k as i32 + s1) ^ s1;
}

pub fn encode_pulses(y: &[i32], n: u32, k: u32, rc: &mut RangeCoder) {
    if k == 0 {
        return;
    }
    let fl = icwrs(n, k, y);
    let ft = celt_pvq_v(n, k);
    rc.enc_uint(fl, ft);
}

pub fn decode_pulses(y: &mut [i32], n: u32, k: u32, rc: &mut RangeCoder) {
    if k == 0 {
        for i in 0..n as usize {
            y[i] = 0;
        }
        return;
    }
    let ft = celt_pvq_v(n, k);
    let fl = rc.dec_uint(ft);

    cwrsi(n, k, fl, y);
}

pub fn pvq_search(x: &[f32], y: &mut [i32], k: i32, n: usize) {
    let mut k = k;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;
    for i in 0..n {
        y[i] = 0;
    }
    if k <= 0 {
        return;
    }

    // Pre-compute |x[i]| to avoid repeated abs() in hot loop
    let mut abs_x = [0.0f32; MAX_PVQ_N];
    let mut sum = 0.0f32;
    for i in 0..n {
        abs_x[i] = x[i].abs();
        sum += abs_x[i];
    }

    if sum > 1e-15 {
        let scale = k as f32 / sum;
        for i in 0..n {
            y[i] = (abs_x[i] * scale).floor() as i32;
            k -= y[i];
            yy += (y[i] * y[i]) as f32;
            xy += y[i] as f32 * abs_x[i];
        }
    }

    // Greedy search: assign remaining pulses one at a time
    // C opus optimization: cache best_num^2 * best_den to reduce multiplications
    // in the comparison from 4 to 2 per iteration.
    while k > 0 {
        let mut best_id = 0;
        let mut best_num = 0.0f32;
        let mut best_den = 1.0f32;
        // Cache best_num * best_num (= pRxy^2 in C opus terms)
        let mut best_num_sq = 0.0f32;

        for i in 0..n {
            let num = xy + abs_x[i];
            let den = yy + 2.0 * y[i] as f32 + 1.0;
            // Compare: num^2 * best_den > best_num_sq * den
            // This avoids recomputing best_num^2 each iteration
            if num * num * best_den > best_num_sq * den {
                best_num = num;
                best_den = den;
                best_num_sq = num * num;
                best_id = i;
            }
        }
        xy = best_num;
        yy += 2.0 * y[best_id] as f32 + 1.0;
        y[best_id] += 1;
        k -= 1;
    }
    for i in 0..n {
        if x[i] < 0.0 {
            y[i] = -y[i];
        }
    }
}

#[inline]
fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
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

#[inline]
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

#[inline]
pub fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
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

#[inline]
pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let mut e = 1e-15f32;
    for i in 0..n {
        e += x[i] * x[i];
    }
    let g = gain * (1.0 / e.sqrt());
    for i in 0..n {
        x[i] *= g;
    }
}

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
    let mut y = [0i32; MAX_PVQ_N];
    exp_rotation(x, n, 1, stride, k, spread);
    pvq_search(x, &mut y[..n], k, n);
    let mask = extract_collapse_mask(&y[..n], n, stride);

    encode_pulses(&y[..n], n as u32, k as u32, rc);

    if resynth {
        // Fuse int-to-float conversion and norm computation in one pass
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

pub fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: i32,
    spread: i32,
    stride: usize,
    rc: &mut RangeCoder,
    gain: f32,
) -> u32 {
    let mut y = [0i32; MAX_PVQ_N];
    decode_pulses(&mut y[..n], n as u32, k as u32, rc);

    let mask = extract_collapse_mask(&y[..n], n, stride);
    // Fuse int-to-float conversion and norm computation
    let mut ryy = 0.0f32;
    for i in 0..n {
        let v = y[i] as f32;
        x[i * stride] = v;
        ryy += v * v;
    }
    let g = gain / (1e-15 + ryy).sqrt();
    for i in 0..n {
        x[i * stride] *= g;
    }

    exp_rotation(x, n, -1, stride, k, spread);

    mask
}
