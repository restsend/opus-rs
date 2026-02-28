pub fn silk_insertion_sort_increasing(a: &mut [i32], idx: &mut [i32], l: usize, k: usize) {
    assert!(k > 0);
    assert!(l > 0);
    assert!(l >= k);

    // Write start indices in index vector
    for i in 0..k {
        idx[i] = i as i32;
    }

    // Sort vector elements by value, increasing order
    for i in 1..k {
        let value = a[i];
        let mut j = i as i32 - 1;
        while j >= 0 && value < a[j as usize] {
            a[(j + 1) as usize] = a[j as usize]; // Shift value
            idx[(j + 1) as usize] = idx[j as usize]; // Shift index
            j -= 1;
        }
        a[(j + 1) as usize] = value; // Write value
        idx[(j + 1) as usize] = i as i32; // Write index
    }

    // If less than L values are asked for, check the remaining values,
    // but only spend CPU to ensure that the K first values are correct
    for i in k..l {
        let value = a[i];
        if value < a[k - 1] {
            let mut j = k as i32 - 2;
            while j >= 0 && value < a[j as usize] {
                a[(j + 1) as usize] = a[j as usize]; // Shift value
                idx[(j + 1) as usize] = idx[j as usize]; // Shift index
                j -= 1;
            }
            a[(j + 1) as usize] = value; // Write value
            idx[(j + 1) as usize] = i as i32; // Write index
        }
    }
}

pub fn silk_insertion_sort_increasing_all_values_int16(a: &mut [i16], l: usize) {
    for i in 1..l {
        let value = a[i];
        let mut j = i as i32 - 1;
        while j >= 0 && value < a[j as usize] {
            a[(j + 1) as usize] = a[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value;
    }
}

pub fn silk_insertion_sort_decreasing_int16(a: &mut [i16], idx: &mut [i32], l: usize, k: usize) {
    assert!(k > 0);
    assert!(l > 0);
    assert!(l >= k);

    // Write start indices in index vector
    for i in 0..k {
        idx[i] = i as i32;
    }

    // Sort vector elements by value, decreasing order
    for i in 1..k {
        let value = a[i];
        let mut j = i as i32 - 1;
        while j >= 0 && value > a[j as usize] {
            a[(j + 1) as usize] = a[j as usize]; // Shift value
            idx[(j + 1) as usize] = idx[j as usize]; // Shift index
            j -= 1;
        }
        a[(j + 1) as usize] = value; // Write value
        idx[(j + 1) as usize] = i as i32; // Write index
    }

    // If less than L values are asked for, check the remaining values,
    // but only spend CPU to ensure that the K first values are correct
    for i in k..l {
        let value = a[i];
        if value > a[k - 1] {
            let mut j = k as i32 - 2;
            while j >= 0 && value > a[j as usize] {
                a[(j + 1) as usize] = a[j as usize]; // Shift value
                idx[(j + 1) as usize] = idx[j as usize]; // Shift index
                j -= 1;
            }
            a[(j + 1) as usize] = value; // Write value
            idx[(j + 1) as usize] = i as i32; // Write index
        }
    }
}
