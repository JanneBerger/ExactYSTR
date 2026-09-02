/// Exact pedigree-based Y-STR match probability calculator (Rust port).
///
/// Implements the same two-pass Elston-Stewart inside-outside algorithm as
/// the Python reference (ystr_pedigree.py) and runs the identical runtime
/// experiments, outputting results as JSON for the Python plotting script.
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Constants (mirror Python defaults)
// ---------------------------------------------------------------------------

const MINALL: i32 = 3;
const MAXALL: i32 = 30;
const MU: f64 = 0.1;
const REPEATS: usize = 5;
const N_SEEDS: usize = 20;
const BASE_SEED: u64 = 22112000;
const SUSPECT_ALLELE: i32 = 16;

const TOPOLOGY_NAMES: [&str; 3] = [
    "Linear chain (1 son)",
    "Wide (3 sons)",
    "Very wide (5 sons)",
];
const TOPOLOGY_FANS: [usize; 3] = [1, 3, 5];

// ---------------------------------------------------------------------------
// Data structure
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Male {
    male_id:    usize,
    generation: usize,
    father_id:  i64,   // -1 for root
    alleles:    Vec<i32>, // indexed by locus index
}

// ---------------------------------------------------------------------------
// Mutation model
// ---------------------------------------------------------------------------

#[inline(always)]
fn stepwise_prob(j: i32, k: i32, mu: f64) -> f64 {
    match (k - j).abs() {
        0 => 1.0 - mu,
        1 => mu / 2.0,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Core algorithm  —  two-pass Elston-Stewart inside-outside sweep
// ---------------------------------------------------------------------------

fn compute_match_probs_fast(
    males:     &[Male],
    locus_idx: usize,
    mu:        f64,
    minall:    i32,
    maxall:    i32,
) -> f64 {
    let n       = males.len();
    let base: Vec<i32> = males.iter().map(|m| m.alleles[locus_idx]).collect();
    let suspect = base[0];

    if suspect == 0 {
        return 0.0;
    }

    // Build tree (id → index map, sons list) --------------------------------
    let id_to_idx: HashMap<usize, usize> =
        males.iter().enumerate().map(|(i, m)| (m.male_id, i)).collect();

    let mut sons: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, m) in males.iter().enumerate() {
        if m.father_id >= 0 {
            if let Some(&fi) = id_to_idx.get(&(m.father_id as usize)) {
                sons[fi].push(i);
            }
        }
    }

    let a = (maxall - minall + 1) as usize;
    let idx = |x: i32| (x - minall) as usize;

    // Inside pass (beta), bottom-up -----------------------------------------
    let mut beta:    Vec<Vec<f64>> = vec![vec![0.0; a]; n];
    let mut message: Vec<Vec<f64>> = vec![vec![0.0; a]; n];

    for i in (0..n).rev() {
        let e = base[i];
        if sons[i].is_empty() {
            if e == 0 {
                beta[i].fill(1.0);
            } else {
                beta[i][idx(e)] = 1.0;
            }
        } else {
            for k in minall..=maxall {
                if e != 0 && e != k {
                    continue;
                }
                let ki = idx(k);
                let mut val = 1.0_f64;
                for &s in &sons[i] {
                    val *= message[s][ki];
                }
                beta[i][ki] = val;
            }
        }

        for k_f in minall..=maxall {
            let lo = (k_f - 1).max(minall);
            let hi = (k_f + 1).min(maxall);
            let mut acc = 0.0_f64;
            for ll in lo..=hi {
                acc += stepwise_prob(k_f, ll, mu) * beta[i][idx(ll)];
            }
            message[i][idx(k_f)] = acc;
        }
    }

    // Outside pass (alpha), top-down ----------------------------------------
    let mut alpha: Vec<Vec<f64>> = vec![vec![1.0; a]; n];

    for f in 0..n {
        let kids = sons[f].clone();
        if kids.is_empty() {
            continue;
        }
        let m_sons = kids.len();

        // Prefix / suffix products (leave-one-out siblings)
        let mut prefix: Vec<Vec<f64>> = vec![vec![1.0; a]; m_sons + 1];
        let mut suffix: Vec<Vec<f64>> = vec![vec![1.0; a]; m_sons + 1];

        for t in 0..m_sons {
            let s = kids[t];
            for kk in 0..a {
                prefix[t + 1][kk] = prefix[t][kk] * message[s][kk];
            }
        }
        for t in (0..m_sons).rev() {
            let s = kids[t];
            for kk in 0..a {
                suffix[t][kk] = suffix[t + 1][kk] * message[s][kk];
            }
        }

        let ef = base[f];
        let mut father_factor: Vec<f64> = vec![0.0; a];
        for k_f in minall..=maxall {
            let kfi = idx(k_f);
            let e_val = if ef == 0 || ef == k_f { 1.0 } else { 0.0 };
            father_factor[kfi] = alpha[f][kfi] * e_val;
        }

        for (t, &i) in kids.iter().enumerate() {
            for k_i in minall..=maxall {
                let lo = (k_i - 1).max(minall);
                let hi = (k_i + 1).min(maxall);
                let mut acc = 0.0_f64;
                for k_f in lo..=hi {
                    let kfi = idx(k_f);
                    acc += stepwise_prob(k_f, k_i, mu)
                        * father_factor[kfi]
                        * prefix[t][kfi]
                        * suffix[t + 1][kfi];
                }
                alpha[i][idx(k_i)] = acc;
            }
        }
    }

    let si = idx(suspect);
    let p1 = beta[0][si];
    if p1 == 0.0 {
        return 0.0;
    }

    let mut total = 0.0_f64;
    let mut count = 0usize;
    for (i, &allele) in base.iter().enumerate() {
        if allele == 0 {
            let p2 = alpha[i][si] * beta[i][si];
            total += p2 / p1;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total / count as f64 }
}

// ---------------------------------------------------------------------------
// Pedigree generator
// ---------------------------------------------------------------------------

fn generate_pedigree(
    n:         usize,
    n_untyped: usize,
    branching: usize,
    n_loci:    usize,
    seed:      u64,
) -> Vec<Male> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let take = n_untyped.min(n - 1);
    let mut pool: Vec<usize> = (1..n).collect();
    pool.shuffle(&mut rng);
    let untyped: HashSet<usize> = pool[..take].iter().cloned().collect();

    let mut males: Vec<Male> = (0..n)
        .map(|i| {
            let father_id = if i == 0 { -1i64 } else { ((i - 1) / branching) as i64 };
            let mut generation = 0usize;
            let mut j = i;
            while j > 0 {
                j = (j - 1) / branching;
                generation += 1;
            }
            let allele = if untyped.contains(&i) { 0 } else { SUSPECT_ALLELE };
            Male {
                male_id: i,
                generation,
                father_id,
                alleles: vec![allele; n_loci],
            }
        })
        .collect();

    males.sort_by_key(|m| (m.generation, m.male_id));
    males
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn median_ms(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn measure_single(males: &[Male], locus_idx: usize, mu: f64, minall: i32, maxall: i32) -> f64 {
    // warm-up
    let _ = compute_match_probs_fast(males, locus_idx, mu, minall, maxall);
    let mut times = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let _ = compute_match_probs_fast(males, locus_idx, mu, minall, maxall);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    median_ms(times)
}

fn measure_multi_locus(males: &[Male], n_loci: usize, mu: f64, minall: i32, maxall: i32) -> f64 {
    // warm-up
    for li in 0..n_loci {
        let _ = compute_match_probs_fast(males, li, mu, minall, maxall);
    }
    let mut times = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        for li in 0..n_loci {
            let _ = compute_match_probs_fast(males, li, mu, minall, maxall);
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    median_ms(times)
}

fn mean_std(data: &[f64]) -> (f64, f64) {
    let m = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    (m, var.sqrt())
}

// ---------------------------------------------------------------------------
// Experiment runners
// ---------------------------------------------------------------------------

type Means = HashMap<String, Vec<f64>>;
type Stds  = HashMap<String, Vec<f64>>;

fn init_maps() -> (Means, Stds) {
    let means: Means = TOPOLOGY_NAMES.iter().map(|n| (n.to_string(), vec![])).collect();
    let stds:  Stds  = TOPOLOGY_NAMES.iter().map(|n| (n.to_string(), vec![])).collect();
    (means, stds)
}

fn experiment_n(ns: &[i64]) -> (Means, Stds) {
    eprintln!(
        "[1/4] Sweep: pedigree size n  (u=30, A=28, L=1, {} untyped-placements/point)",
        N_SEEDS
    );
    let (mut means, mut stds) = init_maps();
    for &n_i64 in ns {
        let n = n_i64 as usize;
        for (ti, &name) in TOPOLOGY_NAMES.iter().enumerate() {
            let branching = TOPOLOGY_FANS[ti];
            let pts: Vec<f64> = (0..N_SEEDS)
                .map(|k| {
                    let males = generate_pedigree(n, 30, branching, 1, BASE_SEED + k as u64);
                    measure_single(&males, 0, MU, MINALL, MAXALL)
                })
                .collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        let vals: Vec<String> = TOPOLOGY_NAMES
            .iter()
            .zip(["L", "W", "V"])
            .map(|(nm, ab)| {
                let m = means.get(*nm).unwrap().last().unwrap();
                let s = stds.get(*nm).unwrap().last().unwrap();
                format!("{}={:.3}±{:.3}ms", ab, m, s)
            })
            .collect();
        eprintln!("  n={:>5}   {}", n, vals.join("  "));
    }
    (means, stds)
}

fn experiment_u(us: &[i64], n: usize) -> (Means, Stds) {
    eprintln!(
        "[2/4] Sweep: untyped males u  (n={}, A=28, L=1, {} untyped-placements/point)",
        n, N_SEEDS
    );
    let (mut means, mut stds) = init_maps();
    for &u_i64 in us {
        let u = u_i64 as usize;
        for (ti, &name) in TOPOLOGY_NAMES.iter().enumerate() {
            let branching = TOPOLOGY_FANS[ti];
            let pts: Vec<f64> = (0..N_SEEDS)
                .map(|k| {
                    let males = generate_pedigree(n, u, branching, 1, BASE_SEED + k as u64);
                    measure_single(&males, 0, MU, MINALL, MAXALL)
                })
                .collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        let vals: Vec<String> = TOPOLOGY_NAMES
            .iter()
            .zip(["L", "W", "V"])
            .map(|(nm, ab)| {
                let m = means.get(*nm).unwrap().last().unwrap();
                let s = stds.get(*nm).unwrap().last().unwrap();
                format!("{}={:.3}±{:.3}ms", ab, m, s)
            })
            .collect();
        eprintln!("  u={:>5}   {}", u, vals.join("  "));
    }
    (means, stds)
}

fn experiment_a(a_sizes: &[i64], n: usize, u: usize) -> (Means, Stds) {
    eprintln!(
        "[3/4] Sweep: allele range A  (n={}, u={}, L=1, {} untyped-placements/point)",
        n, u, N_SEEDS
    );
    let centre = (MINALL + MAXALL) / 2;
    let (mut means, mut stds) = init_maps();
    for &a_size_i64 in a_sizes {
        let a_size = a_size_i64 as i32;
        let mn = centre - a_size / 2;
        let mx = mn + a_size - 1;
        for (ti, &name) in TOPOLOGY_NAMES.iter().enumerate() {
            let branching = TOPOLOGY_FANS[ti];
            let pts: Vec<f64> = (0..N_SEEDS)
                .map(|k| {
                    let males = generate_pedigree(n, u, branching, 1, BASE_SEED + k as u64);
                    measure_single(&males, 0, MU, mn, mx)
                })
                .collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        let vals: Vec<String> = TOPOLOGY_NAMES
            .iter()
            .zip(["L", "W", "V"])
            .map(|(nm, ab)| {
                let m = means.get(*nm).unwrap().last().unwrap();
                let s = stds.get(*nm).unwrap().last().unwrap();
                format!("{}={:.3}±{:.3}ms", ab, m, s)
            })
            .collect();
        eprintln!("  A={:>5}   {}", a_size, vals.join("  "));
    }
    (means, stds)
}

fn experiment_l(ls: &[i64], n: usize, u: usize) -> (Means, Stds) {
    eprintln!(
        "[4/4] Sweep: number of loci L  (n={}, u={}, A=28, {} untyped-placements/point)",
        n, u, N_SEEDS
    );
    let (mut means, mut stds) = init_maps();
    for &n_loci_i64 in ls {
        let n_loci = n_loci_i64 as usize;
        for (ti, &name) in TOPOLOGY_NAMES.iter().enumerate() {
            let branching = TOPOLOGY_FANS[ti];
            let pts: Vec<f64> = (0..N_SEEDS)
                .map(|k| {
                    let males =
                        generate_pedigree(n, u, branching, n_loci, BASE_SEED + k as u64);
                    measure_multi_locus(&males, n_loci, MU, MINALL, MAXALL)
                })
                .collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        let vals: Vec<String> = TOPOLOGY_NAMES
            .iter()
            .zip(["L", "W", "V"])
            .map(|(nm, ab)| {
                let m = means.get(*nm).unwrap().last().unwrap();
                let s = stds.get(*nm).unwrap().last().unwrap();
                format!("{}={:.3}±{:.3}ms", ab, m, s)
            })
            .collect();
        eprintln!("  L={:>5}   {}", n_loci, vals.join("  "));
    }
    (means, stds)
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Output {
    ns:    Vec<i64>,
    res_n: Means,
    std_n: Stds,
    us:    Vec<i64>,
    res_u: Means,
    std_u: Stds,
    #[serde(rename = "As")]
    a_sizes: Vec<i64>,
    res_A:   Means,
    std_A:   Stds,
    #[serde(rename = "Ls")]
    ls:    Vec<i64>,
    res_L: Means,
    std_L: Stds,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let ns: Vec<i64> = vec![50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500];
    let us: Vec<i64> = vec![1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 199];
    let a_sizes: Vec<i64> = vec![4, 6, 8, 10, 14, 18, 22, 26, 28];
    let ls: Vec<i64> = vec![1, 2, 3, 5, 8, 10, 15, 20];

    let (res_n, std_n) = experiment_n(&ns);
    let (res_u, std_u) = experiment_u(&us, 200);
    let (res_a, std_a) = experiment_a(&a_sizes, 200, 50);
    let (res_l, std_l) = experiment_l(&ls, 50, 25);

    let output = Output {
        ns,
        res_n,
        std_n,
        us,
        res_u,
        std_u,
        a_sizes,
        res_A: res_a,
        std_A: std_a,
        ls,
        res_L: res_l,
        std_L: std_l,
    };

    let json = serde_json::to_string_pretty(&output).expect("JSON serialization failed");
    println!("{}", json);
}
