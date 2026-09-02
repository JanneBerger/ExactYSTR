//! Empirical runtime analysis of the Elston-Stewart algorithm (Rust).
//!
//! Mirrors the four experiments from runtime_analysis.py:
//!   (n) pedigree size          — u=30,  A=28, L=1
//!   (u) number of untyped males — n=200, A=28, L=1
//!   (A) allele range size       — n=200, u=50, L=1
//!   (L) number of loci          — n=50,  u=25, A=28
//!
//! Outputs a JSON object (to stdout) consumed by runtime_analysis_rust.py
//! for matplotlib plotting.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::Serialize;

use ystr_rust::{compute_match_probs_fast, Male, MINALL, MAXALL, DEFAULT_MU};

// ---------------------------------------------------------------------------
// Constants (match Python defaults)
// ---------------------------------------------------------------------------

const MU:        f64   = DEFAULT_MU;
const REPEATS:   usize = 5;
const N_SEEDS:   usize = 20;
const BASE_SEED: u64   = 22112000;
const SUSPECT:   i32   = 16;

const TOPO_NAMES: [&str; 3] = [
    "Linear chain (1 son)",
    "Wide (3 sons)",
    "Very wide (5 sons)",
];
const TOPO_FANS: [usize; 3] = [1, 3, 5];

// ---------------------------------------------------------------------------
// Pedigree generator  (no CSV — synthetic trees for benchmarking)
// ---------------------------------------------------------------------------

fn generate_pedigree(
    n:         usize,
    n_untyped: usize,
    branching: usize,
    loci:      &[String],
    seed:      u64,
) -> Vec<Male> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let take = n_untyped.min(n.saturating_sub(1));
    let mut pool: Vec<usize> = (1..n).collect();
    pool.shuffle(&mut rng);
    let untyped: HashSet<usize> = pool[..take].iter().cloned().collect();

    let mut males: Vec<Male> = (0..n)
        .map(|i| {
            let father_id = if i == 0 { -1i64 } else { ((i - 1) / branching) as i64 };
            let mut gen = 0u32;
            let mut j = i;
            while j > 0 { j = (j - 1) / branching; gen += 1; }
            let allele = if untyped.contains(&i) { 0 } else { SUSPECT };
            let alleles = loci.iter().map(|l| (l.clone(), allele)).collect();
            Male { male_id: i as u32, generation: gen, father_id, alleles }
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

fn measure_single(males: &[Male], locus: &str, mu: f64, mn: i32, mx: i32) -> f64 {
    let _ = compute_match_probs_fast(males, locus, mu, mn, mx);
    let mut times = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let _ = compute_match_probs_fast(males, locus, mu, mn, mx);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    median_ms(times)
}

fn measure_multi(males: &[Male], loci: &[String], mu: f64, mn: i32, mx: i32) -> f64 {
    for loc in loci { let _ = compute_match_probs_fast(males, loc, mu, mn, mx); }
    let mut times = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        for loc in loci { let _ = compute_match_probs_fast(males, loc, mu, mn, mx); }
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

fn new_maps() -> (Means, Stds) {
    let m = TOPO_NAMES.iter().map(|n| (n.to_string(), vec![])).collect();
    let s = TOPO_NAMES.iter().map(|n| (n.to_string(), vec![])).collect();
    (m, s)
}

fn progress_line(label: &str, val: usize, means: &Means, stds: &Stds) {
    let parts: Vec<String> = TOPO_NAMES.iter().zip(["L","W","V"]).map(|(nm, ab)| {
        let m = means.get(*nm).unwrap().last().unwrap();
        let s = stds.get(*nm).unwrap().last().unwrap();
        format!("{ab}={m:.3}±{s:.3}ms")
    }).collect();
    eprintln!("  {label}={val:>5}   {}", parts.join("  "));
}

fn experiment_n(ns: &[i64]) -> (Means, Stds) {
    eprintln!("[1/4] Sweep: pedigree size n  (u=30, A=28, L=1, {N_SEEDS} placements/point)");
    let loci = vec!["L".to_string()];
    let (mut means, mut stds) = new_maps();
    for &n_i64 in ns {
        let n = n_i64 as usize;
        for (ti, &name) in TOPO_NAMES.iter().enumerate() {
            let pts: Vec<f64> = (0..N_SEEDS).map(|k| {
                let males = generate_pedigree(n, 30, TOPO_FANS[ti], &loci, BASE_SEED + k as u64);
                measure_single(&males, "L", MU, MINALL, MAXALL)
            }).collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        progress_line("n", n, &means, &stds);
    }
    (means, stds)
}

fn experiment_u(us: &[i64], n: usize) -> (Means, Stds) {
    eprintln!("[2/4] Sweep: untyped males u  (n={n}, A=28, L=1, {N_SEEDS} placements/point)");
    let loci = vec!["L".to_string()];
    let (mut means, mut stds) = new_maps();
    for &u_i64 in us {
        let u = u_i64 as usize;
        for (ti, &name) in TOPO_NAMES.iter().enumerate() {
            let pts: Vec<f64> = (0..N_SEEDS).map(|k| {
                let males = generate_pedigree(n, u, TOPO_FANS[ti], &loci, BASE_SEED + k as u64);
                measure_single(&males, "L", MU, MINALL, MAXALL)
            }).collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        progress_line("u", u, &means, &stds);
    }
    (means, stds)
}

fn experiment_a(a_sizes: &[i64], n: usize, u: usize) -> (Means, Stds) {
    eprintln!("[3/4] Sweep: allele range A  (n={n}, u={u}, L=1, {N_SEEDS} placements/point)");
    let loci   = vec!["L".to_string()];
    let centre = (MINALL + MAXALL) / 2;
    let (mut means, mut stds) = new_maps();
    for &a_i64 in a_sizes {
        let a  = a_i64 as i32;
        let mn = centre - a / 2;
        let mx = mn + a - 1;
        for (ti, &name) in TOPO_NAMES.iter().enumerate() {
            let pts: Vec<f64> = (0..N_SEEDS).map(|k| {
                let males = generate_pedigree(n, u, TOPO_FANS[ti], &loci, BASE_SEED + k as u64);
                measure_single(&males, "L", MU, mn, mx)
            }).collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        progress_line("A", a as usize, &means, &stds);
    }
    (means, stds)
}

fn experiment_l(ls: &[i64], n: usize, u: usize) -> (Means, Stds) {
    eprintln!("[4/4] Sweep: number of loci L  (n={n}, u={u}, A=28, {N_SEEDS} placements/point)");
    let (mut means, mut stds) = new_maps();
    for &l_i64 in ls {
        let nl = l_i64 as usize;
        let loci: Vec<String> = (0..nl).map(|i| format!("L{i}")).collect();
        for (ti, &name) in TOPO_NAMES.iter().enumerate() {
            let pts: Vec<f64> = (0..N_SEEDS).map(|k| {
                let males = generate_pedigree(n, u, TOPO_FANS[ti], &loci, BASE_SEED + k as u64);
                measure_multi(&males, &loci, MU, MINALL, MAXALL)
            }).collect();
            let (m, s) = mean_std(&pts);
            means.get_mut(name).unwrap().push(m);
            stds.get_mut(name).unwrap().push(s);
        }
        progress_line("L", nl, &means, &stds);
    }
    (means, stds)
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

#[allow(non_snake_case)]
#[derive(Serialize)]
struct Output {
    ns:      Vec<i64>,
    res_n:   Means,
    std_n:   Stds,
    us:      Vec<i64>,
    res_u:   Means,
    std_u:   Stds,
    As:      Vec<i64>,
    res_A:   Means,
    std_A:   Stds,
    Ls:      Vec<i64>,
    res_L:   Means,
    std_L:   Stds,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let ns:      Vec<i64> = vec![50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500];
    let us:      Vec<i64> = vec![1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 199];
    let a_sizes: Vec<i64> = vec![4, 6, 8, 10, 14, 18, 22, 26, 28];
    let ls:      Vec<i64> = vec![1, 2, 3, 5, 8, 10, 15, 20];

    let (res_n, std_n) = experiment_n(&ns);
    let (res_u, std_u) = experiment_u(&us, 200);
    let (res_a, std_a) = experiment_a(&a_sizes, 200, 50);
    let (res_l, std_l) = experiment_l(&ls, 50, 25);

    #[allow(non_snake_case)]
    let output = Output {
        ns, res_n, std_n,
        us, res_u, std_u,
        As: a_sizes, res_A: res_a, std_A: std_a,
        Ls: ls,     res_L: res_l, std_L: std_l,
    };

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
