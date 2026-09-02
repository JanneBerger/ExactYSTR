//! Exact pedigree-based Y-STR match probability calculator.
//!
//! Implements the two-pass Elston-Stewart inside-outside algorithm for
//! Y-chromosomal STRs, ported from the Python reference implementation.
//!
//! # Mutation model
//! One-step symmetric stepwise:
//! ```text
//!   p(j → k) = 1 − μ   if k == j
//!              μ / 2    if |k − j| == 1
//!              0        otherwise
//! ```

use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MINALL: i32 = 3;
pub const MAXALL: i32 = 30;
pub const DEFAULT_MU: f64 = 0.1;

// ---------------------------------------------------------------------------
// Data structure
// ---------------------------------------------------------------------------

/// A single male in the pedigree.
#[derive(Debug, Clone)]
pub struct Male {
    pub male_id:    u32,
    pub generation: u32,
    pub father_id:  i64,       // -1 for root (no father)
    pub alleles:    HashMap<String, i32>,  // locus → allele (0 = untyped)
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

/// Load a pedigree from a CSV file.
///
/// Expected columns:
/// - `male_id`      : u32   — unique individual ID
/// - `generation`   : u32   — 0-based (0 = suspect / MRCA)
/// - `father_id`    : i64   — father's male_id; use -1 for root
/// - `allele_<LOC>` : i32   — allele at locus LOC; 0 = untyped
///
/// Returns `(males, loci)` sorted by (generation, male_id).
pub fn load_pedigree_from_csv<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<Male>, Vec<String>), Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;

    let headers = rdr.headers()?.clone();
    let loci: Vec<String> = headers
        .iter()
        .filter(|h| h.starts_with("allele_"))
        .map(|h| h.replace("allele_", ""))
        .collect();

    let mut males: Vec<Male> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let get = |col: &str| -> Result<&str, Box<dyn std::error::Error>> {
            let pos = headers.iter().position(|h| h == col)
                .ok_or_else(|| format!("missing column: {col}"))?;
            Ok(&record[pos])
        };

        let male_id:    u32 = get("male_id")?.parse()?;
        let generation: u32 = get("generation")?.parse()?;
        let father_id:  i64 = get("father_id")?.parse()?;

        let alleles: HashMap<String, i32> = loci
            .iter()
            .map(|loc| {
                let col = format!("allele_{loc}");
                let pos = headers.iter().position(|h| h == col.as_str()).unwrap();
                let val: i32 = record[pos].parse().unwrap_or(0);
                (loc.clone(), val)
            })
            .collect();

        males.push(Male { male_id, generation, father_id, alleles });
    }

    males.sort_by_key(|m| (m.generation, m.male_id));
    Ok((males, loci))
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

/// Compute match probabilities for all untyped males at one locus.
///
/// For each untyped male i the match probability is:
/// ```text
///   MP_i = P(data | male i has suspect allele) / P(data)
/// ```
///
/// Returns `(match_probs, average)`.
/// `match_probs` maps index-in-`males` → match probability (untyped only).
/// Returns `({}, 0.0)` when the suspect is untyped or no untyped males exist.
pub fn compute_match_probs_fast(
    males:  &[Male],
    locus:  &str,
    mu:     f64,
    minall: i32,
    maxall: i32,
) -> (HashMap<usize, f64>, f64) {
    let base: Vec<i32> = males
        .iter()
        .map(|m| *m.alleles.get(locus).unwrap_or(&0))
        .collect();
    let suspect = base[0];

    if suspect == 0 {
        return (HashMap::new(), 0.0);
    }

    let n = males.len();
    let id_to_idx: HashMap<u32, usize> =
        males.iter().enumerate().map(|(i, m)| (m.male_id, i)).collect();

    let mut sons: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, m) in males.iter().enumerate() {
        if m.father_id >= 0 {
            if let Some(&fi) = id_to_idx.get(&(m.father_id as u32)) {
                sons[fi].push(i);
            }
        }
    }

    let a   = (maxall - minall + 1) as usize;
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
                if e != 0 && e != k { continue; }
                let ki = idx(k);
                let mut val = 1.0_f64;
                for &s in &sons[i] { val *= message[s][ki]; }
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
        if kids.is_empty() { continue; }
        let m_sons = kids.len();

        let mut prefix: Vec<Vec<f64>> = vec![vec![1.0; a]; m_sons + 1];
        let mut suffix: Vec<Vec<f64>> = vec![vec![1.0; a]; m_sons + 1];

        for t in 0..m_sons {
            let s = kids[t];
            for kk in 0..a { prefix[t + 1][kk] = prefix[t][kk] * message[s][kk]; }
        }
        for t in (0..m_sons).rev() {
            let s = kids[t];
            for kk in 0..a { suffix[t][kk] = suffix[t + 1][kk] * message[s][kk]; }
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
        return (HashMap::new(), 0.0);
    }

    let mut match_probs: HashMap<usize, f64> = HashMap::new();
    for (i, &allele) in base.iter().enumerate() {
        if allele == 0 {
            let p2 = alpha[i][si] * beta[i][si];
            match_probs.insert(i, p2 / p1);
        }
    }
    let average = if match_probs.is_empty() {
        0.0
    } else {
        match_probs.values().sum::<f64>() / match_probs.len() as f64
    };
    (match_probs, average)
}
