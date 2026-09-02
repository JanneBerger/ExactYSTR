//! Command-line interface for the exact Y-STR pedigree match probability
//! calculator (Rust implementation).
//!
//! Usage:
//!   ystr_cli <PEDIGREE_CSV> [--mu 0.1] [--out results.csv]

use std::error::Error;
use std::path::PathBuf;

use clap::Parser;
use ystr_rust::{compute_match_probs_fast, load_pedigree_from_csv, DEFAULT_MU, MAXALL, MINALL};

#[derive(Parser)]
#[command(
    name = "ystr_cli",
    about = "Exact Y-STR pedigree match probabilities (Elston-Stewart algorithm)",
    version
)]
struct Cli {
    /// Path to the input pedigree CSV file
    pedigree: PathBuf,

    /// Mutation rate per meiosis
    #[arg(long, default_value_t = DEFAULT_MU)]
    mu: f64,

    /// Write results to this CSV file instead of stdout
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();

    let (males, loci) = load_pedigree_from_csv(&args.pedigree).map_err(|e| {
        eprintln!("Error loading pedigree: {e}");
        e
    })?;

    println!("Loaded {} males, {} loci from '{}'",
        males.len(), loci.len(), args.pedigree.display());
    println!("Mutation rate: mu = {}", args.mu);
    println!("Allele range:  [{}, {}]\n", MINALL, MAXALL);

    struct Row {
        locus:      String,
        male_id:    u32,
        generation: u32,
        match_prob: f64,
    }
    let mut rows: Vec<Row> = Vec::new();

    for locus in &loci {
        let (match_probs, average) =
            compute_match_probs_fast(&males, locus, args.mu, MINALL, MAXALL);

        if match_probs.is_empty() {
            println!("[{locus}]  suspect untyped or no untyped males — skipped");
            continue;
        }

        println!("[{locus}]  average MP = {average:.6}");

        let mut entries: Vec<(usize, f64)> = match_probs.into_iter().collect();
        entries.sort_by_key(|(i, _)| *i);
        for (i, prob) in entries {
            let male_id    = males[i].male_id;
            let generation = males[i].generation;
            println!("    male_id={male_id}  gen={generation}  MP={prob:.6}");
            rows.push(Row { locus: locus.clone(), male_id, generation, match_prob: prob });
        }
        println!();
    }

    if let Some(out_path) = &args.out {
        if !rows.is_empty() {
            let file = std::fs::File::create(out_path)?;
            let mut wtr = csv::Writer::from_writer(file);
            wtr.write_record(["locus", "male_id", "generation", "match_prob"])?;
            for r in &rows {
                wtr.write_record(&[
                    r.locus.clone(),
                    r.male_id.to_string(),
                    r.generation.to_string(),
                    format!("{:.10}", r.match_prob),
                ])?;
            }
            wtr.flush()?;
            println!("Results written to '{}'", out_path.display());
        }
    }

    Ok(())
}
