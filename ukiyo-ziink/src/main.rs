use std::{collections::HashMap, path::Path};

use polars::prelude::*;
use ukiyo_nn::Mlp;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::from_path(Path::new(".env"))?;

    let splits = HashMap::from([
        (
            "train",
            "hf://datasets/ylecun/mnist/mnist/train-00000-of-00001.parquet",
        ),
        (
            "test",
            "hf://datasets/ylecun/mnist/mnist/test-00000-of-00001.parquet",
        ),
    ]);
    let df = LazyFrame::scan_parquet(
        PlPath::from_str(splits.get("train").unwrap()),
        ScanArgsParquet::default(),
    )?
    .collect()?;
    let net = Mlp::new(784, &[128, 64, 10]);

    dbg!(df);

    Ok(())
}
