use anyhow::{Context, Result};
use ndarray::Array2;
use std::fs::File;
use std::io::Write;

pub fn csv_to_ndarray(file_path: &str) -> Result<Array2<f64>> {
    let mut rdr = csv::Reader::from_path(file_path).unwrap();

    let df = rdr
        .records()
        .map(|record| {
            record
                .context("Failed while iterating through csv records")
                .unwrap()
                .iter()
                .map(|x| {
                    x.parse()
                        .context("Failed while parsing string to f64")
                        .unwrap()
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let rows = df.len();
    let cols = df[0].len();

    let flat_df = df.into_iter().flatten().collect::<Vec<f64>>();
    let array = Array2::from_shape_vec((rows, cols), flat_df)?;

    Ok(array)
}

pub fn save_array_to_csv(arr: &Array2<f64>, file_path: &str) -> Result<()> {
    let mut file = File::create(file_path)?;
    // writeln!(file, "{},{}", arr.shape()[0], arr.shape()[1])?;
    for row in arr.rows() {
        let row_string: Vec<String> = row.iter().map(|x| x.to_string()).collect();
        writeln!(file, "{}", row_string.join(","))?;
    }
    Ok(())
}
