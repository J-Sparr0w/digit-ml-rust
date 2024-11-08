use anyhow::{Context, Ok, Result};
use layer::Layer;
use ndarray::{s, Array2};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};
use network::Network2;

mod layer;
mod network;

fn shuffle_rows(array: &mut Array2<f64>) {
    let mut rng = thread_rng();

    // Generate row indices.
    let mut indices: Vec<usize> = (0..array.nrows()).collect();

    // Shuffle indices.
    indices.as_mut_slice().shuffle(&mut rng);

    // Reorder the array based on shuffled indices.
    let shuffled_array = indices
        .iter()
        .map(|&i| array.slice(s![i, ..]).to_owned())
        .collect::<Vec<_>>();

    // Copy the shuffled rows back into the original array.
    for (i, row) in shuffled_array.into_iter().enumerate() {
        array.slice_mut(s![i, ..]).assign(&row);
    }
}

fn csv_to_ndarray(file_path: &str) -> Result<Array2<f64>> {
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

fn main() -> Result<()> {
    //data:
    //  each record has 785 columns
    //  column 1- label
    //  columns 2-785- pixel data ranging from 0-255
    // we will transpose the matrix so that each column represents image data
    // final shape: 785xm

    let mut data = csv_to_ndarray("data/train.csv")?;
    let (m, n) = (data.shape()[0], data.shape()[1]);
    shuffle_rows(&mut data); //shuffling is not reproducible

    let data_dev = data.slice(s![0..1000, ..]).t().to_owned();
    let y_dev = data_dev.slice(s![0, ..]);
    let x_dev = data_dev.slice(s![1..n, ..]).mapv(|x| x / 255.); //shape: [784, 1000]

    let data_train = data.slice(s![1000.., ..]).t().to_owned();
    let y_train = data_train.slice(s![0, ..]).to_owned();
    let x_train = data_train.slice(s![1..n, ..]).mapv(|x| x / 255.);

    //Network has two layers
    //input => 785xm
    //output => 10xm
    //where m is the number of samples in training set

    //W(l) => weights of layer l {shape: 10x785}
    //b(l) => biases of layer l
    //A(l) => output after applying activation function on W(l) for Layer l
    //Z(l) => weighted sum of layer l (wx+b)

    //Forward Prop:
    // X = input (or A0)
    // Z1 = W1 . X + b1
    // A1 = ReLu(Z1)
    // Z2 = W2 . A1 + b2
    // A2 = Softmax(Z2)
    // output = A2
    // Flow:  X->Z1->A1->Z2->A2

    //Backward Prop:
    // dZ2 = A2 - Y
    // dW2 = (1/m) * (dZ2 . A1.T)
    // dB2 = (1/m) sum(dZ2)
    // dZ1 = W2.T.dZ2 * ReLU_deriv(Z1)
    // dW1 = (1/m) * (dZ1 . (X.T))
    // dB1 = (1/m) * sum(dZ1)

    let mut network = Network2::init(
        m,
        &x_train,
        Layer::new(x_train.shape()[0], 10, layer::ActivationFn::ReLu),
        Layer::new(10, 10, layer::ActivationFn::Softmax),
        y_train,
    );

    network.train(0.1, 500);

    // println!("out shape: {:?}", out.shape());
    // println!("out shape: {:?}", out.slice(s![.., 0]));

    Ok(())
}
