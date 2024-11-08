use anyhow::{Context, Ok, Result};
use layer::Layer;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};

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
    // Flow:

    // let mut layers: [Layer; 2] = [
    //     ];
    let x = &x_train;
    let mut layer1 = Layer::new(x.shape()[0], 10, layer::ActivationFn::ReLu);
    let mut layer2 = Layer::new(10, 10, layer::ActivationFn::Softmax);

    let iterations = 100;

    for i in 1..=iterations {
        // println!("starting");
        layer1.forward(&x_train);
        // println!("forwarded layer1");
        layer2.forward(&layer1.a);
        // println!("forwarded layer2");

        let alpha = 0.1;
        let one_hot_y = one_hot_encode(&y_train);
        let dz2 = layer2.a.clone() - one_hot_y;
        // println!("m={}, 1/m = {:?}", m, (1 / m) as f64);
        // println!("dz2 = a2-y: {:?}", dz2.slice(s![.., 0]));
        let dw2 = (1. / m as f64) as f64 * dz2.dot(&layer1.a.t());
        let db2 = (1. / m as f64) * dz2.sum();
        let dz1 = layer2.weights.t().dot(&dz2) * relu_deriv(&layer1.z);
        let dw1 = (1. / m as f64) as f64 * (dz1.dot(&x.t()));
        let db1 = (1. / m as f64) as f64 * dz1.sum();

        layer1.update(alpha, &dw1, db1);
        layer2.update(alpha, &dw2, db2);

        if i % 10 == 0 {
            println!("Iteration: {}", i);
            let predictions = get_predictions(&layer2.a);
            println!("Accuracy: {}", get_accuracy(&predictions, &y_train));
        }
    }

    // println!("out shape: {:?}", out.shape());
    // println!("out shape: {:?}", out.slice(s![.., 0]));

    Ok(())
}

fn get_predictions(a2: &Array2<f64>) -> Array1<f64> {
    // a2.axis_iter(Axis(0))

    let preds = a2
        .axis_iter(Axis(1))
        .map(|col| {
            col.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index as f64)
                .unwrap() // Safe unwrap since we know the column is not empty
        })
        .collect();
    Array1::from_vec(preds)
}

fn get_accuracy(predictions: &Array1<f64>, y: &Array1<f64>) -> f64 {
    println!("predictions: {:?}\ny: {:?}", predictions, y);
    let correct_predictions = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, actual)| pred == actual)
        .count();
    correct_predictions as f64 / y.len() as f64
}
pub fn relu_deriv(z: &Array2<f64>) -> Array2<f64> {
    return z.mapv(|x| if x > 0. { 1. } else { 0. });
}

pub fn one_hot_encode(y: &Array1<f64>) -> Array2<f64> {
    // y (y_dev and y_train) is a singlerow currently
    // we want to encode it and turn it into a 2d array =>
    // [1,0,4,2] should be
    // [
    //  [0,1,0,0]
    //  [1,0,0,0]
    //  [0,0,0,1]
    //  [0,0,0,0]
    //  [0,0,1,0]
    //  ]
    let mut arr = Array2::zeros((10, y.shape()[0]));
    for (i, &x) in y.iter().enumerate() {
        arr[(x as usize, i)] = 1.;
    }
    // println!("y= {:?}", y);
    // println!("{:?}", arr.slice(s![.., 0]));
    arr
}
