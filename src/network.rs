use crate::layer::Layer;
use ndarray::{Array1, Array2, Axis};

pub struct Network2<'a> {
    n_input: usize,
    x: &'a Array2<f64>,
    layer1: Layer,
    layer2: Layer,
    y: Array1<f64>,
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
    // ]
    let mut arr = Array2::zeros((10, y.len()));
    println!("y shape= {:?}", y.shape());
    for (i, &x) in y.iter().enumerate() {
        arr[(x as usize, i)] = 1.;
    }
    // println!("y= {:?}", y);
    // println!("{:?}", arr.slice(s![.., 0]));
    arr
}

fn get_predictions(a2: &Array2<f64>) -> Array1<f64> {
    let preds = a2
        .axis_iter(Axis(1))
        .map(|col| {
            col.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect(&format!("a:{}, b: {}", a, b)))
                .map(|(index, _)| index as f64)
                .expect(&format!("Column must not be empty"))
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

impl<'a> Network2<'a> {
    pub fn init(
        n_input: usize,
        x: &'a Array2<f64>,
        layer1: Layer,
        layer2: Layer,
        y: Array1<f64>,
    ) -> Network2 {
        Network2 {
            n_input,
            x,
            layer1,
            layer2,
            y,
        }
    }

    pub fn train(&mut self, alpha: f64, iterations: i32) {
        let m = self.n_input;
        let one_hot_y = one_hot_encode(&self.y);

        for i in 1..=iterations {
            self.layer1.forward(&self.x);
            self.layer2.forward(&self.layer1.a);

            let dz2 = self.layer2.a.clone() - &one_hot_y;
            // println!("m={}, 1/m = {:?}", m, (1 / m) as f64);
            // println!("dz2 = a2-y: {:?}", dz2.slice(s![.., 0]));
            let dw2 = (1. / m as f64) as f64 * dz2.dot(&self.layer1.a.t());
            let db2 = (1. / m as f64) * dz2.sum();
            let dz1 = self.layer2.weights.t().dot(&dz2) * relu_deriv(&self.layer1.z);
            let dw1 = (1. / m as f64) as f64 * (dz1.dot(&self.x.t()));
            let db1 = (1. / m as f64) as f64 * dz1.sum();

            self.layer1.update(alpha, &dw1, db1);
            self.layer2.update(alpha, &dw2, db2);

            if i % 10 == 0 {
                println!("Iteration: {}", i);
                let predictions = get_predictions(&self.layer2.a);
                println!("Accuracy: {}", get_accuracy(&predictions, &self.y));
            }
        }
    }
}



