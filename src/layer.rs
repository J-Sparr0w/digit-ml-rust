use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::{self, Uniform},
    RandomExt,
};

#[derive(Debug)]
pub enum ActivationFn {
    Sigmoid,
    ReLu,
    Softmax,
}

#[derive(Debug)]
pub struct Layer {
    // pub weights: Array2<f64>,
    // biases: Array2<f64>,
    // x: Array2<f64>, // input matrix
    pub n_neurons: usize,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub z: Array2<f64>, //weighted sum matrix
    pub a: Array2<f64>, //output
    pub activation_fn: ActivationFn,
}

impl Layer {
    pub fn new(
        n_input_neurons: usize, // input matrix is 785 x m, therefore this value will be 785
        n_neurons: usize,
        activation_fn: ActivationFn,
    ) -> Layer {
        //biases and weights have been checked and they are random but is reproducible
        let mut rng = StdRng::seed_from_u64(42);
        Layer {
            weights: Array2::random_using(
                (n_neurons, n_input_neurons),
                Uniform::new(-0.5, 0.5),
                &mut rng,
            ),
            biases: Array2::random_using((n_neurons, 1), Uniform::new(-0.5, 0.5), &mut rng),
            z: Array2::default((1, 1)),
            a: Array2::default((1, 1)),
            n_neurons,
            activation_fn,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>) {
        // println!("weight matrix dim: {:?}", self.weights.dim());
        // println!("input matrix dim: {:?}", input.dim());

        // println!("out: {:?}", out.slice(s![.., 0]));
        // println!("result shape: {:?}", result.shape());
        // result

        self.z = self.weights.dot(x)
            + &self
                .biases
                .broadcast((self.n_neurons, x.shape()[1]))
                .unwrap();
        self.apply_activation_fn();
    }

    // pub fn backward(&self, x: &Array2<f64>, out: &Array2<f64>) -> (Array2<f64>, f64) {
    //     let dw = (1 / 785) as f64 * out.dot(&x.t());
    //     let db = (1 / 785) as f64 * out.sum();
    //     (dw, db)
    // }

    pub fn update(&mut self, alpha: f64, dw: &Array2<f64>, db: f64) {
        // self.weights
        //     .iter_mut()
        //     .zip(dw.iter())
        //     .for_each(|(w, dw)| *w -= alpha * dw);
        self.weights.scaled_add(-alpha, dw);
        // println!("biases:\n\t before: \n{}", self.biases);

        self.biases -= (alpha * db) as f64;
        // println!("\t after: {}", self.biases);
        // println!("db: {}", db)
    }

    pub fn apply_activation_fn(&mut self) -> () {
        match self.activation_fn {
            ActivationFn::Sigmoid => todo!(),
            ActivationFn::ReLu => self.a = self.z.mapv(|x| if x < 0. { 0. } else { x }),
            ActivationFn::Softmax => {
                let exp_arr = self.z.mapv(|x| x.exp());
                let sum = exp_arr.sum();
                self.a = exp_arr.mapv(|x| x / sum);
            }
        }
    }
}
