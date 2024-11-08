use crate::layer::Layer;
use ndarray::{Array1, Array2};

struct Network<'a> {
    n_inputs: usize,
    layers: &'a mut [Layer],
    output: Array2<f64>,
}

impl<'a> Network<'a> {
    pub fn init(n_inputs: usize, layers: &'a mut [Layer], n_outputs: usize) -> Self {
        Network {
            n_inputs,
            layers,
            output: Array2::zeros((n_outputs, 1)),
        }
    }

    pub fn forward(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            
        }
    }
}
