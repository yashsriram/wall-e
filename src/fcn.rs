use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Activation {
    Linear,
    LeakyReLu(f32),
    Sigmoid,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FCN {
    layers: Vec<(usize, Activation)>,
    params: Array1<f32>,
}

impl fmt::Display for FCN {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "fcn, layers={:?}, num params={}",
            self.layers,
            self.params.len()
        )
    }
}

impl FCN {
    pub fn new(layers: Vec<(usize, Activation)>) -> FCN {
        assert!(
            layers.len() >= 2,
            "Trying to create a model with less than 2 layers."
        );
        let num_params = {
            let mut num_params = 0;
            for i in 1..layers.len() {
                num_params += (layers[i - 1].0 + 1) * layers[i].0;
            }
            num_params
        };
        FCN {
            layers: layers,
            params: //Array::from_elem((num_params,), 0.01),
            Array::random(num_params, Uniform::new(0.0, 1.0)),
        }
    }

    pub fn params(&self) -> &Array1<f32> {
        &self.params
    }

    pub fn set_params(&mut self, new_params: Array1<f32>) {
        self.params = new_params;
    }

    /// Clones input but not params.
    pub fn at_with(&self, input: &Array1<f32>, params: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.layers[0].0, "Invalid input len for fcn");
        assert_eq!(
            params.len(),
            self.params.len(),
            "Invalid params len for fcn"
        );
        let mut params_offset = 0;
        let mut output = input.to_owned();
        output = match &self.layers[0].1 {
            Activation::Linear => output,
            Activation::LeakyReLu(leak) => output.mapv(|e| if e > 0.0 { e } else { e * leak }),
            Activation::Sigmoid => output.mapv(|e| 1.0 / (1.0 + (-e).exp())),
        };
        for i in 1..self.layers.len() {
            let prev_layer_dof = self.layers[i - 1].0;
            let curr_layer_dof = self.layers[i].0;
            let curr_layer_activation = &self.layers[i].1;
            // let now = Instant::now();
            let matrix = params
                .slice(s![
                    params_offset..(params_offset + prev_layer_dof * curr_layer_dof)
                ])
                .into_shape((curr_layer_dof, prev_layer_dof))
                .unwrap();
            params_offset += prev_layer_dof * curr_layer_dof;
            let bias = params
                .slice(s![params_offset..(params_offset + curr_layer_dof)])
                .into_shape(curr_layer_dof)
                .unwrap();
            // println!("1:{}", now.elapsed().as_nanos());
            // let now = Instant::now();
            output = matrix.dot(&output) + bias;
            output = match curr_layer_activation {
                Activation::Linear => output,
                Activation::LeakyReLu(leak) => output.mapv(|e| if e > 0.0 { e } else { e * leak }),
                Activation::Sigmoid => output.mapv(|e| 1.0 / (1.0 + (-e).exp())),
            };
            params_offset += curr_layer_dof;
            // println!("2:{}", now.elapsed().as_nanos());
        }
        output
    }

    pub fn at(&self, input: &Array1<f32>) -> Array1<f32> {
        self.at_with(&input, &self.params)
    }
}
