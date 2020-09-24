use crate::fcn::*;
use ndarray::prelude::*;
use rand::Rng;

#[derive(Debug)]
pub struct ExpReward;

impl ExpReward {
    pub fn reward(fcn: &FCN, params: &Array1<f32>, num_samples: usize) -> f32 {
        let mut rng = rand::thread_rng();
        let max_x = 6.28;
        let cumulative_reward = (0..num_samples)
            .map(|_| {
                let x = rng.gen::<f32>() * max_x;
                let y_true = x.exp();
                let y_pred = fcn.at_with(&arr1(&[x]), &params)[0];
                -(y_true - y_pred) * (y_true - y_pred)
            })
            .sum::<f32>();
        cumulative_reward / num_samples as f32
    }
}

#[derive(Debug)]
pub struct SinReward;

impl SinReward {
    pub fn reward(fcn: &FCN, params: &Array1<f32>, num_samples: usize) -> f32 {
        let mut rng = rand::thread_rng();
        let max_x = 6.28;
        let cumulative_reward = (0..num_samples)
            .map(|_| {
                let x = rng.gen::<f32>() * max_x;
                let y_true = x.sin();
                let y_pred = fcn.at_with(&arr1(&[x]), &params)[0];
                -(y_true - y_pred) * (y_true - y_pred)
            })
            .sum::<f32>();
        cumulative_reward / num_samples as f32
    }
}
