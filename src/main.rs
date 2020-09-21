use gnuplot::{Caption, Color, Figure, LineWidth};
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_rand::rand_distr::{NormalError, StandardNormal};
use ndarray_rand::RandomExt;
use rand::Rng;
use std::fmt;

struct FCN {
    layers: Vec<usize>,
    params: Array1<f32>,
}

impl fmt::Display for FCN {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Fully connected network, layers={:?}, num params={}",
            self.layers,
            self.params.len()
        )
    }
}

impl FCN {
    fn new(layers: Vec<usize>) -> FCN {
        assert!(
            layers.len() >= 2,
            "Trying to create a model with less than 2 layers."
        );
        let num_params = {
            let mut num_params = 0;
            for i in 1..layers.len() {
                num_params += (layers[i - 1] + 1) * layers[i];
            }
            num_params
        };
        FCN {
            layers: layers,
            params: Array::from_elem((num_params,), 0.01),
        }
    }

    /// Clones input but not params.
    fn at_with(&self, input: &Array1<f32>, params: &ArrayView1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.layers[0], "Invalid input len for fcn");
        assert_eq!(
            params.len(),
            self.params.len(),
            "Invalid params len for fcn"
        );
        let mut params_offset = 0;
        let mut output = input.to_owned();
        for i in 1..self.layers.len() {
            let matrix = params
                .slice(s![
                    params_offset..(params_offset + self.layers[i - 1] * self.layers[i])
                ])
                .into_shape((self.layers[i], self.layers[i - 1]))
                .unwrap();
            params_offset += self.layers[i - 1] * self.layers[i];
            let biases = params
                .slice(s![params_offset..(params_offset + self.layers[i])])
                .into_shape(self.layers[i])
                .unwrap();
            output = matrix.dot(&output) + biases;
            output = output.mapv(|e| if e > 0.0 { e } else { e * 0.1 });
            params_offset += self.layers[i];
        }
        output
    }

    fn at(&self, input: Array1<f32>) -> Array1<f32> {
        self.at_with(&input, &self.params.slice(s![..]))
    }
}

fn reward(fcn: &FCN, params: &ArrayView1<f32>, num_samples: usize) -> f32 {
    let mut rng = rand::thread_rng();
    let max_x = 6.28;
    let cumulative_reward = (0..num_samples)
        .map(|_| {
            let x = rng.gen::<f32>() * max_x;
            let y_true = x.exp();
            let y_pred = fcn.at_with(&arr1(&[x]), &params.slice(s![..]))[0];
            -(y_true - y_pred) * (y_true - y_pred)
        })
        .sum::<f32>();
    cumulative_reward / num_samples as f32
}

fn cem(
    fcn: &mut FCN,
    n_iter: usize,
    batch_size: usize,
    num_evalation_samples: usize,
    elite_frac: f32,
    initial_std: f32,
    noise_factor: f32,
) -> Result<Array1<f32>, NormalError> {
    let n_elite = (batch_size as f32 * elite_frac).round().floor() as usize;
    let mut th_std = Array::from_elem((fcn.params.len(),), initial_std);
    for iter in 0..n_iter {
        let randn_noise_for_batch: Array2<f32> =
            Array::random((batch_size, fcn.params.len()), StandardNormal);
        let scaled_randn_noise_for_batch = randn_noise_for_batch * &th_std;
        let th_means_for_batch = scaled_randn_noise_for_batch + &fcn.params;
        // println!("{:?}", th_means_for_batch);
        let (sorted_th_means, mean_reward) = {
            let mut reward_th_mean_tuples = th_means_for_batch
                .axis_iter(Axis(0))
                .map(|th_mean| (reward(fcn, &th_mean, num_evalation_samples), th_mean))
                .collect::<Vec<(f32, ArrayView1<f32>)>>();
            reward_th_mean_tuples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            reward_th_mean_tuples.reverse();
            let (rewards, sorted_th_means): (Vec<_>, Vec<_>) =
                reward_th_mean_tuples.into_iter().unzip();
            (
                sorted_th_means,
                rewards.iter().sum::<f32>() / rewards.len() as f32,
            )
        };
        let elite_ths = sorted_th_means
            .into_iter()
            .take(n_elite)
            .collect::<Vec<ArrayView1<f32>>>();
        // println!("{:?}", elite_ths);
        let elite_ths = stack(Axis(0), &elite_ths)
            .unwrap()
            .into_shape((n_elite, fcn.params.len()))
            .unwrap();
        // println!("{:?}", elite_ths);
        fcn.params = elite_ths.mean_axis(Axis(0)).unwrap();
        // println!("{:?}", th_mean);
        th_std = elite_ths.std_axis(Axis(0), 0.0);
        th_std += noise_factor / (iter + 1) as f32;
        // println!("{:?}", th_std);
        println!(
            "iter={} mean_reward={:?} reward_with_current_th={:?}, th_std_mean={:?}",
            iter + 1,
            mean_reward,
            reward(fcn, &fcn.params.slice(s![..]), num_evalation_samples),
            th_std.mean(),
        );
    }
    Ok(th_std)
}

fn main() {
    let mut fcn = FCN::new(vec![1, 5, 5, 5, 5, 1]);
    println!("{}", fcn);
    let _th_std = cem(&mut fcn, 200, 50, 300, 0.5, 1.0, 1.0).unwrap();

    let mut fg = Figure::new();
    fg.axes2d()
        .lines(
            (0..=314).map(|x| x as f32 / 50.0),
            (0..=314).map(|x| (x as f32 / 50.0).exp()),
            &[Caption("true"), LineWidth(1.0), Color("green")],
        )
        .lines(
            (0..=314).map(|x| x as f32 / 50.0),
            (0..=314)
                .map(|x| x as f32 / 50.0)
                .map(|x| fcn.at(arr1(&[x]))[[0]]),
            &[Caption("pred"), LineWidth(1.0), Color("red")],
        );
    fg.save_to_png("fit.png", 800, 500);
}
