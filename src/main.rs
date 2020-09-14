use ndarray::prelude::*;
use ndarray::stack;
use ndarray_rand::rand_distr::{NormalError, StandardNormal};
use ndarray_rand::RandomExt;
use rand::Rng;
use std::iter::FromIterator;

struct I1O1LinearModel;

impl I1O1LinearModel {
    fn run(input: ArrayView1<f32>, params: ArrayView1<f32>) -> Array1<f32> {
        let matrix = params.slice(s![0..1]);
        let bias = params.slice(s![1..2]);
        matrix.dot(&input) + &bias
    }
}

struct InHnOnReLuModel;

impl InHnOnReLuModel {
    const HIDDEN_LEN: usize = 10;
    const OUTPUT_LEN: usize = 1;

    fn legal_params_len(input_len: usize) -> usize {
        Self::HIDDEN_LEN * input_len
            + Self::HIDDEN_LEN
            + Self::OUTPUT_LEN * Self::HIDDEN_LEN
            + Self::OUTPUT_LEN
    }

    fn run(input: ArrayView1<f32>, params: ArrayView1<f32>) -> Array1<f32> {
        let input_len = input.len();
        assert_eq!(
            params.len(),
            Self::legal_params_len(input_len),
            "Number of parameters are inconsistent with layer lengths"
        );

        // Layer 1 (input -> hidden)
        let input = input.slice(s![..]).into_shape((input_len, 1)).unwrap();
        let l1_end = Self::HIDDEN_LEN * input_len;
        let matrix1 = params
            .slice(s![0..l1_end])
            .into_shape((Self::HIDDEN_LEN, input_len))
            .unwrap();
        let biases1 = params
            .slice(s![l1_end..l1_end + Self::HIDDEN_LEN])
            .into_shape((Self::HIDDEN_LEN, 1))
            .unwrap();
        let hidden_out = matrix1.dot(&input) + biases1;
        let hidden_out = hidden_out.mapv(|e| if e > 0.0 { e } else { e * 0.1 });

        // Layer 2 (hiden -> output)
        let l2_start = l1_end + Self::HIDDEN_LEN;
        let m2_end = l2_start + Self::OUTPUT_LEN * Self::HIDDEN_LEN;
        let matrix2 = params
            .slice(s![l2_start..m2_end])
            .into_shape((Self::OUTPUT_LEN, Self::HIDDEN_LEN))
            .unwrap();
        let biases2 = params
            .slice(s![m2_end..m2_end + Self::OUTPUT_LEN])
            .into_shape((Self::OUTPUT_LEN, 1))
            .unwrap();
        let output = matrix2.dot(&hidden_out) + biases2;

        Array::from_iter(output.iter().map(|&e| e))
    }
}

fn reward(params: ArrayView1<f32>, num_samples: usize) -> f32 {
    let mut rng = rand::thread_rng();
    let max_x = 6.28;
    let cumulative_reward = (0..num_samples)
        .map(|_| {
            let x = rng.gen::<f32>() * max_x;
            let y_true = x.sin();
            let y_pred = InHnOnReLuModel::run(arr1(&[x]).slice(s![..]), params)[0];
            -(y_true - y_pred).abs()
        })
        .sum::<f32>();
    cumulative_reward / num_samples as f32
}

fn cem(
    mut th_mean: Array1<f32>,
    batch_size: usize,
    n_iter: usize,
    elite_frac: f32,
    initial_std: f32,
    num_evalation_samples: usize,
    noise_factor: f32,
) -> Result<(), NormalError> {
    let mut th_std = Array::from_elem((th_mean.len(),), initial_std);

    let n_elite = (batch_size as f32 * elite_frac).round().floor() as usize;
    for iter in 0..n_iter {
        let randn_noise_for_batch: Array2<f32> =
            Array::random((batch_size, th_mean.len()), StandardNormal);
        let scaled_randn_noise_for_batch = randn_noise_for_batch * &th_std;
        let th_means_for_batch = scaled_randn_noise_for_batch + &th_mean;
        // println!("{:?}", th_means_for_batch);
        let (sorted_th_means, mean_reward) = {
            let mut reward_th_mean_tuples = th_means_for_batch
                .axis_iter(Axis(0))
                .map(|th_mean| (reward(th_mean, num_evalation_samples), th_mean))
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
            .into_shape((n_elite, th_mean.len()))
            .unwrap();
        // println!("{:?}", elite_ths);
        th_mean = elite_ths.mean_axis(Axis(0)).unwrap();
        // println!("{:?}", th_mean);
        th_std = elite_ths.std_axis(Axis(0), 0.0);
        th_std += noise_factor / (iter + 1) as f32;
        // println!("{:?}", th_std);
        println!(
            "iter={} mean_reward={:?} reward_with_current_th={:?}, th_std_mean={:?}",
            iter + 1,
            mean_reward,
            reward(th_mean.slice(s![..]), num_evalation_samples),
            th_std.mean(),
        );
    }
    Ok(())
}

fn main() {
    cem(
        Array::from_elem((InHnOnReLuModel::legal_params_len(1),), 0.0),
        50,
        50,
        0.5,
        1.0,
        300,
        1.0,
    );
}
