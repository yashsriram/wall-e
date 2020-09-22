use ndarray::prelude::*;
use ndarray::stack;
use ndarray_rand::rand_distr::{NormalError, StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use rand::Rng;
use rayon::prelude::*;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug)]
enum Activation {
    Linear,
    LeakyReLu(f32),
    Sigmoid,
}

struct FCN {
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
    fn new(layers: Vec<(usize, Activation)>) -> FCN {
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
            params: Array::from_elem((num_params,), 0.01),
            // Array::random(num_params, Uniform::new(0.0, 1.0)),
        }
    }

    /// Clones input but not params.
    fn at_with(&self, input: &Array1<f32>, params: &Array1<f32>) -> Array1<f32> {
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

    fn at(&self, input: &Array1<f32>) -> Array1<f32> {
        self.at_with(&input, &self.params)
    }
}

fn reward(fcn: &FCN, params: &Array1<f32>, num_samples: usize) -> f32 {
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

#[derive(Debug)]
struct CEO {
    n_iter: usize,
    batch_size: usize,
    num_evalation_samples: usize,
    elite_frac: f32,
    initial_std: f32,
    noise_factor: f32,
}

impl Default for CEO {
    fn default() -> CEO {
        CEO {
            n_iter: 300,
            batch_size: 50,
            num_evalation_samples: 300,
            elite_frac: 0.25,
            initial_std: 2.0,
            noise_factor: 2.0,
        }
    }
}

impl CEO {
    fn optimize(&self, fcn: &mut FCN) -> Result<Array1<f32>, NormalError> {
        let n_elite = (self.batch_size as f32 * self.elite_frac).round().floor() as usize;
        let mut noise_std = Array::from_elem((fcn.params.len(),), self.initial_std);
        for iter in 0..self.n_iter {
            let (sorted_th_means, mean_reward) = {
                let mut reward_th_mean_tuples = (0..self.batch_size)
                    .into_par_iter()
                    .map(|_| {
                        let randn_noise: Array1<f32> =
                            Array::random(fcn.params.len(), StandardNormal);
                        let scaled_randn_noise = randn_noise * &noise_std;
                        let perturbed_params = scaled_randn_noise + &fcn.params;
                        (
                            reward(fcn, &perturbed_params, self.num_evalation_samples),
                            perturbed_params,
                        )
                    })
                    .collect::<Vec<(f32, Array1<f32>)>>();
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
                .iter()
                .take(n_elite)
                .map(|th| th.slice(s![..]))
                .collect::<Vec<ArrayView1<f32>>>();
            let elite_ths = stack(Axis(0), &elite_ths)
                .unwrap()
                .into_shape((n_elite, fcn.params.len()))
                .unwrap();
            fcn.params = elite_ths.mean_axis(Axis(0)).unwrap();
            noise_std = elite_ths.std_axis(Axis(0), 0.0);
            noise_std += self.noise_factor / (iter + 1) as f32;
            println!(
                "iter={} mean_reward={:?} reward_with_current_th={:?}, th_std_mean={:?}",
                iter + 1,
                mean_reward,
                reward(fcn, &fcn.params, self.num_evalation_samples),
                noise_std.mean(),
            );
        }
        Ok(noise_std)
    }
}

fn main() {
    let mut fcn = FCN::new(vec![
        (1, Activation::Linear),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (1, Activation::Linear),
    ]);
    println!("{}", fcn);
    let ceo = CEO::default();
    println!("{:?}", ceo);
    let _th_std = ceo.optimize(&mut fcn).unwrap();

    use gnuplot::*;
    let mut fg = Figure::new();
    fg.axes2d()
        .lines(
            (0..=314).map(|x| x as f32 / 50.0),
            (0..=314).map(|x| (x as f32 / 50.0).exp()),
            &[Caption("true"), LineWidth(1.0), Color("green")],
        )
        .lines(
            (0..=314).map(|x| x as f32 / 50.0),
            (0..=314).map(|x| fcn.at(&arr1(&[x as f32 / 50.0]))[[0]]),
            &[Caption("pred"), LineWidth(1.0), Color("red")],
        )
        .set_legend(
            Graph(0.5),
            Graph(0.9),
            &[Placement(AlignCenter, AlignTop)],
            &[TextAlign(AlignRight)],
        )
        .set_grid_options(true, &[LineStyle(SmallDot), Color("black")])
        .set_x_grid(true)
        .set_y_grid(true)
        .set_title(
            &format!(
                "reward={}\nmodel={}\nceo={:?}",
                reward(&fcn, &fcn.params, ceo.num_evalation_samples),
                fcn,
                ceo
            ),
            &[],
        );
    fg.save_to_png(format!("{}.png", chrono::offset::Local::now()), 800, 500);
}
