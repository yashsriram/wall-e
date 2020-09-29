use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use wall_e::ceo::Reward;
use wall_e::diff_drive_model::DiffDriveModel;
use wall_e::fcn::*;
use wall_e::goal::Goal;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffDriveReward {
    init_x_bounds: (f32, f32),
    init_y_bounds: (f32, f32),
    init_or_bounds: (f32, f32),
    radius: f32,
    goal_x_bounds: (f32, f32),
    goal_y_bounds: (f32, f32),
    num_episode_ticks: usize,
}

impl DiffDriveReward {
    pub fn new(
        init_x_bounds: (f32, f32),
        init_y_bounds: (f32, f32),
        init_or_bounds: (f32, f32),
        radius: f32,
        goal_x_bounds: (f32, f32),
        goal_y_bounds: (f32, f32),
        num_episode_ticks: usize,
    ) -> DiffDriveReward {
        DiffDriveReward {
            init_x_bounds: init_x_bounds,
            init_y_bounds: init_y_bounds,
            init_or_bounds: init_or_bounds,
            radius: radius,
            goal_x_bounds: goal_x_bounds,
            goal_y_bounds: goal_y_bounds,
            num_episode_ticks: num_episode_ticks,
        }
    }
    pub fn init_x_bounds(&self) -> (f32, f32) {
        self.init_x_bounds
    }
    pub fn init_y_bounds(&self) -> (f32, f32) {
        self.init_y_bounds
    }
    pub fn init_or_bounds(&self) -> (f32, f32) {
        self.init_or_bounds
    }
    pub fn radius(&self) -> f32 {
        self.radius
    }
    pub fn goal_x_bounds(&self) -> (f32, f32) {
        self.goal_x_bounds
    }
    pub fn goal_y_bounds(&self) -> (f32, f32) {
        self.goal_y_bounds
    }
}

impl Reward for DiffDriveReward {
    fn reward(&self, fcn: &FCN, params: &Array1<f32>, num_episodes: usize) -> f32 {
        let mut cumulative_reward = 0.0;
        for _ in 0..num_episodes {
            // Set goal
            let goal_coordinates =
                Goal::in_region(self.goal_x_bounds, self.goal_y_bounds).coordinates();
            // Spawn agent
            let mut model = DiffDriveModel::spawn_randomly(
                self.init_x_bounds,
                self.init_y_bounds,
                self.init_or_bounds,
                self.radius,
                goal_coordinates,
            );
            // Start calculating reward
            let mut episode_reward = 0.0;
            for _ in 0..self.num_episode_ticks {
                // Curr state
                let (x, y, or_in_rad) = model.scaled_state();
                // Control for curr state
                let control = fcn.at_with(&arr1(&[x, y, or_in_rad]), params);
                let (v, w) = (control[[0]], control[[1]]);
                // Apply control
                model.set_control(0.0, w);
                model.update(0.1).unwrap();
                // Next state
                let (x, y, or_in_rad) = model.scaled_state();
                let angular_deviation =
                    ((x - or_in_rad.cos()).powf(2.0) + (y - or_in_rad.sin()).powf(2.0)).sqrt();
                // let dist = (x * x + y * y).sqrt();
                // episode_reward -= dist;
                episode_reward -= w.abs();
                episode_reward -= angular_deviation;
                // println!("d {:?}", angular_deviation);
                // println!("e {:?}", episode_reward);
            }

            // let (x, y, or_in_rad) = model.scaled_state(goal_coordinates);
            // let final_dist = (x * x + y * y).sqrt();
            // if final_dist < 6.0 * Goal::SLACK {
            //     episode_reward += 100.0
            // }
            // if final_dist < 4.0 * Goal::SLACK {
            //     episode_reward += 1000.0
            // }

            // let (v, w) = model.control();
            // if final_dist < 2.0 * Goal::SLACK {
            //     episode_reward += 2000.0;
            //     if v.abs() < 5.0 {
            //         episode_reward += 10000.0;
            //     }
            //     if w.abs() < 5.0 {
            //         episode_reward += 10000.0;
            //     }
            // }

            cumulative_reward += episode_reward;
        }

        let average_reward = cumulative_reward / num_episodes as f32;
        average_reward
    }
}
