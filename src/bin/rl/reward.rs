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
    pub fn num_episode_ticks(&self) -> usize {
        self.num_episode_ticks
    }
}

impl Reward for DiffDriveReward {
    fn reward(&self, fcn: &FCN, params: &Array1<f32>, num_episodes: usize) -> f32 {
        let mut total_reward = 0.0;
        for _ in 0..num_episodes {
            let mut episode_reward = 0.0;
            let mut model = DiffDriveModel::spawn_randomly(
                self.init_x_bounds,
                self.init_y_bounds,
                self.init_or_bounds,
                self.radius,
            );
            let goal = Goal::in_region(self.goal_x_bounds, self.goal_y_bounds);
            let (goal_x, goal_y) = goal.coordinates();

            for _ in 0..self.num_episode_ticks {
                let (x, y, or_in_rad) = model.state();
                let control = fcn.at_with(&arr1(&[x, y, or_in_rad, goal_x, goal_y]), params);
                let (v, w) = (control[[0]], control[[1]]);
                model.set_control(v, w);
                model.update(0.1).unwrap();
                let (x, y, or_in_rad) = model.state();
                let dx = x - goal_x;
                let dy = y - goal_y;
                let dist = (dx * dx + dy * dy).sqrt();
                episode_reward -= dist;
                let best_theta = dy.atan2(dx);
                let curr_theta = or_in_rad % std::f32::consts::PI;
                episode_reward -= (curr_theta - best_theta).abs() * 10.0;
                episode_reward -= w.abs();
            }

            let (x, y, _or_in_rad) = model.state();
            let (v, w) = model.control();
            let dx = x - goal_x;
            let dy = y - goal_y;
            let final_dist = (dx * dx + dy * dy).sqrt();
            if final_dist < 6.0 * Goal::SLACK {
                episode_reward += 100.0
            }
            if final_dist < 4.0 * Goal::SLACK {
                episode_reward += 1000.0
            }
            if final_dist < 2.0 * Goal::SLACK {
                episode_reward += 2000.0;
                if v.abs() < 5.0 {
                    episode_reward += 10000.0;
                }
                if w.abs() < 5.0 {
                    episode_reward += 10000.0;
                }
            }
            total_reward += episode_reward;
        }

        total_reward / num_episodes as f32
    }
}
