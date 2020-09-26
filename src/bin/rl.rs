use ggez::input::keyboard::{KeyCode, KeyMods};
use ggez::*;
use ndarray::prelude::*;

extern crate wall_e;
use wall_e::ceo::CEO;
use wall_e::diff_drive_model::DiffDriveModel;
use wall_e::fcn::*;
use wall_e::goal::Goal;

pub fn reward(fcn: &FCN, params: &Array1<f32>, num_episodes: usize) -> f32 {
    let mut total_reward = 0.0;
    for _ in 0..num_episodes {
        let mut episode_reward = 0.0;
        let mut model = DiffDriveModel::spawn(325.0, 325.0, 0.0, 15.0, 0, 20.0, 2.0);
        let goal = Goal::new(500.0, 500.0, 5.0);
        let (goal_x, goal_y) = goal.coordinates();

        for _ in 0..400 {
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
        let final_dist = dx * dx + dy * dy;
        if final_dist < 30.0 {
            episode_reward += 100.0
        }
        if final_dist < 20.0 {
            episode_reward += 1000.0
        }
        if final_dist < 10.0 && v.abs() < 5.0 {
            episode_reward += 10000.0;
        }
        total_reward += episode_reward;
    }
    total_reward / num_episodes as f32
}

struct App {
    model: DiffDriveModel,
    fcn: FCN,
    goal: Goal,
    time: usize,
    sim_time: usize,
}

impl event::EventHandler for App {
    fn update(&mut self, _ctx: &mut ggez::Context) -> ggez::GameResult {
        let (goal_x, goal_y) = self.goal.coordinates();
        let (x, y, or_in_rad) = self.model.state();
        let control = self.fcn.at(&arr1(&[x, y, or_in_rad, goal_x, goal_y]));
        let (v, w) = (control[[0]], control[[1]]);
        self.model.set_control(v, w);
        self.model.update(0.1)?;

        if self.time > self.sim_time {
            return Err(ggez::GameError::ConfigError("Simulation done!".to_owned()));
        }
        self.time += 1;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut ggez::Context) -> ggez::GameResult {
        graphics::clear(ctx, [0.0, 0.0, 0.0, 1.0].into());

        self.model.draw(ctx)?;
        self.goal.draw(ctx)?;

        let (v, w) = self.model.control();
        graphics::set_window_title(
            ctx,
            &format!(
                "fps={:.2}, v={:.2}, w={:.2}, time={:?}",
                timer::fps(ctx),
                v,
                w,
                self.time
            ),
        );
        graphics::present(ctx)
    }

    fn key_down_event(
        &mut self,
        ctx: &mut Context,
        keycode: KeyCode,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
        match keycode {
            KeyCode::Escape => event::quit(ctx),
            KeyCode::S => self.model.set_control(0.0, 0.0),
            KeyCode::Up => self.model.increment_control(2.0, 0.0),
            KeyCode::Down => self.model.increment_control(-2.0, 0.0),
            KeyCode::Left => self.model.increment_control(0.0, -0.05),
            KeyCode::Right => self.model.increment_control(0.0, 0.05),
            _ => (),
        }
    }
}

fn main() {
    let mut fcn = FCN::new(vec![
        (5, Activation::Linear),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (2, Activation::Linear),
    ]);
    println!("{}", fcn);
    let mut ceo = CEO::default();
    ceo.n_iter = 500;
    ceo.num_evalation_samples = 1;
    println!("{:?}", ceo);
    let _th_std = ceo.optimize(&mut fcn, &reward).unwrap();

    let now = chrono::offset::Local::now();
    use std::fs::File;
    serde_json::to_writer(
        &File::create(format!("{},{}.json", now.date(), now.time())).unwrap(),
        &fcn,
    )
    .unwrap();

    let ref mut app = App {
        model: DiffDriveModel::spawn(325.0, 325.0, 0.0, 15.0, 500, 20.0, 2.0),
        fcn: fcn,
        goal: Goal::new(500.0, 500.0, 5.0),
        time: 0,
        sim_time: 400,
    };
    let mut conf = conf::Conf::new();
    conf.window_mode.width = 650.0;
    conf.window_mode.height = 650.0;
    let (ref mut ctx, ref mut event_loop) = ContextBuilder::new("wall_e", "buggedbit")
        .conf(conf)
        .build()
        .unwrap();
    event::run(ctx, event_loop, app).unwrap();
}
