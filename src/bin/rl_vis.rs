use ggez::input::keyboard::{KeyCode, KeyMods};
use ggez::*;
use ndarray::prelude::*;
use std::env;
use std::fs::File;
use std::io::BufReader;
use wall_e::fcn::*;

extern crate wall_e;
use wall_e::diff_drive_model::DiffDriveModel;
use wall_e::goal::Goal;

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

pub fn main() -> ggez::GameResult {
    let args = env::args().collect::<Vec<String>>();
    let file = File::open(&args[1]).unwrap();
    let reader = BufReader::new(file);
    let fcn: FCN = serde_json::from_reader(reader).unwrap();

    let ref mut app = App {
        model: DiffDriveModel::spawn(325.0, 325.0, 0.0, 15.0, 500, 20.0, 2.0),
        fcn: fcn,
        goal: Goal::new(600.0, 325.0),
        time: 0,
        sim_time: 400,
    };
    let mut conf = conf::Conf::new();
    conf.window_mode.width = 650.0;
    conf.window_mode.height = 650.0;
    let (ref mut ctx, ref mut event_loop) = ContextBuilder::new("wall_e", "buggedbit")
        .conf(conf)
        .build()?;
    event::run(ctx, event_loop, app)
}
