use super::Experiment;
use ggez::input::keyboard::{KeyCode, KeyMods};
use ggez::nalgebra::Point2;
use ggez::*;
use ndarray::prelude::*;
use wall_e::diff_drive_model::DiffDriveModel;
use wall_e::goal::Goal;

pub struct Visualizer {
    exp: Experiment,
    model: DiffDriveModel,
    goal: Goal,
    time: usize,
    is_paused: bool,
    model_start_bound_rect: graphics::Rect,
    goal_bound_rect: graphics::Rect,
    dt: f32,
}

impl Visualizer {
    pub fn restart(&mut self) {
        // New goal
        let goal = Goal::in_region(
            self.exp.reward.goal_x_bounds(),
            self.exp.reward.goal_y_bounds(),
        );
        // New agent
        let model = DiffDriveModel::spawn_randomly(
            self.exp.reward.start_x_bounds(),
            self.exp.reward.start_y_bounds(),
            self.exp.reward.start_or_bounds(),
            self.exp.reward.radius(),
            goal.coordinates(),
        );
        // Restart
        self.goal = goal;
        self.model = model;
        self.time = 0;
    }
}

impl From<Experiment> for Visualizer {
    fn from(exp: Experiment) -> Visualizer {
        let (xl, xh) = exp.reward.start_x_bounds();
        let (yl, yh) = exp.reward.start_y_bounds();
        let model_start_bound_rect = graphics::Rect::new(xl, yl, xh - xl, yh - yl);
        let (xl, xh) = exp.reward.goal_x_bounds();
        let (yl, yh) = exp.reward.goal_y_bounds();
        let goal_bound_rect = graphics::Rect::new(xl, yl, xh - xl, yh - yl);
        // Sample goal
        let goal = Goal::in_region(exp.reward.goal_x_bounds(), exp.reward.goal_y_bounds());
        // Spawn agent
        let model = DiffDriveModel::spawn_randomly(
            exp.reward.start_x_bounds(),
            exp.reward.start_y_bounds(),
            exp.reward.start_or_bounds(),
            exp.reward.radius(),
            goal.coordinates(),
        );
        Visualizer {
            exp: exp,
            model: model,
            goal: goal,
            time: 0,
            is_paused: false,
            model_start_bound_rect: model_start_bound_rect,
            goal_bound_rect: goal_bound_rect,
            dt: 0.1,
        }
    }
}

impl event::EventHandler for Visualizer {
    fn update(&mut self, _ctx: &mut ggez::Context) -> ggez::GameResult {
        if self.is_paused {
            return Ok(());
        }
        let (x, y, or_in_rad) = self.model.scaled_state();
        let control = self.exp.fcn.at(&arr1(&[x, y, or_in_rad]));
        let (v, w) = (control[[0]], control[[1]]);
        self.model.set_control(v, w);
        self.model.update(self.dt)?;
        self.time += 1;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut ggez::Context) -> ggez::GameResult {
        graphics::clear(ctx, [0.0, 0.0, 0.0, 1.0].into());

        // Draw bounds
        let model_start_bound_rect = graphics::Mesh::new_rectangle(
            ctx,
            graphics::DrawMode::stroke(1.0),
            self.model_start_bound_rect,
            graphics::Color::from((1.0, 1.0, 1.0)),
        )?;
        graphics::draw(ctx, &model_start_bound_rect, (Point2::new(0.0, 0.0),))?;
        let goal_bound_rect = graphics::Mesh::new_rectangle(
            ctx,
            graphics::DrawMode::stroke(1.0),
            self.goal_bound_rect,
            graphics::Color::from((0.0, 1.0, 0.0)),
        )?;
        graphics::draw(ctx, &goal_bound_rect, (Point2::new(0.0, 0.0),))?;
        // Draw model
        self.model.draw(ctx)?;
        // Draw goal
        self.goal.draw(ctx)?;

        let (v, w) = self.model.control();
        graphics::set_window_title(
            ctx,
            &format!(
                "fps={:.2}, time={:.4}, dt={:?}, v={:.2}, w={:.2},",
                timer::fps(ctx),
                self.time,
                self.dt,
                v,
                w,
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
            KeyCode::P => {
                self.is_paused = !self.is_paused;
            }
            KeyCode::R => {
                self.restart();
            }
            KeyCode::PageUp => {
                self.dt += 0.01;
            }
            KeyCode::PageDown => {
                self.dt -= 0.01;
            }
            _ => (),
        }
    }
}
