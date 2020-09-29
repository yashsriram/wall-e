use super::Experiment;
use ggez::input::keyboard::{KeyCode, KeyMods};
use ggez::nalgebra::Point2;
use ggez::*;
use ndarray::prelude::*;
use wall_e::diff_drive_model::DiffDriveModel;
use wall_e::fcn::*;
use wall_e::goal::Goal;

pub struct Visualizer {
    model: DiffDriveModel,
    fcn: FCN,
    goal: Goal,
    time: usize,
    is_paused: bool,
    model_start_bound_rect: graphics::Rect,
    goal_bound_rect: graphics::Rect,
}

impl From<Experiment> for Visualizer {
    fn from(ex: Experiment) -> Visualizer {
        let (xl, xh) = ex.reward.start_x_bounds();
        let (yl, yh) = ex.reward.start_y_bounds();
        let model_start_bound_rect = graphics::Rect::new(xl, yl, xh - xl, yh - yl);
        let (xl, xh) = ex.reward.goal_x_bounds();
        let (yl, yh) = ex.reward.goal_y_bounds();
        let goal_bound_rect = graphics::Rect::new(xl, yl, xh - xl, yh - yl);
        let goal = Goal::in_region(ex.reward.goal_x_bounds(), ex.reward.goal_y_bounds());
        Visualizer {
            model: DiffDriveModel::spawn_randomly(
                ex.reward.start_x_bounds(),
                ex.reward.start_y_bounds(),
                ex.reward.start_or_bounds(),
                ex.reward.radius(),
                goal.coordinates(),
            ),
            fcn: ex.fcn,
            goal: goal,
            time: 0,
            is_paused: false,
            model_start_bound_rect: model_start_bound_rect,
            goal_bound_rect: goal_bound_rect,
        }
    }
}

impl event::EventHandler for Visualizer {
    fn update(&mut self, _ctx: &mut ggez::Context) -> ggez::GameResult {
        if self.is_paused {
            return Ok(());
        }
        let (x, y, or_in_rad) = self.model.scaled_state();
        let control = self.fcn.at(&arr1(&[x, y, or_in_rad]));
        let (v, w) = (control[[0]], control[[1]]);
        self.model.set_control(v, w);
        self.model.update(0.1)?;
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
                "fps={:.2}, time={:?}, v={:.2}, w={:.2},",
                timer::fps(ctx),
                self.time,
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
            _ => (),
        }
    }
}
