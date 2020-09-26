use ggez::input::keyboard::{KeyCode, KeyMods};
use ggez::*;

extern crate wall_e;
use wall_e::diff_drive_model::DiffDriveModel;
use wall_e::goal::Goal;

struct App {
    model: DiffDriveModel,
    goal: Goal,
}

impl event::EventHandler for App {
    fn update(&mut self, _ctx: &mut ggez::Context) -> ggez::GameResult {
        self.model.update(0.1)?;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut ggez::Context) -> ggez::GameResult {
        graphics::clear(ctx, [0.0, 0.0, 0.0, 1.0].into());

        self.model.draw(ctx)?;
        self.goal.draw(ctx)?;

        let (v, w) = self.model.control();
        graphics::set_window_title(
            ctx,
            &format!("fps={:.2}, v={:.2}, w={:.2}", timer::fps(ctx), v, w),
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
    let ref mut app = App {
        model: DiffDriveModel::spawn(325.0, 325.0, 0.0, 15.0, 500, 40.0, 5.0),
        goal: Goal::new(600.0, 325.0),
    };
    let mut conf = conf::Conf::new();
    conf.window_mode.width = 650.0;
    conf.window_mode.height = 650.0;
    let (ref mut ctx, ref mut event_loop) = ContextBuilder::new("wall_e", "buggedbit")
        .conf(conf)
        .build()?;
    event::run(ctx, event_loop, app)
}
