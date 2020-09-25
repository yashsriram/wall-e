use ggez::input::keyboard::{KeyCode, KeyMods};
use ggez::*;

mod diff_drive_agent;
use diff_drive_agent::DiffDriveAgent;

#[derive(Default)]
struct App {
    agent: DiffDriveAgent,
}

impl event::EventHandler for App {
    fn update(&mut self, _ctx: &mut ggez::Context) -> ggez::GameResult {
        for _ in 0..10 {
            self.agent.update(0.01)?;
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut ggez::Context) -> ggez::GameResult {
        let (v, w) = self.agent.control();
        graphics::clear(ctx, [0.0, 0.0, 0.0, 1.0].into());
        self.agent.draw(ctx)?;
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
            KeyCode::S => self.agent.stop(),
            KeyCode::Up => self.agent.increment_v(2.0),
            KeyCode::Down => self.agent.increment_v(-2.0),
            KeyCode::Left => self.agent.increment_w(-0.05),
            KeyCode::Right => self.agent.increment_w(0.05),
            _ => (),
        }
    }
}

pub fn main() -> ggez::GameResult {
    let ref mut app = App::default();
    let mut conf = conf::Conf::new();
    conf.window_mode.width = 650.0;
    conf.window_mode.height = 650.0;
    let (ref mut ctx, ref mut event_loop) = ContextBuilder::new("wall_e", "buggedbit")
        .conf(conf)
        .build()?;
    event::run(ctx, event_loop, app)
}
