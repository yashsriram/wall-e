use ggez::nalgebra::Point2;
use ggez::*;
use rand::Rng;

#[derive(Debug)]
pub struct Goal {
    x: f32,
    y: f32,
}

impl Goal {
    pub const SLACK: f32 = 0.02;

    pub fn in_region(x_bounds: (f32, f32), y_bounds: (f32, f32)) -> Goal {
        let mut rng = rand::thread_rng();
        let x = x_bounds.0 + (x_bounds.1 - x_bounds.0) * rng.gen::<f32>();
        let y = y_bounds.0 + (y_bounds.1 - y_bounds.0) * rng.gen::<f32>();
        Goal { x: x, y: y }
    }

    pub fn coordinates(&self) -> (f32, f32) {
        (self.x, self.y)
    }

    pub fn draw(&self, ctx: &mut ggez::Context) -> ggez::GameResult {
        let circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            Point2::new(self.x, self.y),
            Goal::SLACK * 40.0,
            0.5,
            graphics::Color::from((0.0, 1.0, 0.0)),
        )?;
        graphics::draw(ctx, &circle, (Point2::new(0.0, 0.0),))
    }
}
