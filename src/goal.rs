use ggez::nalgebra::Point2;
use ggez::*;

pub struct Goal {
    x: f32,
    y: f32,
    slack: f32,
}

impl Goal {
    pub fn new(x: f32, y: f32, slack: f32) -> Goal {
        Goal {
            x: x,
            y: y,
            slack: slack,
        }
    }

    pub fn coordinates(&self) -> (f32, f32) {
        (self.x, self.y)
    }

    pub fn draw(&self, ctx: &mut ggez::Context) -> ggez::GameResult {
        let circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            Point2::new(self.x, self.y),
            self.slack,
            0.5,
            graphics::Color::from((0.0, 1.0, 0.0)),
        )?;
        graphics::draw(ctx, &circle, (Point2::new(0.0, 0.0),))
    }
}
