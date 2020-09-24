use ggez::nalgebra::Point2;
use ggez::*;

pub struct DiffDriveAgent {
    x: f32,
    y: f32,
    or_in_rad: f32,
    radius: f32,
    v: f32,
    w: f32,
}

impl DiffDriveAgent {
    pub fn update(&mut self, dt: f32) -> ggez::GameResult {
        self.x += self.v * self.or_in_rad.cos() * dt;
        self.y += self.v * self.or_in_rad.sin() * dt;
        self.or_in_rad += self.w * dt;
        Ok(())
    }

    pub fn draw(&self, ctx: &mut ggez::Context) -> ggez::GameResult {
        let circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            Point2::new(self.x, self.y),
            self.radius,
            0.5,
            graphics::WHITE,
        )?;
        graphics::draw(ctx, &circle, (Point2::new(0.0, 0.0),))?;

        let line = graphics::Mesh::new_line(
            ctx,
            &[
                Point2::new(self.x, self.y),
                Point2::new(
                    self.x + self.radius * self.or_in_rad.cos(),
                    self.y + self.radius * self.or_in_rad.sin(),
                ),
            ],
            2.0,
            graphics::BLACK,
        )?;
        graphics::draw(ctx, &line, (Point2::new(0.0, 0.0),))
    }

    pub fn v(&self) -> f32 {
        self.v
    }

    pub fn increment_v(&mut self, dv: f32) {
        self.v += dv;
    }

    pub fn w(&self) -> f32 {
        self.w
    }

    pub fn increment_w(&mut self, dw: f32) {
        self.w += dw;
    }

    pub fn stop(&mut self) {
        self.v = 0.0;
        self.w = 0.0;
    }
}

impl Default for DiffDriveAgent {
    fn default() -> DiffDriveAgent {
        DiffDriveAgent {
            x: 325.0,
            y: 325.0,
            or_in_rad: 0.0,
            radius: 20.0,
            v: 0.0,
            w: 0.0,
        }
    }
}
