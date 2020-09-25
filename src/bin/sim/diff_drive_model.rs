use ggez::nalgebra::Point2;
use ggez::*;
use trail::Trail;

mod trail {
    use super::*;

    pub struct Trail {
        queue: Vec<Point2<f32>>,
        limit: usize,
    }

    impl Trail {
        pub fn new(limit: usize) -> Trail {
            Trail {
                queue: Vec::with_capacity(limit),
                limit: limit,
            }
        }

        pub fn add(&mut self, x: f32, y: f32) {
            if self.queue.len() > 0 {
                let new_point = Point2::new(x, y);
                if self.queue[self.queue.len() - 1] != new_point {
                    self.queue.push(Point2::new(x, y));
                }
            } else {
                self.queue.push(Point2::new(x, y));
            }
            if self.queue.len() > self.limit {
                self.queue.remove(0);
            }
        }

        pub fn draw(&self, ctx: &mut ggez::Context) -> ggez::GameResult {
            if self.queue.len() > 1 {
                let line = graphics::Mesh::new_line(
                    ctx,
                    &self.queue,
                    2.0,
                    graphics::Color::from((0.0, 1.0, 1.0)),
                )?;
                graphics::draw(ctx, &line, (Point2::new(0.0, 0.0),))?;
            }
            Ok(())
        }
    }
}

pub struct DiffDriveModel {
    x: f32,
    y: f32,
    or_in_rad: f32,
    radius: f32,
    v: f32,
    w: f32,
    trail: Trail,
}

impl DiffDriveModel {
    pub fn new(
        x: f32,
        y: f32,
        or_in_rad: f32,
        radius: f32,
        v: f32,
        w: f32,
        trail_limit: usize,
    ) -> DiffDriveModel {
        let mut trail = Trail::new(trail_limit);
        trail.add(x, y);
        DiffDriveModel {
            x: x,
            y: y,
            or_in_rad: or_in_rad,
            radius: radius,
            v: v,
            w: w,
            trail: trail,
        }
    }

    pub fn update(&mut self, dt: f32) -> ggez::GameResult {
        self.x += self.v * self.or_in_rad.cos() * dt;
        self.y += self.v * self.or_in_rad.sin() * dt;
        self.or_in_rad += self.w * dt;

        Ok(())
    }

    pub fn draw(&mut self, ctx: &mut ggez::Context) -> ggez::GameResult {
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
            graphics::Color::from((1.0, 0.0, 0.0)),
        )?;
        graphics::draw(ctx, &line, (Point2::new(0.0, 0.0),))?;

        self.trail.add(self.x, self.y);
        self.trail.draw(ctx)?;
        Ok(())
    }

    pub fn control(&self) -> (f32, f32) {
        (self.v, self.w)
    }

    pub fn state(&self) -> (f32, f32, f32) {
        (self.x, self.y, self.or_in_rad)
    }

    pub fn stop(&mut self) {
        self.v = 0.0;
        self.w = 0.0;
    }

    pub fn increment_v(&mut self, dv: f32) {
        self.v += dv;
    }

    pub fn increment_w(&mut self, dw: f32) {
        self.w += dw;
    }

    pub fn set_control(&mut self, v: f32, w: f32) {
        self.v = v;
        self.w = w;
    }
}
