use ggez::*;
use serde::{Deserialize, Serialize};
mod vis;
use vis::*;

mod reward;
use reward::*;

extern crate wall_e;
use wall_e::ceo::CEO;
use wall_e::fcn::*;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Experiment {
    fcn: FCN,
    ceo: CEO,
    reward: DiffDriveReward,
}

impl Experiment {
    const VIS_WIDTH: f32 = 500.0;
    const VIS_HEIGHT: f32 = 500.0;
}

fn run() -> Experiment {
    let mut fcn = FCN::new(vec![
        (3, Activation::Linear),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (5, Activation::LeakyReLu(0.1)),
        (2, Activation::Linear),
    ]);

    let mut ceo = CEO::default();
    ceo.generations = 1000;
    ceo.batch_size = 100;
    ceo.num_evalation_samples = 6;
    ceo.elite_frac = 0.25;
    ceo.initial_std = 3.0;
    ceo.noise_factor = 3.0;

    let reward = DiffDriveReward::new(
        // (100.0, 100.0),
        // (400.0, 400.0),
        (20.0, 240.0),
        (20.0, 240.0),
        (0.0, 6.28),
        10.0,
        // (250.0, 480.0),
        // (20.0, 250.0),
        (260.0, 480.0),
        (260.0, 480.0),
        500,
    );

    let _th_std = ceo.optimize(&mut fcn, &reward).unwrap();

    let exp = Experiment {
        fcn: fcn,
        ceo: ceo,
        reward: reward,
    };

    exp
}

fn main() {
    use std::env;
    use std::fs::File;
    use std::io::BufReader;

    let args = env::args();
    let exp = if args.len() == 1 {
        // Run
        let exp = run();
        // Save
        let now = chrono::offset::Local::now();
        serde_json::to_writer(
            &File::create(format!("exp{},{}.json", now.date(), now.time())).unwrap(),
            &exp,
        )
        .unwrap();
        exp
    } else {
        if args.len() != 2 {
            panic!("Bad cmd line parameters.");
        }
        // Load from file
        let args = args.collect::<Vec<String>>();
        let file = File::open(&args[1]).unwrap();
        let reader = BufReader::new(file);
        let exp: Experiment = serde_json::from_reader(reader).unwrap();
        exp
    };
    println!("{:?}", exp);
    // Visualize
    let ref mut app = Visualizer::from(exp);
    let mut conf = conf::Conf::new();
    conf.window_mode.width = Experiment::VIS_WIDTH;
    conf.window_mode.height = Experiment::VIS_HEIGHT;
    let (ref mut ctx, ref mut event_loop) = ContextBuilder::new("wall_e", "buggedbit")
        .conf(conf)
        .build()
        .unwrap();
    event::run(ctx, event_loop, app).unwrap();
}
