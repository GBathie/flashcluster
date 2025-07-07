// use std::time::Instant;

// use hierarchical_clustering::{
//     cut_weights::{CwParams, approx::MultiplyMode},
//     points::{PointSet, read_file},
//     spanning_tree::MstParams,
//     ultrametric::{DistortionMode, Ultrametric},
// };
// use log::info;

// pub fn run_method(fname: &str, points: &PointSet, mst: MstParams, cw: CwParams) {
//     let start = Instant::now();
//     let um = Ultrametric::new(points, mst, cw);
//     let dur = start.elapsed().as_secs_f64();

//     let disto = um.distortion(points, DistortionMode::Default);
//     // let disto = -1.0;

//     println!("{fname}; {disto}; {dur:.3}; {:?}; {:?}", mst, cw);
// }

// use clap::Parser;

// #[derive(Debug, Parser)]
// pub struct CliArgs {
//     pub file: String,
//     #[arg(short, long)]
//     pub gamma: f32,
//     #[arg(short, long)]
//     pub alpha: f32,
// }

fn main() {
    //     env_logger::init();

    //     let CliArgs { file, gamma, alpha } = CliArgs::parse();
    //     let points = read_file(&file);
    //     info!("Points shape: {:?}", points.dim());
    //     run_method(
    //         &file,
    //         &points,
    //         MstParams { gamma },
    //         CwParams {
    //             alpha,
    //             mode: MultiplyMode::SquareRoot,
    //         },
    //     );
}
