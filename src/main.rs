mod engine;
mod nn;

use nn::Mlp;

fn main() {
    let xs: [&[f32]; 4] = [
        &[2.0, 3.0, -1.0],
        &[3.0, -1.0, 0.5],
        &[0.5, 1.0, 1.0],
        &[1.0, 1.0, -1.0],
    ];
    let ys: Vec<f32> = vec![1.0, -1.0, -1.0, 1.0];
    let n = Mlp::new(3, &[4, 4, 1]);
    (0..20).for_each(|i| {
        print!("step {}: ", i);
        n.train(&xs, &ys);
    });
}
