use std::collections::{BTreeSet, HashMap};
use std::iter::once;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, StandardNormal};
use ukiyo_tensor::Tensor;

fn main() {
    let file = include_str!("names.txt");
    let mut chars = file.chars().collect::<BTreeSet<_>>();
    chars.remove(&'\n');
    const TERM_TOK: char = '.';
    chars.insert(TERM_TOK);
    let stoi = chars
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect::<HashMap<_, _>>();
    let itos = stoi
        .iter()
        .map(|(&c, &i)| (i, c))
        .collect::<HashMap<_, _>>();

    let mut bigrams = Tensor::<i32>::zeros(vec![chars.len(), chars.len()]);
    file.lines().for_each(|line| {
        let chs = once(TERM_TOK).chain(line.chars()).chain(once(TERM_TOK));
        chs.clone().zip(chs.skip(1)).for_each(|bigram| {
            let ix1 = stoi[&bigram.0];
            let ix2 = stoi[&bigram.1];
            bigrams[&[ix1, ix2]] += 1;
        });
    });
    // model smoothing
    let bigrams_smooth = bigrams.map(|a| a + 1);
    // bigrams 27x27. sum over the rows to get 27x1. then broadcast_op div to get 27x27
    let bigrams_sum_row = bigrams_smooth.reduce(1, |&a, &b| a + b, true);
    let bigrams_prob =
        bigrams_smooth.broadcast_op(&bigrams_sum_row, &[(0, 0), (1, 1)], |&a, &b| {
            a as f32 / b as f32
        });
    let mut rng = StdRng::seed_from_u64(2147483647);
    let mut out = vec![];
    let mut ix = 0;
    (0..5).for_each(|_| {
        loop {
            let row = bigrams_prob.slice(0, ix..=ix);
            let dist = WeightedIndex::new(row.data()).unwrap();
            ix = dist.sample(&mut rng);
            out.push(itos[&ix]);
            if ix == stoi[&TERM_TOK] {
                break;
            }
        }
    });
    println!("{}", out.iter().collect::<String>().replace(".", ".\n"));

    // Evaluation
    let mut log_likelihood = 0.0;
    let mut n = 0;
    file.lines().for_each(|line| {
        let chs = once(TERM_TOK).chain(line.chars()).chain(once(TERM_TOK));
        chs.clone().zip(chs.skip(1)).for_each(|bigram| {
            let ix1 = stoi[&bigram.0];
            let ix2 = stoi[&bigram.1];
            let prob: f32 = bigrams_prob[&[ix1, ix2]];
            let logprob = prob.ln();
            log_likelihood += logprob;
            n += 1;
        });
    });
    let nll = -log_likelihood;
    println!("Normalized Negative Log Likelihood: {}", nll / n as f32);

    // Neural network approach
    let mut xs = vec![];
    let mut ys = vec![];
    file.lines().take(1).for_each(|line| {
        let chs = once(TERM_TOK).chain(line.chars()).chain(once(TERM_TOK));
        chs.clone().zip(chs.skip(1)).for_each(|bigram| {
            let ix1 = stoi[&bigram.0];
            let ix2 = stoi[&bigram.1];
            xs.push(ix1);
            ys.push(ix2);
        });
    });
    let xs_tensor = Tensor::<usize>::new(xs.clone(), vec![5]);
    let ys_tensor = Tensor::<usize>::new(ys.clone(), vec![5]);
    let weights = Tensor::<f32>::rand_with::<StandardNormal>(
        vec![chars.len(), chars.len()],
        &mut rng,
        StandardNormal,
    );

    // forward pass
    let xenc = Tensor::<usize>::one_hot(&xs_tensor, chars.len()).map(|&a| a as f32);
    let intermediate = xenc.broadcast_op(&weights, &[(1, 0)], |&a, &b| a * b);
    let logits = intermediate.reduce(1, |&a, &b| a + b, false);
    // softmax layer: logits -> probs
    let counts = logits.map(|a| a.exp());
    let counts_sum_row = counts.reduce(1, |&a, &b| a + b, true);
    let probs = counts.broadcast_op(&counts_sum_row, &[(0, 0), (1, 1)], |&a, &b| a / b);
    let mut loss = Tensor::<f32>::zeros(vec![ys.len()]);
    (0..xs.len()).zip(ys.iter()).for_each(|(x_idx, &y)| {
        loss[&[x_idx]] = probs[&[x_idx, y]];
    });
    println!("{}", loss);
    let loss_tensor = loss
        .map(|a| a.ln())
        .reduce(0, |&a, &b| a + b, false)
        .map(|a| -a / loss.data().len() as f32);
    println!("{}", loss_tensor);
    // backward pass
    // loss_tensor.zero_grad();
    // loss_tensor.backward();

    // update params
    // weights = weights.broadcast_op(&weights.grad(), &[(0, 0), (1, 1)], |weight, &grad| {
    //     weight.increase_data(-0.1 * grad);
    // });
}
