use std::collections::{BTreeSet, HashMap};
use std::iter::once;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;
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

    let mut b = Tensor::<i32>::zeros(vec![chars.len(), chars.len()]);
    file.lines().for_each(|line| {
        let chs = once(TERM_TOK).chain(line.chars()).chain(once(TERM_TOK));
        chs.clone().zip(chs.skip(1)).for_each(|bigram| {
            b[&[stoi[&bigram.0], stoi[&bigram.1]]] += 1;
        });
    });
    let mut rng = StdRng::seed_from_u64(2147483647);
    let mut out = vec![];
    let mut ix = 0;
    (0..5).for_each(|_| {
        loop {
            let row = b.slice(0, ix..=ix);
            // NOTE: Karpathy casts to f32 and normalizes here to make it clear that we are sampling from a
            // distribution. However, multinomial sampling can be done on integers (i.e., weights) directly, so
            // this cast is unnecessary.
            let dist = WeightedIndex::new(row.data()).unwrap();
            ix = dist.sample(&mut rng);
            out.push(itos[&ix]);
            if ix == stoi[&TERM_TOK] {
                break;
            }
        }
    });
    println!("{}", out.iter().collect::<String>().replace(".", ".\n"));
}
