use std::collections::{BTreeSet, HashMap};
use std::iter::once;

use ukiyo_tensor::Tensor;

fn main() {
    const TERM_TOK: char = '.';
    let file = include_str!("names.txt");
    let mut chars = file.chars().collect::<BTreeSet<_>>();
    chars.remove(&'\n');
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

    let mut b = Tensor::<i32>::zeros(&[chars.len(), chars.len()]);
    file.lines().for_each(|line| {
        let chs = once(TERM_TOK).chain(line.chars()).chain(once(TERM_TOK));
        chs.clone().zip(chs.skip(1)).for_each(|bigram| {
            b[&[stoi[&bigram.0], stoi[&bigram.1]]] += 1;
            // b.entry(bigram).and_modify(|e| *e += 1).or_insert(1);
            // println!("{} {}", bigram.0, bigram.1);
        });
    });
    println!("{b}");
}
