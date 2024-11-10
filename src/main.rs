use std::fmt::Display;

#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {
    n: usize,
    saba: Vec<(u32, u32)>,
    iwashi: Vec<(u32, u32)>,
}

impl Input {
    const MAP_SIZE: u32 = 100000;

    fn read_input() -> Self {
        input! {
            n: usize,
            saba: [(u32, u32); n],
            iwashi: [(u32, u32); n],
        }

        Self { n, saba, iwashi }
    }
}

struct Output {
    points: Vec<(u32, u32)>,
}

impl Output {
    fn new(points: Vec<(u32, u32)>) -> Self {
        Self { points }
    }
}

impl Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.points.len())?;

        for &(x, y) in self.points.iter() {
            writeln!(f)?;
            write!(f, "{} {}", x, y)?;
        }

        Ok(())
    }
}

#[fastout]
fn main() {
    let input = Input::read_input();
    let output = Output::new(vec![
        (0, 0),
        (0, Input::MAP_SIZE),
        (Input::MAP_SIZE, Input::MAP_SIZE),
        (Input::MAP_SIZE, 0),
    ]);
    println!("{}", output);
}
