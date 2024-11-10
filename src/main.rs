use std::fmt::Display;

use annealing::{Annealer, Neighbor, NeighborGenerator, SingleScore};
use grid::{ConnectionChecker, Coord, CoordDiff, Map2d, ADJACENTS};
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

#[fastout]
fn main() {
    let input = Input::read_input();
    let mut map_size = 10;
    let env = Env::new(&input, map_size);
    let state = State::init(&env);
    let neigh_gen = NeighGen;

    const TEMPS: [f64; 5] = [5e2, 5e1, 1e1, 5e0, 1e0];

    // div = 1
    let annealer = Annealer::new(TEMPS[0], TEMPS[1], thread_rng().gen(), 1024);
    let (state, stats) = annealer.run(&env, state, &neigh_gen, 0.2);
    eprintln!("[MAP_SIZE = {}]", map_size);
    eprintln!("{}", stats);
    let (env, state) = split_half(&input, &env, &state);
    map_size *= 2;

    // div = 2
    let annealer = Annealer::new(TEMPS[1], TEMPS[2], thread_rng().gen(), 1024);
    let (state, stats) = annealer.run(&env, state, &neigh_gen, 0.4);
    eprintln!("[MAP_SIZE = {}]", map_size);
    eprintln!("{}", stats);
    let (env, state) = split_half(&input, &env, &state);
    map_size *= 2;

    // div = 4
    let annealer = Annealer::new(TEMPS[2], TEMPS[3], thread_rng().gen(), 1024);
    let (state, stats) = annealer.run(&env, state, &neigh_gen, 0.5);
    eprintln!("[MAP_SIZE = {}]", map_size);
    eprintln!("{}", stats);
    let (env, state) = split_half(&input, &env, &state);
    map_size *= 2;

    // div = 8
    let annealer = Annealer::new(TEMPS[3], TEMPS[4], thread_rng().gen(), 1024);
    let (state, stats) = annealer.run(&env, state, &neigh_gen, 0.8);
    eprintln!("[MAP_SIZE = {}]", map_size);
    eprintln!("{}", stats);
    eprintln!("{} {}", state.score, state.len);

    let output = state.to_output(&env);
    println!("{}", output);
}

fn split_half(input: &Input, env: &Env, state: &State) -> (Env, State) {
    let old_state = state;
    let mut map = Map2d::with_default(env.map_size * 2 + 2);

    for row in 1..=env.map_size {
        for col in 1..=env.map_size {
            let c = Coord::new(row, col);

            if old_state.map[c] {
                map[row * 2 - 1][col * 2 - 1] = true;
                map[row * 2][col * 2 - 1] = true;
                map[row * 2 - 1][col * 2] = true;
                map[row * 2][col * 2] = true;
            }
        }
    }

    let env = Env::new(input, env.map_size * 2);
    let state = State {
        map,
        score: old_state.score,
        len: old_state.len,
    };

    (env, state)
}

#[derive(Debug, Clone)]
struct Input {
    _n: usize,
    saba: Vec<(usize, usize)>,
    iwashi: Vec<(usize, usize)>,
}

impl Input {
    const MAP_SIZE: usize = 100000;
    const LEN_LIMIT: usize = 4 * Self::MAP_SIZE;

    fn read_input() -> Self {
        input! {
            n: usize,
            saba: [(usize, usize); n],
            iwashi: [(usize, usize); n],
        }

        Self {
            _n: n,
            saba,
            iwashi,
        }
    }
}

struct Output {
    points: Vec<(usize, usize)>,
}

impl Output {
    fn new(points: Vec<(usize, usize)>) -> Self {
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

struct Env {
    map_size: usize,
    fish_map: Map2d<i32>,
    coord_prefix_sum: Vec<usize>,
    len_list: Vec<usize>,
}

impl Env {
    fn new(input: &Input, map_size: usize) -> Self {
        // 番兵を置く
        let mut fish_map = Map2d::with_default(map_size + 2);

        for &(x, y) in input.saba.iter() {
            let x = (x * map_size / Input::MAP_SIZE).min(map_size - 1) + 1;
            let y = (y * map_size / Input::MAP_SIZE).min(map_size - 1) + 1;

            fish_map[x][y] += 1;
        }

        for &(x, y) in input.iwashi.iter() {
            let x = (x * map_size / Input::MAP_SIZE).min(map_size - 1) + 1;
            let y = (y * map_size / Input::MAP_SIZE).min(map_size - 1) + 1;

            fish_map[x][y] -= 1;
        }

        let mut coord_prefix_sum = vec![0];

        for i in 0..=map_size {
            coord_prefix_sum.push(Input::MAP_SIZE * i / map_size);
        }

        coord_prefix_sum.push(Input::MAP_SIZE);

        let mut len_list = vec![];

        for i in 0..coord_prefix_sum.len() - 1 {
            len_list.push(coord_prefix_sum[i + 1] - coord_prefix_sum[i]);
        }

        Self {
            map_size,
            fish_map,
            coord_prefix_sum,
            len_list,
        }
    }

    fn dump_map(&self) {
        for col in (1..=self.map_size).rev() {
            for row in 1..=self.map_size {
                eprint!("{:4} ", self.fish_map[row][col]);
            }

            eprintln!();
        }
    }
}

#[derive(Clone)]
struct State {
    map: Map2d<bool>,
    score: i32,
    len: usize,
}

impl State {
    fn init(env: &Env) -> Self {
        let mut map = Map2d::with_default(env.map_size + 2);

        for row in 0..env.map_size {
            for col in 0..env.map_size {
                map[row + 1][col + 1] = true;
            }
        }

        Self {
            map,
            score: 1,
            len: Input::MAP_SIZE * 4,
        }
    }

    fn dump_map(&self, env: &Env) {
        for col in (1..=env.map_size).rev() {
            for row in 1..=env.map_size {
                eprint!("{}", if self.map[row][col] { '#' } else { '.' });
            }

            eprintln!();
        }
    }

    fn to_output(&self, env: &Env) -> Output {
        let mut coord = Coord::new(!0, !0);

        'find_start: for x in 1..=env.map_size {
            for y in 1..=env.map_size {
                let c = Coord::new(x, y);

                if self.map[c] {
                    coord = c;
                    break 'find_start;
                }
            }
        }

        let mut vertexes = vec![
            (
                env.coord_prefix_sum[coord.x() + 1],
                env.coord_prefix_sum[coord.y()],
            ),
            (
                env.coord_prefix_sum[coord.x()],
                env.coord_prefix_sum[coord.y()],
            ),
        ];

        const ADJS: [CoordDiff; 4] = [
            CoordDiff::new(0, 1),
            CoordDiff::new(1, 0),
            CoordDiff::new(0, -1),
            CoordDiff::new(-1, 0),
        ];

        let mut dir = 0;

        while vertexes[0] != *vertexes.last().unwrap() {
            // 左に曲がれる
            let turn_left = (dir + 3) % 4;

            if self.map[coord + ADJS[turn_left]] {
                dir = turn_left;
                coord = coord + ADJS[turn_left];
                continue;
            }

            // 点を追加
            let p = match dir {
                0 => (
                    env.coord_prefix_sum[coord.x()],
                    env.coord_prefix_sum[coord.y() + 1],
                ),
                1 => (
                    env.coord_prefix_sum[coord.x() + 1],
                    env.coord_prefix_sum[coord.y() + 1],
                ),
                2 => (
                    env.coord_prefix_sum[coord.x() + 1],
                    env.coord_prefix_sum[coord.y()],
                ),
                3 => (
                    env.coord_prefix_sum[coord.x()],
                    env.coord_prefix_sum[coord.y()],
                ),
                _ => unreachable!(),
            };

            vertexes.push(p);

            // 直進できる or not
            if self.map[coord + ADJS[dir]] {
                coord += ADJS[dir];
            } else {
                dir = (dir + 1) % 4;
            }
        }

        vertexes.pop();

        Output::new(vertexes)
    }
}

impl annealing::State for State {
    type Env = Env;
    type Score = SingleScore;

    fn score(&self, _env: &Self::Env) -> Self::Score {
        SingleScore(self.score as i64)
    }
}

struct NeighGen;

impl NeighborGenerator for NeighGen {
    type Env = Env;
    type State = State;

    fn generate(
        &self,
        env: &Self::Env,
        state: &Self::State,
        rng: &mut impl Rng,
    ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>> {
        loop {
            let neigh = if rng.gen_bool(0.5) {
                OffNeigh::gen(env, state, rng)
            } else {
                OnNeigh::gen(env, state, rng)
            };

            if let Some(neigh) = neigh {
                return neigh;
            }
        }
    }
}

struct OffNeigh {
    coord: Coord,
    len_diff: isize,
}

impl OffNeigh {
    fn gen(
        env: &Env,
        state: &State,
        rng: &mut impl Rng,
    ) -> Option<Box<dyn Neighbor<Env = Env, State = State>>> {
        let x = rng.gen_range(1..=env.map_size);
        let y = rng.gen_range(1..=env.map_size);
        let c = Coord::new(x, y);

        if !state.map[c] {
            return None;
        }

        // 周囲が全部埋まってたらダメ
        // 連結性がなくなるのもダメ
        let ok = !ADJACENTS.iter().all(|&adj| state.map[c + adj])
            && CONNECTION_CHECKER.can_remove(c, |c| state.map[c]);

        if !ok {
            return None;
        }

        let mut len_diff = 0;

        let down = Coord::new(x - 1, y);
        len_diff += if state.map[down] {
            env.len_list[y] as isize
        } else {
            -(env.len_list[y] as isize)
        };

        let up = Coord::new(x + 1, y);
        len_diff += if state.map[up] {
            env.len_list[y] as isize
        } else {
            -(env.len_list[y] as isize)
        };

        let left = Coord::new(x, y - 1);
        len_diff += if state.map[left] {
            env.len_list[x] as isize
        } else {
            -(env.len_list[x] as isize)
        };

        let right = Coord::new(x, y + 1);
        len_diff += if state.map[right] {
            env.len_list[x] as isize
        } else {
            -(env.len_list[x] as isize)
        };

        if state.len.wrapping_add_signed(len_diff) <= Input::LEN_LIMIT {
            Some(Box::new(OffNeigh { coord: c, len_diff }))
        } else {
            None
        }
    }
}

impl Neighbor for OffNeigh {
    type Env = Env;
    type State = State;

    fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }

    fn postprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
        state.score -= env.fish_map[self.coord];
        state.len = state.len.wrapping_add_signed(self.len_diff);
        state.map[self.coord] = false;
    }

    fn eval(
        &mut self,
        env: &Self::Env,
        state: &Self::State,
        _progress: f64,
        _threshold: f64,
    ) -> Option<<Self::State as annealing::State>::Score> {
        Some(SingleScore((state.score - env.fish_map[self.coord]) as i64))
    }

    fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }
}

struct OnNeigh {
    coord: Coord,
    len_diff: isize,
}

impl OnNeigh {
    fn gen(
        env: &Env,
        state: &State,
        rng: &mut impl Rng,
    ) -> Option<Box<dyn Neighbor<Env = Env, State = State>>> {
        let x = rng.gen_range(1..=env.map_size);
        let y = rng.gen_range(1..=env.map_size);
        let c = Coord::new(x, y);

        if state.map[c] {
            return None;
        }

        // 周囲が全部空だったらダメ
        // 空きマスの連結性がなくなるのもダメ
        let ok = !ADJACENTS.iter().all(|&adj| !state.map[c + adj])
            && CONNECTION_CHECKER.can_remove(c, |c| !state.map[c]);

        if !ok {
            return None;
        }

        let mut len_diff = 0;

        let down = Coord::new(x - 1, y);
        len_diff += if state.map[down] {
            -(env.len_list[y] as isize)
        } else {
            env.len_list[y] as isize
        };

        let up = Coord::new(x + 1, y);
        len_diff += if state.map[up] {
            -(env.len_list[y] as isize)
        } else {
            env.len_list[y] as isize
        };

        let left = Coord::new(x, y - 1);
        len_diff += if state.map[left] {
            -(env.len_list[x] as isize)
        } else {
            env.len_list[x] as isize
        };

        let right = Coord::new(x, y + 1);
        len_diff += if state.map[right] {
            -(env.len_list[x] as isize)
        } else {
            env.len_list[x] as isize
        };

        if state.len.wrapping_add_signed(len_diff) <= Input::LEN_LIMIT {
            Some(Box::new(OnNeigh { coord: c, len_diff }))
        } else {
            None
        }
    }
}

impl Neighbor for OnNeigh {
    type Env = Env;
    type State = State;

    fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }

    fn postprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
        state.score += env.fish_map[self.coord];
        state.len = state.len.wrapping_add_signed(self.len_diff);
        state.map[self.coord] = true;
    }

    fn eval(
        &mut self,
        env: &Self::Env,
        state: &Self::State,
        _progress: f64,
        _threshold: f64,
    ) -> Option<<Self::State as annealing::State>::Score> {
        Some(SingleScore((state.score + env.fish_map[self.coord]) as i64))
    }

    fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }
}

const CONNECTION_CHECKER: ConnectionChecker = ConnectionChecker::new();

#[allow(dead_code)]
mod annealing {
    //! 焼きなましライブラリ

    use itertools::Itertools;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;
    use std::{
        fmt::{Debug, Display},
        time::Instant,
    };

    /// 焼きなましの状態
    pub trait State {
        type Env;
        type Score: Score + Clone + PartialEq + Debug;

        /// 生スコア（大きいほど良い）
        fn score(&self, env: &Self::Env) -> Self::Score;
    }

    pub trait Score {
        /// 焼きなまし用スコア（大きいほど良い）
        /// デフォルトでは生スコアをそのまま返す
        fn annealing_score(&self, _progress: f64) -> f64 {
            self.raw_score() as f64
        }

        /// 生スコア
        fn raw_score(&self) -> i64;
    }

    /// 単一の値からなるスコア
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct SingleScore(pub i64);

    impl Score for SingleScore {
        fn raw_score(&self) -> i64 {
            self.0
        }
    }

    /// 焼きなましの近傍
    ///
    /// * 受理パターンの流れ: `preprocess()` -> `eval()` -> `postprocess()`
    /// * 却下パターンの流れ: `preprocess()` -> `eval()` -> `rollback()`
    pub trait Neighbor {
        type Env;
        type State: State<Env = Self::Env>;

        /// `eval()` 前の変形操作を行う
        fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State);

        /// 変形後の状態の評価を行う
        ///
        /// # Arguments
        ///
        /// * `env` - 環境
        /// * `state` - 状態
        /// * `progress` - 焼きなましの進捗（[0, 1]の範囲をとる）
        /// * `threshold` - 近傍採用の閾値。新しいスコアがこの値を下回る場合はrejectされる
        ///
        /// # Returns
        ///
        /// 現在の状態のスコア。スコアが `threshold` を下回ることが明らかな場合は `None` を返すことで評価の打ち切りを行うことができる。
        ///
        /// 評価の打ち切りについては[焼きなまし法での評価関数の打ち切り](https://qiita.com/not522/items/cd20b87157d15850d31c)を参照。
        fn eval(
            &mut self,
            env: &Self::Env,
            state: &Self::State,
            _progress: f64,
            _threshold: f64,
        ) -> Option<<Self::State as State>::Score> {
            Some(state.score(env))
        }

        /// `eval()` 後の変形操作を行う（2-optの区間reverse処理など）
        fn postprocess(&mut self, _env: &Self::Env, _state: &mut Self::State);

        /// `preprocess()` で変形した `state` をロールバックする
        fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State);
    }

    /// 焼きなましの近傍を生成する構造体
    pub trait NeighborGenerator {
        type Env;
        type State: State;

        /// 近傍を生成する
        fn generate(
            &self,
            env: &Self::Env,
            state: &Self::State,
            rng: &mut impl Rng,
        ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>>;
    }

    /// 焼きなましの統計データ
    #[derive(Debug, Clone, Copy)]
    pub struct AnnealingStatistics {
        all_iter: usize,
        accepted_count: usize,
        updated_count: usize,
        init_score: i64,
        final_score: i64,
    }

    impl AnnealingStatistics {
        fn new(init_score: i64) -> Self {
            Self {
                all_iter: 0,
                accepted_count: 0,
                updated_count: 0,
                init_score,
                final_score: init_score,
            }
        }
    }

    impl Display for AnnealingStatistics {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "===== annealing =====")?;
            writeln!(f, "init score : {}", self.init_score)?;
            writeln!(f, "score      : {}", self.final_score)?;
            writeln!(f, "all iter   : {}", self.all_iter)?;
            writeln!(f, "accepted   : {}", self.accepted_count)?;
            writeln!(f, "updated    : {}", self.updated_count)?;

            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct Annealer {
        /// 開始温度
        start_temp: f64,
        /// 終了温度
        end_temp: f64,
        /// 乱数シード
        seed: u128,
        /// 時間計測を行うインターバル
        clock_interval: usize,
    }

    impl Annealer {
        pub fn new(start_temp: f64, end_temp: f64, seed: u128, clock_interval: usize) -> Self {
            Self {
                start_temp,
                end_temp,
                seed,
                clock_interval,
            }
        }

        pub fn run<E, S: State<Env = E> + Clone, G: NeighborGenerator<Env = E, State = S>>(
            &self,
            env: &E,
            mut state: S,
            neighbor_generator: &G,
            duration_sec: f64,
        ) -> (S, AnnealingStatistics) {
            let mut best_state = state.clone();
            let mut current_score = state.score(&env);
            let mut best_score = current_score.annealing_score(1.0);

            let mut diagnostics = AnnealingStatistics::new(current_score.raw_score());
            let mut rng = Pcg64Mcg::new(self.seed);
            let mut threshold_generator = ThresholdGenerator::new(rng.gen());

            let duration_inv = 1.0 / duration_sec;
            let since = Instant::now();

            let mut progress = 0.0;
            let mut temperature = self.start_temp;

            loop {
                diagnostics.all_iter += 1;

                if diagnostics.all_iter % self.clock_interval == 0 {
                    progress = (Instant::now() - since).as_secs_f64() * duration_inv;
                    temperature = f64::powf(self.start_temp, 1.0 - progress)
                        * f64::powf(self.end_temp, progress);

                    if progress >= 1.0 {
                        break;
                    }
                }

                // 変形
                let mut neighbor = neighbor_generator.generate(env, &state, &mut rng);
                neighbor.preprocess(env, &mut state);

                // スコア計算
                let threshold =
                    threshold_generator.next(current_score.annealing_score(progress), temperature);
                let Some(new_score) = neighbor.eval(env, &state, progress, threshold) else {
                    // 明らかに閾値に届かない場合はreject
                    neighbor.rollback(env, &mut state);
                    debug_assert_eq!(state.score(&env), current_score);
                    continue;
                };

                if new_score.annealing_score(progress) >= threshold {
                    // 解の更新
                    neighbor.postprocess(env, &mut state);
                    debug_assert_eq!(state.score(&env), new_score);

                    current_score = new_score;
                    diagnostics.accepted_count += 1;

                    let new_score = current_score.annealing_score(1.0);

                    if best_score < new_score {
                        best_score = new_score;
                        best_state = state.clone();
                        diagnostics.updated_count += 1;
                    }
                } else {
                    neighbor.rollback(env, &mut state);
                    debug_assert_eq!(state.score(&env), current_score);
                }
            }

            diagnostics.final_score = best_state.score(&env).raw_score();

            (best_state, diagnostics)
        }
    }

    /// 焼きなましにおける評価関数の打ち切り基準となる次の閾値を返す構造体
    ///
    /// 参考: [焼きなまし法での評価関数の打ち切り](https://qiita.com/not522/items/cd20b87157d15850d31c)
    struct ThresholdGenerator {
        iter: usize,
        log_randoms: Vec<f64>,
    }

    impl ThresholdGenerator {
        const LEN: usize = 1 << 16;

        fn new(seed: u128) -> Self {
            let mut rng = Pcg64Mcg::new(seed);
            let log_randoms = (0..Self::LEN)
                .map(|_| rng.gen_range(0.0f64..1.0).ln())
                .collect_vec();

            Self {
                iter: 0,
                log_randoms,
            }
        }

        /// 評価関数の打ち切り基準となる次の閾値を返す
        fn next(&mut self, prev_score: f64, temperature: f64) -> f64 {
            let threshold = prev_score + temperature * self.log_randoms[self.iter % Self::LEN];
            self.iter += 1;
            threshold
        }
    }

    #[cfg(test)]
    mod test {
        use itertools::Itertools;
        use rand::Rng;

        use super::{Annealer, Neighbor, Score};

        #[derive(Debug, Clone)]
        struct Input {
            n: usize,
            distances: Vec<Vec<i32>>,
        }

        impl Input {
            fn gen_testcase() -> Self {
                let n = 4;
                let distances = vec![
                    vec![0, 2, 3, 10],
                    vec![2, 0, 1, 3],
                    vec![3, 1, 0, 2],
                    vec![10, 3, 2, 0],
                ];

                Self { n, distances }
            }
        }

        #[derive(Debug, Clone)]
        struct State {
            order: Vec<usize>,
            dist: i32,
        }

        impl State {
            fn new(input: &Input) -> Self {
                let mut order = (0..input.n).collect_vec();
                order.push(0);
                let dist = order
                    .iter()
                    .tuple_windows()
                    .map(|(&prev, &next)| input.distances[prev][next])
                    .sum();

                Self { order, dist }
            }
        }

        impl super::State for State {
            type Env = Input;
            type Score = Dist;

            fn score(&self, _env: &Self::Env) -> Self::Score {
                Dist(self.dist)
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Dist(i32);

        impl Score for Dist {
            fn annealing_score(&self, _progress: f64) -> f64 {
                // 大きい方が良いとするため符号を反転
                -self.0 as f64
            }

            fn raw_score(&self) -> i64 {
                self.0 as i64
            }
        }

        struct TwoOpt {
            begin: usize,
            end: usize,
            new_dist: Option<i32>,
        }

        impl TwoOpt {
            fn new(begin: usize, end: usize) -> Self {
                Self {
                    begin,
                    end,
                    new_dist: None,
                }
            }
        }

        impl Neighbor for TwoOpt {
            type Env = Input;
            type State = State;

            fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State) {
                // do nothing
            }

            fn eval(
                &mut self,
                env: &Self::Env,
                state: &Self::State,
                _progress: f64,
                _threshold: f64,
            ) -> Option<<Self::State as super::State>::Score> {
                let v0 = state.order[self.begin - 1];
                let v1 = state.order[self.begin];
                let v2 = state.order[self.end - 1];
                let v3 = state.order[self.end];

                let d00 = env.distances[v0][v1];
                let d01 = env.distances[v0][v2];
                let d10 = env.distances[v2][v3];
                let d11 = env.distances[v1][v3];

                let new_dist = state.dist - d00 - d10 + d01 + d11;
                self.new_dist = Some(new_dist);

                Some(Dist(new_dist))
            }

            fn postprocess(&mut self, _env: &Self::Env, state: &mut Self::State) {
                state.order[self.begin..self.end].reverse();
                state.dist = self
                    .new_dist
                    .expect("postprocess()を呼ぶ前にeval()を呼んでください。");
            }

            fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State) {
                // do nothing
            }
        }

        struct NeighborGenerator;

        impl super::NeighborGenerator for NeighborGenerator {
            type Env = Input;
            type State = State;

            fn generate(
                &self,
                _env: &Self::Env,
                state: &Self::State,
                rng: &mut impl Rng,
            ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>> {
                loop {
                    let begin = rng.gen_range(1..state.order.len());
                    let end = rng.gen_range(1..state.order.len());

                    if begin + 2 <= end {
                        return Box::new(TwoOpt::new(begin, end));
                    }
                }
            }
        }

        #[test]
        fn annealing_tsp_test() {
            let input = Input::gen_testcase();
            let state = State::new(&input);
            let annealer = Annealer::new(1e1, 1e-1, 42, 1000);
            let neighbor_generator = NeighborGenerator;

            let (state, diagnostics) = annealer.run(&input, state, &neighbor_generator, 0.1);

            eprintln!("{}", diagnostics);

            eprintln!("score: {}", state.dist);
            eprintln!("state.dist: {:?}", state.order);
            assert_eq!(state.dist, 10);
            assert!(state.order == vec![0, 1, 3, 2, 0] || state.order == vec![0, 2, 3, 1, 0]);
        }
    }
}

#[allow(dead_code)]
mod grid {
    use std::ops::{Add, AddAssign, Index, IndexMut};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Coord {
        row: u8,
        col: u8,
    }

    impl Coord {
        pub const fn new(row: usize, col: usize) -> Self {
            Self {
                row: row as u8,
                col: col as u8,
            }
        }

        pub const fn x(&self) -> usize {
            self.row as usize
        }

        pub const fn y(&self) -> usize {
            self.col as usize
        }

        pub const fn in_map(&self, size: usize) -> bool {
            self.row < size as u8 && self.col < size as u8
        }

        pub const fn to_index(&self, size: usize) -> usize {
            self.row as usize * size + self.col as usize
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: u8, x1: u8) -> usize {
            (x0 as i64 - x1 as i64).abs() as usize
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordDiff {
        dr: i8,
        dc: i8,
    }

    impl CoordDiff {
        pub const fn new(dr: i32, dc: i32) -> Self {
            Self {
                dr: dr as i8,
                dc: dc as i8,
            }
        }

        pub const fn invert(&self) -> Self {
            Self {
                dr: -self.dr,
                dc: -self.dc,
            }
        }

        pub const fn dr(&self) -> i32 {
            self.dr as i32
        }

        pub const fn dc(&self) -> i32 {
            self.dc as i32
        }
    }

    impl Add<CoordDiff> for Coord {
        type Output = Coord;

        fn add(self, rhs: CoordDiff) -> Self::Output {
            Coord {
                row: self.row.wrapping_add(rhs.dr as u8),
                col: self.col.wrapping_add(rhs.dc as u8),
            }
        }
    }

    impl AddAssign<CoordDiff> for Coord {
        fn add_assign(&mut self, rhs: CoordDiff) {
            self.row = self.row.wrapping_add(rhs.dr as u8);
            self.col = self.col.wrapping_add(rhs.dc as u8);
        }
    }

    pub const ADJACENTS: [CoordDiff; 4] = [
        CoordDiff::new(-1, 0),
        CoordDiff::new(0, 1),
        CoordDiff::new(1, 0),
        CoordDiff::new(0, -1),
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        size: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, size: usize) -> Self {
            debug_assert!(size * size == map.len());
            Self { size, map }
        }
    }

    impl<T: Default + Clone> Map2d<T> {
        pub fn with_default(size: usize) -> Self {
            let map = vec![T::default(); size * size];
            Self { size, map }
        }
    }

    impl<T> Index<Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> IndexMut<Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> Index<&Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> IndexMut<&Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.size;
            let end = begin + self.size;
            &self.map[begin..end]
        }
    }

    impl<T> IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.size;
            let end = begin + self.size;
            &mut self.map[begin..end]
        }
    }

    #[derive(Debug, Clone)]
    pub struct ConstMap2d<T, const N: usize> {
        map: Vec<T>,
    }

    impl<T, const N: usize> ConstMap2d<T, N> {
        pub fn new(map: Vec<T>) -> Self {
            assert_eq!(map.len(), N * N);
            Self { map }
        }
    }

    impl<T: Default + Clone, const N: usize> ConstMap2d<T, N> {
        pub fn with_default() -> Self {
            let map = vec![T::default(); N * N];
            Self { map }
        }
    }

    impl<T, const N: usize> Index<Coord> for ConstMap2d<T, N> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> IndexMut<Coord> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> Index<&Coord> for ConstMap2d<T, N> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coord) -> &Self::Output {
            &self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> IndexMut<&Coord> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> Index<usize> for ConstMap2d<T, N> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * N;
            let end = begin + N;
            &self.map[begin..end]
        }
    }

    impl<T, const N: usize> IndexMut<usize> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * N;
            let end = begin + N;
            &mut self.map[begin..end]
        }
    }

    /// 3x3領域のみを見て簡易的に関節点判定を行う構造体
    ///
    /// https://speakerdeck.com/shun_pi/ahc023can-jia-ji-zan-ding-ban-shou-shu-kitu-duo-me?slide=18
    #[derive(Debug, Clone)]
    pub struct ConnectionChecker {
        dict: [bool; Self::DICT_SIZE],
    }

    impl ConnectionChecker {
        const DICT_SIZE: usize = 1 << 9;
        const AREA_SIZE: usize = 3;
        const CENTER_INDEX: usize = 4;

        pub const fn new() -> Self {
            Self {
                dict: Self::gen_dict(),
            }
        }

        /// coordを削除して良いかどうか、3x3領域のみを見て簡易的に判定する
        ///
        /// * `coord` - 削除しようとしている点
        /// * `f` - 各セルに、注目している要素が入っているかどうかを判定する関数
        pub fn can_remove(&self, coord: Coord, f: impl Fn(Coord) -> bool) -> bool {
            assert!(f(coord), "削除しようとしているセルが空です。");

            let mut flag = 0;
            let mut index = 0;

            for dr in -1..=1 {
                for dc in -1..=1 {
                    let diff = CoordDiff::new(dr, dc);
                    let c = coord + diff;
                    flag |= (f(c) as usize) << index;
                    index += 1;
                }
            }

            self.dict[flag]
        }

        const fn gen_dict() -> [bool; Self::DICT_SIZE] {
            let mut dict = [false; Self::DICT_SIZE];
            let mut flag = 0;

            while flag < Self::DICT_SIZE {
                // 3x3マスの中央を除いた後
                let after_remove = flag & !(1 << Self::CENTER_INDEX);
                dict[flag] = Self::is_connected(after_remove as u32);
                flag += 1;
            }

            dict
        }

        const fn is_connected(flag: u32) -> bool {
            const fn dfs(coord: Coord, flag: u32, mut visited: u32) -> u32 {
                visited |= 1 << coord.to_index(ConnectionChecker::AREA_SIZE);
                let mut i = 0;

                while i < ADJACENTS.len() {
                    let adj = ADJACENTS[i];
                    let next = Coord::new(
                        coord.x().wrapping_add_signed(adj.dr() as isize),
                        coord.y().wrapping_add_signed(adj.dc() as isize),
                    );

                    if next.in_map(ConnectionChecker::AREA_SIZE) {
                        let bit = 1 << next.to_index(ConnectionChecker::AREA_SIZE);

                        if (visited & bit) == 0 && (flag & bit) > 0 {
                            visited = dfs(next, flag, visited);
                        }
                    }

                    i += 1;
                }

                visited
            }

            const fn coord(index: usize) -> Coord {
                Coord::new(
                    index / ConnectionChecker::AREA_SIZE,
                    index % ConnectionChecker::AREA_SIZE,
                )
            }

            if flag == 0 {
                return false;
            }

            let first_index = flag.trailing_zeros() as usize;
            let visited = dfs(coord(first_index), flag, 0);

            visited == flag
        }
    }

    #[cfg(test)]
    mod test {
        use super::{ConstMap2d, Coord, CoordDiff, Map2d};

        #[test]
        fn coord_add() {
            let c = Coord::new(2, 4);
            let d = CoordDiff::new(-3, 5);
            let actual = c + d;

            let expected = Coord::new(!0, 9);
            assert_eq!(expected, actual);
        }

        #[test]
        fn coord_add_assign() {
            let mut c = Coord::new(2, 4);
            let d = CoordDiff::new(-3, 5);
            c += d;

            let expected = Coord::new(!0, 9);
            assert_eq!(expected, c);
        }

        #[test]
        fn map_new() {
            let map = Map2d::new(vec![0, 1, 2, 3], 2);
            let actual = map[Coord::new(1, 0)];
            let expected = 2;
            assert_eq!(expected, actual);
        }

        #[test]
        fn map_default() {
            let map = Map2d::with_default(2);
            let actual = map[Coord::new(1, 0)];
            let expected = 0;
            assert_eq!(expected, actual);
        }

        #[test]
        fn const_map_new() {
            let map = ConstMap2d::<_, 2>::new(vec![0, 1, 2, 3]);
            let actual = map[Coord::new(1, 0)];
            let expected = 2;
            assert_eq!(expected, actual);
        }

        #[test]
        fn const_map_default() {
            let map = ConstMap2d::<_, 2>::with_default();
            let actual = map[Coord::new(1, 0)];
            let expected = 0;
            assert_eq!(expected, actual);
        }
    }
}
