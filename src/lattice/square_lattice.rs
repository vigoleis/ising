use crate::lattice::{Lattice, SupportsWolfAlgorithm};
use itertools::iproduct;
use rand::{self, distr::Distribution};

#[derive(Debug)]
pub struct SquareLattice {
    width: usize,
    interaction: f64,
    sights: Vec<Vec<i8>>,
    uniform_dist: rand::distr::Uniform<usize>,
}

impl SquareLattice {
    pub fn new(width: usize, interaction: f64) -> Self {
        Self::new_with_ones(width, interaction)
    }

    pub fn new_with_ones(width: usize, interaction: f64) -> Self {
        SquareLattice {
            width,
            interaction,
            sights: vec![vec![1; width]; width],
            uniform_dist: rand::distr::Uniform::new(0, width).unwrap(),
        }
    }

    fn get_all_neighbour_indices(&self, idx_i: usize, idx_j: usize) -> [(usize, usize); 4] {
        [
            (
                (idx_i as isize - 1).rem_euclid(self.width as isize) as usize,
                idx_j,
            ),
            (
                idx_i,
                (idx_j as isize - 1).rem_euclid(self.width as isize) as usize,
            ),
            ((idx_i + 1).rem_euclid(self.width), idx_j),
            (idx_i, (idx_j + 1).rem_euclid(self.width)),
        ]
    }

    pub fn get_all_neighbour_values(&self, idx_i: usize, idx_j: usize) -> [i8; 4] {
        let mut out = [0i8; 4];
        for (i, neighbour_idx) in self
            .get_all_neighbour_indices(idx_i, idx_j)
            .into_iter()
            .enumerate()
        {
            out[i] = self.sights[neighbour_idx.0][neighbour_idx.1];
        }
        return out;
    }

    pub fn is_state_equal(&self, other: &Self) -> bool {
        self.sights == other.sights
    }

    pub fn flip_all(self) -> Self {
        let mut out_lattice = self;
        for idx in out_lattice.get_all_indices() {
            out_lattice = out_lattice.flip(idx);
        }

        return out_lattice;
    }

    fn get_energy_at_site(&self, idx: <Self as Lattice>::Idx) -> f64 {
        -self.interaction * (self.idx_into(idx) as f64) * self.sum_neighbouring_spins(idx)
    }
}

impl Lattice for SquareLattice {
    type Idx = (usize, usize);

    fn number_sites(&self) -> usize {
        self.width
            .checked_pow(2)
            .expect("Overflow when calculating the number of lattice sites for square lattice.")
    }

    fn linear_system_size(&self) -> usize {
        self.width
    }

    fn sum_neighbouring_spins(&self, flip_idx: Self::Idx) -> f64 {
        self.get_all_neighbour_values(flip_idx.0, flip_idx.1)
            .iter()
            .map(|item| *item as f64)
            .sum::<f64>()
    }

    fn calc_delta_energy(&self, flip_idx: Self::Idx) -> f64 {
        -2. * self.get_energy_at_site(flip_idx)
    }

    fn flip(mut self, flip_idx: Self::Idx) -> Self {
        self.sights[flip_idx.0][flip_idx.1] *= -1;
        self
    }

    fn idx_into(&self, idx: Self::Idx) -> i8 {
        self.sights[idx.0][idx.1]
    }

    fn get_all_indices(&self) -> Vec<Self::Idx> {
        iproduct!(0..self.width, 0..self.width).collect()
    }

    fn draw_random_index<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> <Self as Lattice>::Idx {
        (self.uniform_dist.sample(rng), self.uniform_dist.sample(rng))
    }

    fn get_sum_of_spins(&self) -> i64 {
        self.sights
            .iter()
            .map(|v| v.iter().map(|i| *i as i64).sum::<i64>())
            .sum::<i64>()
    }

    fn get_energy(&self) -> f64 {
        self.get_all_indices()
            .into_iter()
            .map(|idx| self.get_energy_at_site(idx))
            .sum::<f64>()
    }

    fn get_energy_step(&self) -> f64 {
        4. * self.interaction
    }

    fn describe(&self) -> String {
        format!(
            "{}x{} Square Lattice With Nearest Neighbour Interaction {}",
            self.width, self.width, self.interaction,
        )
    }
}

impl SupportsWolfAlgorithm for SquareLattice {
    fn iter_neighbours(&self, index: Self::Idx) -> impl Iterator<Item = <Self as Lattice>::Idx> {
        self.get_all_neighbour_indices(index.0, index.1).into_iter()
    }

    fn get_pots_interaction(&self) -> f64 {
        2. * self.interaction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64;
    use rand::{rngs::SmallRng, SeedableRng};
    use std::collections::HashSet;

    #[test]
    fn square_lattice_number_sights_3x3() {
        let test_lattice = SquareLattice::new(3, 1.5);
        assert_eq!(test_lattice.number_sites(), 9);
    }

    #[test]
    fn square_lattice_number_sights_5x5() {
        let test_lattice = SquareLattice::new(5, 1.5);
        assert_eq!(test_lattice.number_sites(), 25);
    }

    // the test lattice looks like
    //  1, -1,  1
    //  1,  1,  1
    // -1, -1,  1
    fn build_square_test_lattice() -> SquareLattice {
        let mut test_lattice = SquareLattice::new(3, 1.5);

        test_lattice.sights[0][1] = -1;
        test_lattice.sights[2][0] = -1;
        test_lattice.sights[2][1] = -1;

        return test_lattice;
    }

    #[test]
    fn square_lattice_get_all_neighbour_indices() {
        let test_lattice = build_square_test_lattice();

        assert_eq!(
            test_lattice.get_all_neighbour_indices(0, 0),
            [(2, 0), (0, 2), (1, 0), (0, 1)],
            "Neighbours of (0, 0) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_indices(1, 1),
            [(0, 1), (1, 0), (2, 1), (1, 2)],
            "Neighbours of (1, 1) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_indices(1, 2),
            [(0, 2), (1, 1), (2, 2), (1, 0)],
            "Neighbours of (0, 0) not matching."
        );
    }

    #[test]
    fn square_lattice_get_all_neighbour_values() {
        let test_lattice = build_square_test_lattice();

        assert_eq!(
            test_lattice.get_all_neighbour_values(0, 0),
            [-1, 1, 1, -1],
            "Neighbours of (0, 0) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(1, 1),
            [-1, 1, -1, 1],
            "Neighbours of (1, 1) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(2, 2),
            [1, -1, 1, -1],
            "Neighbours of (2, 2) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(0, 2),
            [1, -1, 1, 1],
            "Neighbours of (0, 2) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(1, 2),
            [1, 1, 1, 1],
            "Neighbours of (1, 2) not matching."
        );
    }

    #[test]
    fn square_lattice_get_energy_at_site() {
        let test_lattice = build_square_test_lattice();

        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((0, 0)),
            0.,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((1, 1)),
            0.,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((2, 2)),
            0.,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((0, 2)),
            -3.,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((1, 2)),
            -6.,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn square_lattice_sum_neigbouring_spins() {
        let test_lattice = build_square_test_lattice();

        assert_eq!(
            test_lattice.sum_neighbouring_spins((0, 0)),
            0.,
            "Sum of neighbours of (0, 0) not matching."
        );
        assert_eq!(
            test_lattice.sum_neighbouring_spins((1, 1)),
            0.,
            "Sum of neighbours of (1, 1) not matching."
        );
        assert_eq!(
            test_lattice.sum_neighbouring_spins((2, 2)),
            0.,
            "Sum of neighbours of (2, 2) not matching."
        );
        assert_eq!(
            test_lattice.sum_neighbouring_spins((0, 2)),
            2.,
            "Sum of neighbours of (0, 2) not matching."
        );
        assert_eq!(
            test_lattice.sum_neighbouring_spins((1, 2)),
            4.,
            "Sum of neighbours of (1, 2) not matching."
        );
    }

    #[test]
    fn square_lattice_calc_delta_energy() {
        let test_lattice = build_square_test_lattice();

        assert_eq!(
            test_lattice.calc_delta_energy((0, 0)),
            0.,
            "Energy flipping of (0, 0) not matching."
        );
        assert_eq!(
            test_lattice.calc_delta_energy((1, 1)),
            0.,
            "Energy flipping of (1, 1) not matching."
        );
        assert_eq!(
            test_lattice.calc_delta_energy((2, 2)),
            0.,
            "Energy flipping of (2, 2) not matching."
        );
        assert_eq!(
            test_lattice.calc_delta_energy((0, 2)),
            6.,
            "Energy flipping of (0, 2) not matching."
        );
        assert_eq!(
            test_lattice.calc_delta_energy((1, 2)),
            12.,
            "Energy flipping of (1, 2) not matching."
        );
        assert_eq!(
            test_lattice.calc_delta_energy((0, 1)),
            -6.,
            "Energy flipping of (0, 1) not matching."
        );
    }

    #[test]
    fn square_lattice_flip() {
        let benchmark_lattice = build_square_test_lattice();
        let mut test_lattice = SquareLattice::new(3, -2.1);

        test_lattice = test_lattice.flip((0, 1));
        test_lattice = test_lattice.flip((2, 0));
        test_lattice = test_lattice.flip((2, 1));

        assert_eq!(test_lattice.sights, benchmark_lattice.sights);

        // check that the public comparison method works as well
        assert!(test_lattice.is_state_equal(&benchmark_lattice));
    }

    #[test]
    fn square_lattice_idx_into() {
        let test_lattice = build_square_test_lattice();

        assert_eq!(test_lattice.idx_into((0, 0)), 1i8);
        assert_eq!(test_lattice.idx_into((1, 1)), 1i8);
        assert_eq!(test_lattice.idx_into((2, 2)), 1i8);
        assert_eq!(test_lattice.idx_into((2, 0)), -1i8);
        assert_eq!(test_lattice.idx_into((2, 0)), -1i8);
        assert_eq!(test_lattice.idx_into((2, 1)), -1i8);
    }

    #[test]
    fn square_lattice_get_all_indices_3x3() {
        let test_lattice = SquareLattice::new(3, 12.);
        assert_eq!(
            test_lattice.get_all_indices(),
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2)
            ]
        );

        // make sure all indices can be accessed in the lattice
        for idx in test_lattice.get_all_indices() {
            test_lattice.idx_into(idx);
        }
    }

    #[test]
    fn square_lattice_get_sum_of_spins() {
        let test_lattice = build_square_test_lattice();

        assert_eq!(test_lattice.get_sum_of_spins(), 3);
    }

    #[test]
    fn square_lattice_get_energy() {
        let test_lattice = build_square_test_lattice();

        approx::assert_relative_eq!(test_lattice.get_energy(), -6., epsilon = f64::EPSILON);
    }

    #[test]
    fn square_lattice_draw_random_index() {
        // this method is pretty much trivial. The test is thus only a regression test
        // the goal is to get notified if something is changed accidentally
        let test_lattice = build_square_test_lattice();
        let mut test_rng = SmallRng::seed_from_u64(15);

        let outcome = test_lattice.draw_random_index(&mut test_rng);

        assert_eq!(outcome, (0, 2));
    }

    #[test]
    fn square_lattice_iter_neighbours() {
        let test_lattice = build_square_test_lattice();

        // test neighbours of (1, 0)
        let expected_nns_of_1_0: HashSet<(usize, usize)> =
            HashSet::from([(0, 0), (1, 2), (2, 0), (1, 1)]);
        let actual_nns_of_1_0: Vec<(usize, usize)> = test_lattice.iter_neighbours((1, 0)).collect();

        // assert no value is returned twice
        assert_eq!(
            expected_nns_of_1_0.len(),
            actual_nns_of_1_0.len(),
            "Not as many neighbours of (1, 0) as expected."
        );
        // asser that the values are equal, ignore order in iterator
        assert_eq!(
            expected_nns_of_1_0,
            HashSet::from_iter(actual_nns_of_1_0.into_iter()),
            "Neighbours of (1, 0) not as expected."
        );

        // test neighbours of (2, 2)
        let expected_nns_of_2_2: HashSet<(usize, usize)> =
            HashSet::from([(1, 2), (2, 1), (0, 2), (2, 0)]);
        let actual_nns_of_2_2: Vec<(usize, usize)> = test_lattice.iter_neighbours((2, 2)).collect();

        // assert no value is returned twice
        assert_eq!(
            expected_nns_of_2_2.len(),
            actual_nns_of_2_2.len(),
            "Not as many neighbours of (2, 2) as expected."
        );
        // asser that the values are equal, ignore order in iterator
        assert_eq!(
            expected_nns_of_2_2,
            HashSet::from_iter(actual_nns_of_2_2.into_iter()),
            "Neighbours of (2, 2) not as expected."
        );
    }
}
