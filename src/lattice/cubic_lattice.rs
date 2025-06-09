use crate::lattice::Lattice;
use crate::lattice::SupportsWolfAlgorithm;
use itertools::iproduct;
use rand::distr::{Distribution, Uniform};

pub struct CubicLattice {
    width: usize,
    interaction: f64,
    sights: Vec<Vec<Vec<i8>>>,
    uniform_dist: Uniform<usize>,
}

impl CubicLattice {
    pub fn new(width: usize, interaction: f64) -> Self {
        CubicLattice {
            width,
            interaction,
            sights: vec![vec![vec![1; width]; width]; width],
            uniform_dist: Uniform::new(0, width).unwrap(),
        }
    }

    fn get_all_neighbour_indices(
        &self,
        idx_i: usize,
        idx_j: usize,
        idx_k: usize,
    ) -> [(usize, usize, usize); 6] {
        [
            (
                (idx_i as isize - 1).rem_euclid(self.width as isize) as usize,
                idx_j,
                idx_k,
            ),
            (
                idx_i,
                (idx_j as isize - 1).rem_euclid(self.width as isize) as usize,
                idx_k,
            ),
            (
                idx_i,
                idx_j,
                (idx_k as isize - 1).rem_euclid(self.width as isize) as usize,
            ),
            ((idx_i + 1).rem_euclid(self.width), idx_j, idx_k),
            (idx_i, (idx_j + 1).rem_euclid(self.width), idx_k),
            (
                idx_i,
                idx_j,
                (idx_k as isize + 1).rem_euclid(self.width as isize) as usize,
            ),
        ]
    }

    pub fn get_all_neighbour_values(&self, idx_i: usize, idx_j: usize, idx_k: usize) -> [i8; 6] {
        let mut out = [0i8; 6];
        for (i, nn_index) in self
            .get_all_neighbour_indices(idx_i, idx_j, idx_k)
            .into_iter()
            .enumerate()
        {
            out[i] = self.sights[nn_index.0][nn_index.1][nn_index.2];
        }
        return out;
    }

    fn get_energy_at_site(&self, idx: <Self as Lattice>::Idx) -> f64 {
        -self.interaction * (self.idx_into(idx) as f64) * self.sum_neighbouring_spins(idx)
    }
}

impl Lattice for CubicLattice {
    type Idx = (usize, usize, usize);

    fn number_sites(&self) -> usize {
        self.width
            .checked_pow(3)
            .expect("Overflow when calculating the number of lattice sites for cubic lattice.")
    }

    fn linear_system_size(&self) -> usize {
        self.width
    }

    fn sum_neighbouring_spins(&self, flip_idx: Self::Idx) -> f64 {
        self.get_all_neighbour_values(flip_idx.0, flip_idx.1, flip_idx.2)
            .iter()
            .map(|item| *item as f64)
            .sum::<f64>()
    }

    fn calc_delta_energy(&self, flip_idx: Self::Idx) -> f64 {
        -2. * self.get_energy_at_site(flip_idx)
    }

    fn flip(mut self, flip_idx: Self::Idx) -> Self {
        self.sights[flip_idx.0][flip_idx.1][flip_idx.2] *= -1;
        self
    }

    fn idx_into(&self, idx: Self::Idx) -> i8 {
        self.sights[idx.0][idx.1][idx.2]
    }

    fn get_all_indices(&self) -> Vec<Self::Idx> {
        iproduct!(0..self.width, 0..self.width, 0..self.width).collect()
    }

    fn draw_random_index<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> <Self as Lattice>::Idx {
        (
            self.uniform_dist.sample(rng),
            self.uniform_dist.sample(rng),
            self.uniform_dist.sample(rng),
        )
    }

    fn get_sum_of_spins(&self) -> i64 {
        self.sights
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| v.iter().map(|i| *i as i64).sum::<i64>())
                    .sum::<i64>()
            })
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
            "{}x{}x{} Cubic Lattice With Nearest Neighbour Interaction {}",
            self.width, self.width, self.width, self.interaction
        )
    }
}

impl SupportsWolfAlgorithm for CubicLattice {
    fn iter_neighbours(&self, index: Self::Idx) -> impl Iterator<Item = <Self as Lattice>::Idx> {
        self.get_all_neighbour_indices(index.0, index.1, index.2)
            .into_iter()
    }

    fn get_potts_interaction(&self) -> f64 {
        2. * self.interaction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx;
    use core::f64;
    use std::collections::HashSet;

    #[test]
    fn cubic_lattice_number_sights_3x3x3() {
        let test_lattice = CubicLattice::new(3, -0.2);
        assert_eq!(test_lattice.number_sites(), 27);
    }

    #[test]
    fn cubic_lattice_number_sights_5x5() {
        let test_lattice = CubicLattice::new(5, -0.2);
        assert_eq!(test_lattice.number_sites(), 125);
    }

    // the test lattice looks like
    //  1, -1,  1
    //  1,  1,  1
    // -1, -1,  1

    //  1,  1,  1
    //  1, -1,  1
    //  1,  1,  1

    //  1,  1,  1
    //  1,  1, -1
    // -1,  1,  1
    fn build_cubic_test_lattice() -> CubicLattice {
        let mut test_lattice = CubicLattice::new(3, -3.2);

        test_lattice.sights[0][1][0] = -1;
        test_lattice.sights[2][0][0] = -1;
        test_lattice.sights[2][1][0] = -1;
        test_lattice.sights[1][1][1] = -1;
        test_lattice.sights[2][0][2] = -1;
        test_lattice.sights[1][2][2] = -1;

        return test_lattice;
    }

    #[test]
    fn cubic_lattice_get_all_neighbour_indices() {
        let test_lattice = build_cubic_test_lattice();

        assert_eq!(
            test_lattice.get_all_neighbour_indices(0, 0, 0),
            [
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1)
            ],
            "Neighbours of (0, 0, 0) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_indices(1, 1, 1),
            [
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 0),
                (2, 1, 1),
                (1, 2, 1),
                (1, 1, 2)
            ],
            "Neighbours of (1, 1, 1) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_indices(2, 2, 2),
            [
                (1, 2, 2),
                (2, 1, 2),
                (2, 2, 1),
                (0, 2, 2),
                (2, 0, 2),
                (2, 2, 0)
            ],
            "Neighbours of (2, 2, 2) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_indices(2, 0, 1),
            [
                (1, 0, 1),
                (2, 2, 1),
                (2, 0, 0),
                (0, 0, 1),
                (2, 1, 1),
                (2, 0, 2)
            ],
            "Neighbours of (2, 0, 1) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_indices(1, 2, 2),
            [
                (0, 2, 2),
                (1, 1, 2),
                (1, 2, 1),
                (2, 2, 2),
                (1, 0, 2),
                (1, 2, 0)
            ],
            "Neighbours of (1, 2, 2) not matching."
        );
    }

    #[test]
    fn cubic_lattice_get_all_neighbour_values() {
        let test_lattice = build_cubic_test_lattice();

        assert_eq!(
            test_lattice.get_all_neighbour_values(0, 0, 0),
            [-1, 1, 1, 1, -1, 1],
            "Neighbours of (0, 0, 0) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(1, 1, 1),
            [1, 1, 1, 1, 1, 1],
            "Neighbours of (1, 1, 1) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(2, 2, 2),
            [-1, 1, 1, 1, -1, 1],
            "Neighbours of (2, 2, 2) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(2, 0, 1),
            [1, 1, -1, 1, 1, -1],
            "Neighbours of (2, 0, 1) not matching."
        );
        assert_eq!(
            test_lattice.get_all_neighbour_values(1, 1, 2),
            [1, 1, -1, 1, -1, 1],
            "Neighbours of (1, 2, 2) not matching."
        );
    }

    #[test]
    fn cubic_lattice_sum_neighbouring_spins() {
        let test_lattice = build_cubic_test_lattice();

        assert_eq!(
            test_lattice.sum_neighbouring_spins((0, 0, 0)),
            2.,
            "Sum of neighbours of (0, 0, 0) not matching."
        );

        assert_eq!(
            test_lattice.sum_neighbouring_spins((1, 1, 1)),
            6.,
            "Sum of neighbours of (1, 1, 1) not matching."
        );

        assert_eq!(
            test_lattice.sum_neighbouring_spins((2, 2, 2)),
            2.,
            "Sum of neighbours of (2, 2, 2) not matching."
        );

        assert_eq!(
            test_lattice.sum_neighbouring_spins((2, 0, 1)),
            2.,
            "Sum of neighbours of (2, 0, 1) not matching."
        );

        assert_eq!(
            test_lattice.sum_neighbouring_spins((1, 1, 2)),
            2.,
            "Sum of neighbours of (1, 1, 2) not matching."
        );
    }

    #[test]
    fn cubic_lattice_get_energy_at_site() {
        let test_lattice = build_cubic_test_lattice();

        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((0, 0, 0)),
            6.4,
            epsilon = f64::EPSILON
        );

        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((1, 1, 1)),
            -19.2,
            epsilon = f64::EPSILON
        );

        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((2, 2, 2)),
            6.4,
            epsilon = f64::EPSILON
        );

        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((2, 0, 1)),
            6.4,
            epsilon = f64::EPSILON
        );

        approx::assert_relative_eq!(
            test_lattice.get_energy_at_site((1, 1, 2)),
            6.4,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn cubic_lattice_calc_delta_energy() {
        let test_lattice = build_cubic_test_lattice();

        approx::assert_relative_eq!(
            test_lattice.calc_delta_energy((0, 0, 0)),
            -12.8,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.calc_delta_energy((1, 1, 1)),
            38.4,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.calc_delta_energy((2, 2, 2)),
            -12.8,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.calc_delta_energy((2, 0, 1)),
            -12.8,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            test_lattice.calc_delta_energy((1, 1, 2)),
            -12.8,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn cubic_lattice_flip() {
        let benchmark_lattice = build_cubic_test_lattice();
        let mut test_lattice = CubicLattice::new(3, -2.1);

        test_lattice = test_lattice.flip((0, 1, 0));
        test_lattice = test_lattice.flip((2, 0, 0));
        test_lattice = test_lattice.flip((2, 1, 0));
        test_lattice = test_lattice.flip((1, 1, 1));
        test_lattice = test_lattice.flip((2, 0, 2));
        test_lattice = test_lattice.flip((1, 2, 2));

        assert_eq!(test_lattice.sights, benchmark_lattice.sights);
    }

    #[test]
    fn cubic_lattice_get_all_indices_2x2() {
        let test_lattice = CubicLattice::new(2, 0.33);
        assert_eq!(
            test_lattice.get_all_indices(),
            vec![
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]
        );

        // make sure all indices can be accessed in the lattice
        for idx in test_lattice.get_all_indices() {
            test_lattice.idx_into(idx);
        }
    }

    #[test]
    fn cubic_latticeget_sum_of_spins() {
        let test_lattice = build_cubic_test_lattice();

        assert_eq!(test_lattice.get_sum_of_spins(), 15);
    }

    #[test]
    fn cubic_lattice_get_energy() {
        let test_lattice = build_cubic_test_lattice();

        approx::assert_relative_eq!(test_lattice.get_energy(), 134.4, epsilon = f64::EPSILON);
    }

    #[test]
    fn cubic_lattice_iter_all_neighbours() {
        let test_lattice = build_cubic_test_lattice();
        // test neighbours of (1, 0, 0)
        let expected_nns_of_1_0_0: HashSet<(usize, usize, usize)> = HashSet::from([
            (0, 0, 0),
            (1, 2, 0),
            (1, 0, 2),
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
        ]);
        let actual_nns_of_1_0_0: Vec<(usize, usize, usize)> =
            test_lattice.iter_neighbours((1, 0, 0)).collect();

        // assert no value is returned twice
        assert_eq!(
            expected_nns_of_1_0_0.len(),
            actual_nns_of_1_0_0.len(),
            "Not as many neighbours of (1, 0, 0) as expected."
        );
        // asser that the values are equal, ignore order in iterator
        assert_eq!(
            expected_nns_of_1_0_0,
            HashSet::from_iter(actual_nns_of_1_0_0.into_iter()),
            "Neighbours of (1, 0, 0) not as expected."
        );

        // test neighbours of (2, 2, 1)
        let expected_nns_of_2_2_1: HashSet<(usize, usize, usize)> = HashSet::from([
            (1, 2, 1),
            (2, 1, 1),
            (2, 2, 0),
            (0, 2, 1),
            (2, 0, 1),
            (2, 2, 2),
        ]);
        let actual_nns_of_2_2_1: Vec<(usize, usize, usize)> =
            test_lattice.iter_neighbours((2, 2, 1)).collect();

        // assert no value is returned twice
        assert_eq!(
            expected_nns_of_2_2_1.len(),
            actual_nns_of_2_2_1.len(),
            "Not as many neighbours of (2, 2, 1) as expected."
        );
        // asser that the values are equal, ignore order in iterator
        assert_eq!(
            expected_nns_of_2_2_1,
            HashSet::from_iter(actual_nns_of_2_2_1.into_iter()),
            "Neighbours of (2, 2, 1) not as expected."
        );
    }
}
