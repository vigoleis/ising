use crate::lattice::{Lattice, SupportsWolfAlgorithm, K_BOLTZMANN};
use crate::monte_carlo::mc_results::MonteCarloResult;
use crate::monte_carlo::MonteCarloSettings;
use rand::distr::{Bernoulli, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

#[derive(Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ClusterFlipSettings {
    num_warmup_cluster_flips: u32,
    num_cluster_flips_between: u32,
    num_samples_per_temperature: u32,
}

impl ClusterFlipSettings {
    pub fn new(
        num_warmup_cluster_flips: u32,
        num_cluster_flips_between: u32,
        num_samples_per_temperature: u32,
    ) -> Self {
        ClusterFlipSettings {
            num_warmup_cluster_flips,
            num_cluster_flips_between,
            num_samples_per_temperature,
        }
    }
}

fn wolf_probability(interaction: f64, temperature: f64) -> f64 {
    -(-interaction / (temperature * K_BOLTZMANN)).exp_m1()
}

fn flip_cluster<L: Lattice>(current_lattice: L, cluster: Vec<L::Idx>) -> L {
    let mut lattice_out = current_lattice;
    for idx in cluster {
        lattice_out = lattice_out.flip(idx)
    }

    return lattice_out;
}

fn flip_one_wolf_cluster<L: SupportsWolfAlgorithm, R: rand::Rng + ?Sized>(
    lattice: L,
    inclusion_dist: rand::distr::Bernoulli,
    rng: &mut R,
) -> L {
    let initial_index = lattice.draw_random_index(rng);
    let cluster_state = lattice.idx_into(initial_index);

    let mut cluster = vec![initial_index];
    let mut sites_to_check = VecDeque::from([initial_index]);
    let mut indices_checked = HashSet::from([initial_index]);

    while let Some(candidate_for_cluster) = sites_to_check.pop_front() {
        'neighbours: for neighbour_idx in lattice.iter_neighbours(candidate_for_cluster) {
            if indices_checked.contains(&neighbour_idx) {
                continue 'neighbours;
            }

            if lattice.idx_into(neighbour_idx) == cluster_state {
                if inclusion_dist.sample(rng) {
                    cluster.push(neighbour_idx);
                    sites_to_check.push_back(neighbour_idx);
                    indices_checked.insert(neighbour_idx);
                }
            } else {
                // if the site has not the same state as the cluster, it will never be added
                // store it here to avoid having to index into the lattice each time it comes up
                indices_checked.insert(neighbour_idx);
            }
        }
    }
    return flip_cluster(lattice, cluster);
}

fn wolf_monte_carlo_for_temperature<L: SupportsWolfAlgorithm, R: rand::Rng + ?Sized>(
    current_lattice: L,
    temperature: f64,
    settings: &ClusterFlipSettings,
    rng: &mut R,
) -> (L, Vec<f64>, Vec<f64>) {
    let mut current_lattice = current_lattice;
    let inclusion_probability =
        wolf_probability(current_lattice.get_pots_interaction(), temperature);
    let inclusion_dist = Bernoulli::new(inclusion_probability).unwrap();

    for _ in 0..settings.num_warmup_cluster_flips {
        current_lattice = flip_one_wolf_cluster::<L, R>(current_lattice, inclusion_dist, rng);
    }

    let mut sampled_magnetisations = vec![0.; settings.num_samples_per_temperature as usize];
    let mut sampled_energies = vec![0.; settings.num_samples_per_temperature as usize];
    for i in 0..settings.num_samples_per_temperature {
        sampled_magnetisations[i as usize] = current_lattice.get_magnetisation();
        sampled_energies[i as usize] = current_lattice.get_energy_per_site();

        for _ in 0..settings.num_cluster_flips_between {
            current_lattice = flip_one_wolf_cluster::<L, R>(current_lattice, inclusion_dist, rng)
        }
    }

    return (current_lattice, sampled_magnetisations, sampled_energies);
}

pub fn wolf_cluster_monte_carlo<L: SupportsWolfAlgorithm, R: rand::Rng + ?Sized>(
    lattice: L,
    temperatures: Vec<f64>,
    settings: ClusterFlipSettings,
    rng: &mut R,
) -> MonteCarloResult {
    let mut current_lattice = lattice;
    let mut sampled_magnetisations =
        vec![vec![0.; settings.num_samples_per_temperature as usize]; temperatures.len()];
    let mut sampled_energies =
        vec![vec![0.; settings.num_samples_per_temperature as usize]; temperatures.len()];
    for (i, temperature) in temperatures.iter().enumerate() {
        let (tmp_lattice, tmp_magnetisations, tmp_energies) =
            wolf_monte_carlo_for_temperature(current_lattice, *temperature, &settings, rng);
        current_lattice = tmp_lattice;
        sampled_magnetisations[i] = tmp_magnetisations;
        sampled_energies[i] = tmp_energies;
    }

    return MonteCarloResult::new(
        temperatures,
        sampled_magnetisations,
        sampled_energies,
        current_lattice.number_sites(),
        current_lattice.linear_system_size(),
        current_lattice.describe(),
        MonteCarloSettings::ClusterUpdate(settings),
    );
}

#[cfg(test)]
mod test {
    use core::f64;

    use super::*;
    use crate::lattice::square_lattice::SquareLattice;
    use approx;
    use rand::{rngs::SmallRng, SeedableRng};

    #[test]
    fn wolf_probability_some_values() {
        approx::assert_relative_eq!(wolf_probability(0., 1.2), 0., epsilon = f64::EPSILON);
        approx::assert_relative_eq!(wolf_probability(20., 0.), 1., epsilon = f64::EPSILON);
        approx::assert_relative_eq!(
            wolf_probability(f64::INFINITY, 12.2),
            1.,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            wolf_probability(1., 1.),
            0.63212055882855767840447622983854,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            wolf_probability(1., 2.),
            0.39346934028736657639620046500882,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn flip_cluster_function1() {
        let test_lattice = SquareLattice::new_with_ones(5, 1.);
        let mut expected_lattice = SquareLattice::new_with_ones(5, 1.);

        // flip all the spins in the test and expected lattice
        expected_lattice = expected_lattice.flip_all();
        let max_cluster = test_lattice.get_all_indices();
        let mut actual_lattice = flip_cluster(test_lattice, max_cluster);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );

        // flip all the spins in the test and expected lattice once again
        expected_lattice = expected_lattice.flip_all();
        let max_cluster = actual_lattice.get_all_indices();
        actual_lattice = flip_cluster(actual_lattice, max_cluster);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn flip_cluster_function2() {
        let test_lattice = SquareLattice::new_with_ones(5, 1.);

        let mut expected_lattice = SquareLattice::new_with_ones(5, 1.);
        expected_lattice = expected_lattice.flip((0, 0));
        expected_lattice = expected_lattice.flip((0, 1));
        expected_lattice = expected_lattice.flip((1, 0));
        expected_lattice = expected_lattice.flip((1, 1));

        let test_cluster = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let actual_lattice = flip_cluster(test_lattice, test_cluster);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn flip_one_wolf_cluster_always_include1() {
        let mut rng = SmallRng::seed_from_u64(42);
        let test_dist = Bernoulli::new(1.).unwrap();
        let test_lattice = SquareLattice::new_with_ones(5, 1.);
        let mut expected_lattice = SquareLattice::new_with_ones(5, 1.);
        expected_lattice = expected_lattice.flip_all();

        let actual_lattice = flip_one_wolf_cluster(test_lattice, test_dist, &mut rng);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn flip_one_wolf_cluster_always_include2() {
        let mut rng = SmallRng::seed_from_u64(42);
        let test_dist = Bernoulli::new(1.).unwrap();

        let mut test_lattice = SquareLattice::new_with_ones(5, 1.);
        // the first index drawn for this rng seed is (4, 1)
        test_lattice = test_lattice.flip((4, 1));

        let expected_lattice = SquareLattice::new_with_ones(5, 1.);

        let actual_lattice = flip_one_wolf_cluster(test_lattice, test_dist, &mut rng);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn flip_one_wolf_cluster_always_include3() {
        let mut rng = SmallRng::seed_from_u64(42);
        let test_dist = Bernoulli::new(1.).unwrap();

        let mut test_lattice = SquareLattice::new_with_ones(5, 1.);
        test_lattice = test_lattice.flip((2, 0));
        test_lattice = test_lattice.flip((2, 1));
        test_lattice = test_lattice.flip((2, 2));
        test_lattice = test_lattice.flip((2, 3));
        test_lattice = test_lattice.flip((2, 4));
        test_lattice = test_lattice.flip((0, 2));
        test_lattice = test_lattice.flip((1, 2));
        test_lattice = test_lattice.flip((3, 2));
        test_lattice = test_lattice.flip((4, 2));
        test_lattice = test_lattice.flip((0, 0));
        test_lattice = test_lattice.flip((1, 0));
        test_lattice = test_lattice.flip((3, 4));
        test_lattice = test_lattice.flip((4, 4));

        // initial lattice has state
        // -1,  1, -1,  1,  1
        // -1,  1, -1,  1,  1
        // -1, -1, -1, -1, -1
        //  1,  1, -1,  1, -1
        //  1,  1, -1,  1, -1
        // the first index drawn for this rng seed is (4, 1)

        let mut expected_lattice = SquareLattice::new_with_ones(5, 1.);
        expected_lattice = expected_lattice.flip_all();
        expected_lattice = expected_lattice.flip((0, 3));
        expected_lattice = expected_lattice.flip((0, 4));
        expected_lattice = expected_lattice.flip((1, 3));
        expected_lattice = expected_lattice.flip((1, 4));
        expected_lattice = expected_lattice.flip((3, 3));
        expected_lattice = expected_lattice.flip((4, 3));

        // the expected state after flipping a cluster starting at (4, 1) is
        // -1, -1, -1,  1,  1
        // -1, -1, -1,  1,  1
        // -1, -1, -1, -1, -1
        // -1, -1, -1,  1, -1
        // -1, -1, -1,  1, -1

        let actual_lattice = flip_one_wolf_cluster(test_lattice, test_dist, &mut rng);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn flip_one_wolf_cluster_always_include4() {
        let mut rng = SmallRng::seed_from_u64(11);
        let test_dist = Bernoulli::new(1.).unwrap();

        let mut test_lattice = SquareLattice::new_with_ones(5, 1.);
        test_lattice = test_lattice.flip((2, 0));
        test_lattice = test_lattice.flip((2, 1));
        test_lattice = test_lattice.flip((2, 2));
        test_lattice = test_lattice.flip((2, 3));
        test_lattice = test_lattice.flip((2, 4));
        test_lattice = test_lattice.flip((0, 2));
        test_lattice = test_lattice.flip((1, 2));
        test_lattice = test_lattice.flip((3, 2));
        test_lattice = test_lattice.flip((4, 2));
        test_lattice = test_lattice.flip((0, 0));
        test_lattice = test_lattice.flip((1, 0));
        test_lattice = test_lattice.flip((3, 4));
        test_lattice = test_lattice.flip((4, 4));

        // initial lattice has state
        // -1,  1, -1,  1,  1
        // -1,  1, -1,  1,  1
        // -1, -1, -1, -1, -1
        //  1,  1, -1,  1, -1
        //  1,  1, -1,  1, -1
        // the first index drawn for this rng seed is (4, 4)

        let expected_lattice = SquareLattice::new_with_ones(5, 1.);

        // the expected state after flipping a cluster starting at (4, 1) is
        //  1,  1,  1,  1,  1
        //  1,  1,  1,  1,  1
        //  1,  1,  1,  1,  1
        //  1,  1,  1,  1,  1
        //  1,  1,  1,  1,  1

        let actual_lattice = flip_one_wolf_cluster(test_lattice, test_dist, &mut rng);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn flip_one_wolf_cluster_never_include() {
        let mut rng = SmallRng::seed_from_u64(11);
        let test_dist = Bernoulli::new(0.).unwrap();
        let test_lattice = SquareLattice::new_with_ones(5, 1.);
        let mut expected_lattice = SquareLattice::new_with_ones(5, 1.);
        expected_lattice = expected_lattice.flip((4, 4));

        let actual_lattice = flip_one_wolf_cluster(test_lattice, test_dist, &mut rng);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn test_wolf_monte_carlo_for_temperature_0() {
        // setting the temperature to 0 means all candidates will always be included
        // this is therefore euivalent to using the function `mock_acpt_algo_always_include`
        let test_lattice = SquareLattice::new_with_ones(5, 1.);
        let test_settings = ClusterFlipSettings {
            num_warmup_cluster_flips: 1,
            num_cluster_flips_between: 2,
            num_samples_per_temperature: 1,
        };
        let mut rng = SmallRng::seed_from_u64(11);
        let mut benchmark_lattice = SquareLattice::new_with_ones(5, 1.);

        let (test_lattice, sampled_magnetisations, sampled_energies) =
            wolf_monte_carlo_for_temperature(test_lattice, 0., &test_settings, &mut rng);

        assert_eq!(sampled_magnetisations, vec![-1.]);
        assert_eq!(sampled_energies, vec![-4.]);
        benchmark_lattice = benchmark_lattice.flip_all();
        assert!(
            test_lattice.is_state_equal(&benchmark_lattice),
            "Got {test_lattice:?} instead of expected state {benchmark_lattice:?}"
        );

        let (test_lattice, sampled_magnetisations, sampled_energies) =
            wolf_monte_carlo_for_temperature(test_lattice, 0., &test_settings, &mut rng);

        assert_eq!(sampled_magnetisations, vec![1.]);
        assert_eq!(sampled_energies, vec![-4.]);
        benchmark_lattice = benchmark_lattice.flip_all();
        assert!(
            test_lattice.is_state_equal(&benchmark_lattice),
            "Got {test_lattice:?} instead of expected state {benchmark_lattice:?}"
        );
    }

    #[test]
    fn test_wolf_monte_carlo_for_temperature_inf() {
        // setting the temperature to infinity means no candidates are ever included
        // this is therefore euivalent to using the function `mock_acpt_algo_never_include`
        let test_lattice = SquareLattice::new_with_ones(5, 1.);
        let test_settings = ClusterFlipSettings {
            num_warmup_cluster_flips: 0,
            num_cluster_flips_between: 1,
            num_samples_per_temperature: 1,
        };
        let mut rng = SmallRng::seed_from_u64(11);
        let mut benchmark_lattice = SquareLattice::new_with_ones(5, 1.);

        let (test_lattice, _, _) =
            wolf_monte_carlo_for_temperature(test_lattice, f64::INFINITY, &test_settings, &mut rng);

        // the first random starting index is (4, 4)
        benchmark_lattice = benchmark_lattice.flip((4, 4));
        assert!(
            test_lattice.is_state_equal(&benchmark_lattice),
            "Got {test_lattice:?} instead of expected state {benchmark_lattice:?}"
        );

        let (test_lattice, _, _) =
            wolf_monte_carlo_for_temperature(test_lattice, f64::INFINITY, &test_settings, &mut rng);

        // the second random starting index is (0, 0)
        benchmark_lattice = benchmark_lattice.flip((0, 0));
        assert!(
            test_lattice.is_state_equal(&benchmark_lattice),
            "Got {test_lattice:?} instead of expected state {benchmark_lattice:?}"
        );
    }
}
