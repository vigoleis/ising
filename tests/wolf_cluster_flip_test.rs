use ising::lattice::square_lattice::SquareLattice;
use ising::monte_carlo;
use rand::{rngs::SmallRng, SeedableRng};
use std::iter::zip;

// NB: this test can run for about 12s
#[test]
fn slow_square_lattice_wolf_mc() {
    let test_lattice = SquareLattice::new(50, 1.);

    let settings = monte_carlo::wolf_cluster_mc::ClusterFlipSettings::new(30, 10, 100);

    let mut test_rng = SmallRng::seed_from_u64(42);

    // analytical critical temperature is ca 2.2691853142 (see https://en.wikipedia.org/wiki/Square_lattice_Ising_model)
    let results = monte_carlo::wolf_cluster_mc::wolf_cluster_monte_carlo(
        test_lattice,
        vec![2.1, 2.2, 2.3, 2.4],
        settings,
        &mut test_rng,
    );

    let expected_agg_magnetisations = [
        0.86759, // finite magnetisation
        0.77791, // finite magnetisation
        // phase transition
        0.46192, // zero magnetisation
        0.19960, // zero magnetisation
    ];

    for (expected, actual) in zip(
        expected_agg_magnetisations,
        results.get_avg_abs_magnetisations(),
    ) {
        approx::assert_abs_diff_eq!(expected, *actual, epsilon = 1e-5);
    }
}
