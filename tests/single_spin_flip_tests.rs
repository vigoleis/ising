use ising::lattice::square_lattice::SquareLattice;
use ising::monte_carlo;
use rand::{rngs::SmallRng, SeedableRng};
use std::iter::zip;

// NB: this test can run for about 5s
#[test]
fn slow_square_lattice_single_flip_mc() {
    let test_lattice = SquareLattice::new(50, 1.);

    let settings = monte_carlo::single_spin_flip::SingleSpinFlipSettings::new(
        100,
        10,
        100,
        monte_carlo::single_spin_flip::SpinFlipAcceptanceAlgorithm::MRT2,
    );

    let mut test_rng = SmallRng::seed_from_u64(11);

    // analytical critical temperature is ca 2.2691853142 (see https://en.wikipedia.org/wiki/Square_lattice_Ising_model)
    let results = monte_carlo::single_spin_flip::single_spin_flip_monte_carlo(
        test_lattice,
        vec![2.1, 2.2, 2.3, 2.4],
        settings,
        &mut test_rng,
    );
    let expected_agg_magnetisations = [
        0.868776, // finite magnetisation
        0.769656, // finite magnetisation
        // phase transition
        0.51268, // zero magnetisation
        0.20942, // zero magnetisation
    ];

    for (expected, actual) in zip(
        expected_agg_magnetisations,
        results.get_avg_abs_magnetisations(),
    ) {
        approx::assert_abs_diff_eq!(expected, *actual, epsilon = 1e-5);
    }
}
