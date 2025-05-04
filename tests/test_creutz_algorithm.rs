use ising::lattice::square_lattice::SquareLattice;
use ising::monte_carlo::creutz;
use rand::{rngs::SmallRng, SeedableRng};

#[test]
fn slow_square_lattice_creutz() {
    let test_lattice = SquareLattice::new(30, 1.);

    let settings = creutz::CreutzSettings::new(-2., 0.01, 1_000, 20., 1_000);

    let mut test_rng = SmallRng::seed_from_u64(42);

    let results = creutz::creutz_algorithm(test_lattice, settings, &mut test_rng);

    assert!(results.get_is_success(), "Creutz algorithm failed");

    match results.get_temperature() {
        Some(temperature) => approx::assert_abs_diff_eq!(*temperature, 3.31261, epsilon = 1e-5),
        None => panic!("Temperature calculation failed"),
    }
}
