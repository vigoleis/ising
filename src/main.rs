use ising::lattice;
use ising::monte_carlo;
use ising::vis;
use rand;

pub fn main() {
    try_mc_vis();
}

fn try_mc_vis() {
    let test_lattice = lattice::square_lattice::SquareLattice::new(50, 1.);

    let settings = monte_carlo::single_spin_flip::SingleSpinFlipSettings::new(
        100,
        10,
        100,
        monte_carlo::single_spin_flip::SpinFlipAcceptanceAlgorithm::MRT2,
    );

    let mut rng = rand::rng();

    // analytical critical temperature is ca 2.2691853142 (see https://en.wikipedia.org/wiki/Square_lattice_Ising_model)
    let results = monte_carlo::single_spin_flip::single_spin_flip_monte_carlo(
        test_lattice,
        vec![1.75, 2., 2.25, 2.5, 2.75, 3., 3.5],
        settings,
        &mut rng,
    );
    vis::plot_summary(&results, true, None);
    vis::plot_binder_cumulants(&results, true, None);
}

fn try_creutz_vis() {
    let lattice = lattice::square_lattice::SquareLattice::new(30, 1.);

    let settings = monte_carlo::creutz::CreutzSettings::new(-0.9, 0.05, 1000, 20., 1_0000);
    let mut rng = rand::rng();

    let results = monte_carlo::creutz::creutz_algorithm(lattice, settings, &mut rng);

    if !results.get_is_success() {
        println!("Failure");
        return;
    }

    vis::plot_demon_energy_histogram(&results, true, None);
}
