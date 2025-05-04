use std::cell::OnceCell;

use crate::lattice::{Lattice, K_BOLTZMANN};
use crate::utils;
use rand;

#[derive(Debug, Default)]
pub struct CreutzSettings {
    starting_energy: f64,
    starting_energy_eps: f64,
    starting_energy_max_tries: u64,
    max_demon_energy: f64,
    number_iterations: u64,
}

impl CreutzSettings {
    pub fn new(
        starting_energy: f64,
        starting_energy_eps: f64,
        starting_energy_max_tries: u64,
        max_demon_energy: f64,
        number_iterations: u64,
    ) -> Self {
        CreutzSettings {
            starting_energy,
            starting_energy_eps,
            starting_energy_max_tries,
            max_demon_energy,
            number_iterations,
        }
    }

    pub fn get_max_demon_energy(&self) -> f64 {
        self.max_demon_energy
    }
}

#[derive(Debug, Default)]
pub struct CreutzResult {
    sampled_magnetisations: Vec<f64>,
    sampled_energies: Vec<f64>,
    demon_energies: Vec<f64>,
    demon_energy_hist_step: f64,
    lattice_description: String,
    settings: CreutzSettings,
    success: bool, // default value is false
    // the following are caches for values that are calculated lazily
    demon_energy_hist: OnceCell<utils::Histogram>,
    demon_energy_hist_fit: OnceCell<Option<(f64, f64)>>,
    fitted_temperature: OnceCell<Option<f64>>,
}

impl CreutzResult {
    pub fn new(
        sampled_magnetisations: Vec<f64>,
        sampled_energies: Vec<f64>,
        demon_energies: Vec<f64>,
        demon_energy_hist_step: f64,
        lattice_description: String,
        settings: CreutzSettings,
        success: bool,
    ) -> Self {
        CreutzResult {
            sampled_magnetisations,
            sampled_energies,
            demon_energies,
            demon_energy_hist_step,
            lattice_description,
            settings,
            success,
            demon_energy_hist: OnceCell::new(),
            demon_energy_hist_fit: OnceCell::new(),
            fitted_temperature: OnceCell::new(),
        }
    }

    pub fn get_sampled_magnetisations(&self) -> &Vec<f64> {
        &self.sampled_magnetisations
    }

    pub fn get_sampled_energies(&self) -> &Vec<f64> {
        &self.sampled_energies
    }

    pub fn get_demon_energies(&self) -> &Vec<f64> {
        &self.demon_energies
    }

    pub fn get_lattice_description(&self) -> &str {
        &self.lattice_description
    }

    pub fn get_used_settings(&self) -> &CreutzSettings {
        &self.settings
    }

    pub fn get_is_success(&self) -> bool {
        self.success
    }

    fn calc_demon_energy_histogram(&self) -> utils::Histogram {
        utils::histogram(
            &self.demon_energies,
            &utils::linspace_vector_from_step(
                0.,
                self.get_used_settings().get_max_demon_energy(),
                self.demon_energy_hist_step,
            ),
        )
    }

    pub fn get_demon_energy_histogram(&self) -> &utils::Histogram {
        self.demon_energy_hist
            .get_or_init(|| self.calc_demon_energy_histogram())
    }

    fn calc_fit_to_demon_energy_histogram(&self) -> Option<(f64, f64)> {
        let demon_energy_hist = self.get_demon_energy_histogram();
        let log_demon_energy = demon_energy_hist
            .get_counts()
            .into_iter()
            .map(|count| if *count > 0 { (*count as f64).ln() } else { 0. })
            .collect();
        utils::least_squares_lin_fit(demon_energy_hist.get_left_bin_edges(), &log_demon_energy)
    }

    pub fn fit_to_demon_energy_histogram(&self) -> Option<&(f64, f64)> {
        self.demon_energy_hist_fit
            .get_or_init(|| self.calc_fit_to_demon_energy_histogram())
            .as_ref()
    }

    fn calc_fitted_temperature(&self) -> Option<f64> {
        match self.fit_to_demon_energy_histogram() {
            Some((_, slope)) => Some(1. / (K_BOLTZMANN * -slope)),
            None => None,
        }
    }

    pub fn get_temperature(&self) -> Option<&f64> {
        self.fitted_temperature
            .get_or_init(|| self.calc_fitted_temperature())
            .as_ref()
    }
}

fn is_approx_equal(number: f64, other_number: f64, epsilon: f64) -> bool {
    number + epsilon >= other_number && number - epsilon <= other_number
}

fn creutz_acceptence_func(new_demon_energy: f64, max_demon_energy: f64) -> bool {
    new_demon_energy >= 0. && new_demon_energy <= max_demon_energy
}

fn get_to_target_energy<L: Lattice, R: rand::Rng + ?Sized>(
    lattice: L,
    target_energy: f64,
    epsilon: f64,
    max_iters: u64,
    rng: &mut R,
) -> Option<L> {
    let mut current_lattice = lattice;
    for _ in 0..max_iters {
        let current_energy = current_lattice.get_energy_per_site();
        if is_approx_equal(current_energy, target_energy, epsilon) {
            return Some(current_lattice);
        }
        let rand_idx = current_lattice.draw_random_index(rng);
        current_lattice = current_lattice.flip(rand_idx)
    }
    return None;
}

pub fn creutz_algorithm<L: Lattice, R: rand::Rng + ?Sized>(
    lattice: L,
    settings: CreutzSettings,
    rng: &mut R,
) -> CreutzResult {
    let initial_lattice = get_to_target_energy(
        lattice,
        settings.starting_energy,
        settings.starting_energy_eps,
        settings.starting_energy_max_tries,
        rng,
    );

    if initial_lattice.is_none() {
        return CreutzResult::default();
    }

    let mut current_lattice = initial_lattice.unwrap();
    let mut current_demon_energy = 0.;
    let mut demon_energies = vec![0.; settings.number_iterations as usize];
    let mut sampled_magnetisations = vec![0.; settings.number_iterations as usize];
    let mut sampled_energies = vec![0.; settings.number_iterations as usize];

    for i in 0..settings.number_iterations as usize {
        let flip_idx = current_lattice.draw_random_index(rng);
        let delta_energy = current_lattice.calc_delta_energy(flip_idx);
        let new_demon_energy = current_demon_energy - delta_energy;
        if creutz_acceptence_func(new_demon_energy, settings.max_demon_energy) {
            current_lattice = current_lattice.flip(flip_idx);
            current_demon_energy = new_demon_energy;
        }
        demon_energies[i] = current_demon_energy;
        sampled_magnetisations[i] = current_lattice.get_magnetisation();
        sampled_energies[i] = current_lattice.get_energy_per_site();
    }

    CreutzResult::new(
        sampled_magnetisations,
        sampled_energies,
        demon_energies,
        current_lattice.get_energy_step(),
        current_lattice.describe(),
        settings,
        true,
    )
}

#[cfg(test)]
mod test {
    use core::f64;

    use super::*;
    use crate::lattice::square_lattice::SquareLattice;
    use approx;
    use rand::{rngs::SmallRng, SeedableRng};

    #[test]
    fn is_approx_equal_func() {
        assert!(is_approx_equal(1., 1., 0.1));
        assert!(is_approx_equal(100., 99., 1.));
        assert!(is_approx_equal(0., -1., 1.5));
        assert!(!is_approx_equal(2., 1., 0.1));
        assert!(!is_approx_equal(-2., 1., 1.));
        assert!(!is_approx_equal(-2., -1., 0.1));
    }

    #[test]
    fn creutz_acceptence() {
        assert!(creutz_acceptence_func(99., 100.));
        assert!(creutz_acceptence_func(100., 100.));
        assert!(creutz_acceptence_func(1., 2.5));
        assert!(creutz_acceptence_func(0., 0.));
        assert!(!creutz_acceptence_func(101., 100.));
        assert!(!creutz_acceptence_func(12., 11.));
        assert!(!creutz_acceptence_func(100., -1.));
        assert!(!creutz_acceptence_func(11., -12.3));
    }

    #[test]
    fn get_to_target_energy_no_update_needed() {
        let test_lattice = SquareLattice::new_with_ones(3, 1.);
        let mut test_rng = SmallRng::seed_from_u64(11);

        let expected_out_lattice = SquareLattice::new(3, 1.);

        let out_lattice_opt = get_to_target_energy(test_lattice, -4., 0.01, 10, &mut test_rng);
        match out_lattice_opt {
            Some(out_lattice) => {
                assert!(out_lattice.is_state_equal(&expected_out_lattice));
                approx::assert_relative_eq!(
                    out_lattice.get_energy_per_site(),
                    -4.,
                    epsilon = f64::EPSILON
                );
            }
            None => panic!("Getting to target energy failed."),
        }
    }

    #[test]
    fn get_to_target_energy_one_update_needed() {
        let test_lattice = SquareLattice::new_with_ones(3, 1.);
        let mut test_rng = SmallRng::seed_from_u64(11);

        let target_energy = -2.2222; // energy in the test lattice if exactly one site is flipped
        let mut expected_out_lattice = SquareLattice::new(3, 1.);
        // the first index chosen randomly by the selected rng is (2, 2).
        expected_out_lattice = expected_out_lattice.flip((2, 2));

        let out_lattice_opt =
            get_to_target_energy(test_lattice, target_energy, 0.01, 10, &mut test_rng);
        match out_lattice_opt {
            Some(out_lattice) => {
                assert!(out_lattice.is_state_equal(&expected_out_lattice));
                approx::assert_abs_diff_eq!(
                    out_lattice.get_energy_per_site(),
                    target_energy,
                    epsilon = 0.1
                );
            }
            None => panic!("Getting to target energy failed."),
        }
    }

    #[test]
    fn creutz_algorithm_simple() {
        let mut test_lattice = SquareLattice::new_with_ones(3, 1.);
        let test_settings = CreutzSettings::new(-2.222, 0.001, 1, 9., 1);
        let mut test_rng = SmallRng::seed_from_u64(11);
        // the first index chosen randomly by the selected rng is (2, 2).
        // flipping it will decrease the energy of the system, so the filp should be accepted
        test_lattice = test_lattice.flip((2, 2));

        let res = creutz_algorithm(test_lattice, test_settings, &mut test_rng);

        assert!(res.get_is_success());
        assert_eq!(*res.get_demon_energies(), vec![8.]);
        approx::assert_abs_diff_eq!(res.get_sampled_energies()[0], -4., epsilon = 0.001);
        approx::assert_abs_diff_eq!(res.get_sampled_magnetisations()[0], 1., epsilon = 0.001);
    }
}
