use super::mc_results;
use super::MonteCarloSettings;
use crate::lattice::{Lattice, K_BOLTZMANN};
use rand;

fn mrt2_prob(delta_energy: f64, temperature: f64) -> f64 {
    f64::min(1.0, (-delta_energy / (K_BOLTZMANN * temperature)).exp())
}

fn mrt2<L: Lattice>(current_lattice: &L, flip_idx: L::Idx, temperature: f64) -> f64 {
    let delta_energy = current_lattice.calc_delta_energy(flip_idx);
    mrt2_prob(delta_energy, temperature)
}

fn glauber_prob(neighbouring_spins: f64, temperature: f64) -> f64 {
    let c = ((2. * neighbouring_spins) / (K_BOLTZMANN * temperature)).exp();
    c / (1. + c)
}

fn glauber_dynamics<L: Lattice>(current_lattice: &L, flip_idx: L::Idx, temperature: f64) -> f64 {
    let neighbouring_spins = current_lattice.sum_neighbouring_spins(flip_idx);
    glauber_prob(neighbouring_spins, temperature)
}

#[derive(Debug, PartialEq, Eq, Default)]
pub enum SpinFlipAcceptanceAlgorithm {
    #[default]
    MRT2,
    Glauber,
}

impl SpinFlipAcceptanceAlgorithm {
    fn get_algorithm_func<L: Lattice>(&self) -> fn(&L, L::Idx, f64) -> f64 {
        match self {
            SpinFlipAcceptanceAlgorithm::MRT2 => mrt2,
            SpinFlipAcceptanceAlgorithm::Glauber => glauber_dynamics,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct SingleSpinFlipSettings {
    num_warmup_sweeps: u32,
    num_sweeps_between: u32,
    num_samples_per_temperature: u32,
    acpt_prob_algo: SpinFlipAcceptanceAlgorithm,
}

impl SingleSpinFlipSettings {
    pub fn new(
        num_warmup_sweeps: u32,
        num_sweeps_between: u32,
        num_samples_per_temperature: u32,
        acpt_prob_algo: SpinFlipAcceptanceAlgorithm,
    ) -> Self {
        SingleSpinFlipSettings {
            num_warmup_sweeps,
            num_sweeps_between,
            num_samples_per_temperature,
            acpt_prob_algo,
        }
    }

    pub fn get_algorithm_func<L: Lattice>(&self) -> fn(&L, L::Idx, f64) -> f64 {
        self.acpt_prob_algo.get_algorithm_func()
    }
}

fn single_spin_flip_sweep<L: Lattice, R: rand::Rng + ?Sized>(
    current_lattice: L,
    temperature: f64,
    acpt_pob_algo: fn(&L, <L as Lattice>::Idx, f64) -> f64,
    rng: &mut R,
) -> L {
    let mut current_lattice = current_lattice;
    for idx in current_lattice.get_all_indices() {
        let acpt_prob = acpt_pob_algo(&current_lattice, idx, temperature);
        if rng.random_bool(acpt_prob) {
            current_lattice = current_lattice.flip(idx);
        }
    }
    return current_lattice;
}

fn single_spin_flip_monte_carlo_for_temperature<L: Lattice, R: rand::Rng + ?Sized>(
    current_lattice: L,
    temperature: f64,
    settings: &SingleSpinFlipSettings,
    rng: &mut R,
) -> (L, Vec<f64>, Vec<f64>) {
    let mut current_lattice = current_lattice;
    let acpt_prob_algo: fn(&L, <L as Lattice>::Idx, f64) -> f64 =
        settings.get_algorithm_func::<L>();

    for _ in 0..settings.num_warmup_sweeps {
        current_lattice =
            single_spin_flip_sweep::<L, R>(current_lattice, temperature, acpt_prob_algo, rng);
    }

    let mut sampled_magnetisations = vec![0.; settings.num_samples_per_temperature as usize];
    let mut sampled_energies = vec![0.; settings.num_samples_per_temperature as usize];
    for i in 0..settings.num_samples_per_temperature {
        sampled_magnetisations[i as usize] = current_lattice.get_magnetisation();
        sampled_energies[i as usize] = current_lattice.get_energy_per_site();

        for _ in 0..settings.num_sweeps_between {
            current_lattice =
                single_spin_flip_sweep::<L, R>(current_lattice, temperature, acpt_prob_algo, rng)
        }
    }

    return (current_lattice, sampled_magnetisations, sampled_energies);
}

pub fn single_spin_flip_monte_carlo<L: Lattice, R: rand::Rng + ?Sized>(
    lattice: L,
    temperatures: Vec<f64>,
    settings: SingleSpinFlipSettings,
    rng: &mut R,
) -> mc_results::MonteCarloResult {
    let mut current_lattice = lattice;
    let mut sampled_magnetisations =
        vec![vec![0.; settings.num_samples_per_temperature as usize]; temperatures.len()];
    let mut sampled_energies =
        vec![vec![0.; settings.num_samples_per_temperature as usize]; temperatures.len()];
    for (i, temperature) in temperatures.iter().enumerate() {
        let (tmp_lattice, tmp_magnetisations, tmp_energies) =
            single_spin_flip_monte_carlo_for_temperature(
                current_lattice,
                *temperature,
                &settings,
                rng,
            );
        current_lattice = tmp_lattice;
        sampled_magnetisations[i] = tmp_magnetisations;
        sampled_energies[i] = tmp_energies;
    }

    return mc_results::MonteCarloResult::new(
        temperatures,
        sampled_magnetisations,
        sampled_energies,
        current_lattice.number_sites(),
        current_lattice.linear_system_size(),
        current_lattice.describe(),
        MonteCarloSettings::SingleSpinflip(settings),
    );
}

#[cfg(test)]
mod test {
    use crate::lattice::square_lattice::SquareLattice;

    use super::*;
    use approx;
    use core::f64;

    #[test]
    fn mrt2_prob_is_capped() {
        assert_eq!(mrt2_prob(1., -1.), 1.);
        assert_eq!(mrt2_prob(-1., 1.), 1.);
        assert_eq!(mrt2_prob(-1000., 0.5), 1.);
    }

    #[test]
    fn mrt2_prob_extreme_values() {
        assert_eq!(mrt2_prob(0., 25.), 1.);
        assert_eq!(mrt2_prob(f64::INFINITY, 25.), 0.);
        assert_eq!(mrt2_prob(-f64::INFINITY, 0.5), 1.);
        assert_eq!(mrt2_prob(11.33, f64::INFINITY), 1.);
    }

    #[test]
    fn mrt2_prob_some_normal_values() {
        approx::assert_relative_eq!(
            mrt2_prob(11., 25.),
            0.64403642108314135853,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            mrt2_prob(2., 1.),
            0.1353352832366126918,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(mrt2_prob(-1., 1.5), 1., epsilon = f64::EPSILON);
    }

    #[test]
    fn glauber_prob_extreme_values() {
        assert_eq!(glauber_prob(0., 25.), 0.5);
        assert_eq!(glauber_prob(100., f64::INFINITY), 0.5);
        approx::assert_relative_eq!(glauber_prob(1000., 25.), 1., epsilon = f64::EPSILON);
        approx::assert_relative_eq!(glauber_prob(-1e4, 3.), 0., epsilon = f64::EPSILON);
    }

    #[test]
    fn glauber_prob_some_normal_values() {
        approx::assert_relative_eq!(
            glauber_prob(4., 3.),
            0.935030830871336,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            glauber_prob(1., 0.5),
            0.9820137900379085,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            glauber_prob(-1., 1.5),
            0.20860852732604496,
            epsilon = f64::EPSILON
        );
        approx::assert_relative_eq!(
            glauber_prob(-4., 4.),
            0.11920292202211755,
            epsilon = f64::EPSILON
        );
    }

    fn mock_acpt_algo_always_flip<L: Lattice>(
        _lattice: &L,
        _flip_idx: L::Idx,
        _temperature: f64,
    ) -> f64 {
        1.
    }

    fn mock_acpt_algo_never_flip<L: Lattice>(
        _lattice: &L,
        _flip_idx: L::Idx,
        _temperature: f64,
    ) -> f64 {
        0.
    }

    #[test]
    fn single_spin_flip_sweep_no_flips() {
        let mut rng = rand::rngs::mock::StepRng::new(1, 1);
        let test_lattice = SquareLattice::new_with_ones(5, 1.);
        let expected_lattice = SquareLattice::new_with_ones(5, 1.);

        let actual_lattice =
            single_spin_flip_sweep(test_lattice, 1., mock_acpt_algo_never_flip, &mut rng);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }

    #[test]
    fn single_spin_flip_sweep_all_flips() {
        let mut rng = rand::rngs::mock::StepRng::new(1, 1);
        let test_lattice = SquareLattice::new_with_ones(5, 1.);
        let mut expected_lattice = SquareLattice::new_with_ones(5, 1.);
        expected_lattice = expected_lattice.flip_all();

        let actual_lattice =
            single_spin_flip_sweep(test_lattice, 1., mock_acpt_algo_always_flip, &mut rng);

        assert!(
            expected_lattice.is_state_equal(&actual_lattice),
            "Got {actual_lattice:?} instead of expected state {expected_lattice:?}"
        );
    }
}
