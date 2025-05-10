use crate::lattice::K_BOLTZMANN;
use crate::monte_carlo::MonteCarloSettings;
use num_traits::abs;
use num_traits::Pow;
use serde::Deserialize;
use serde::Serialize;
use statrs::statistics::Statistics;

use crate::utils::SerdeOnceCell;

fn apply_to_each_element(vec_of_vecs: &Vec<Vec<f64>>, func: impl Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    vec_of_vecs.iter().map(|v| func(v)).collect()
}

fn avg_of_each_element(vec_of_vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    apply_to_each_element(vec_of_vecs, |v| v.mean())
}

fn avg_of_each_abs_element(vec_of_vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    apply_to_each_element(vec_of_vecs, |v| (v.iter().map(|i| abs(*i))).mean())
}

fn var_of_each_element(vec_of_vecs: &Vec<Vec<f64>>) -> Vec<f64> {
    apply_to_each_element(vec_of_vecs, |v| v.variance())
}

fn binder_cumulant(magnestisatoin_sample: &Vec<f64>) -> f64 {
    1. - magnestisatoin_sample.iter().map(|m| m.pow(4)).mean()
        / (3. * magnestisatoin_sample.iter().map(|m| m.pow(2)).mean().pow(2))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonteCarloResult {
    temperatures: Vec<f64>,
    sampled_magnetisations: Vec<Vec<f64>>,
    sampled_energies: Vec<Vec<f64>>,
    num_spins: usize,
    linear_system_size: usize,
    lattice_description: String,
    settings: MonteCarloSettings,
    // the following are caches for values that are calculated lazily
    avg_abs_magnetisations: SerdeOnceCell<Vec<f64>>,
    avg_energies: SerdeOnceCell<Vec<f64>>,
    heat_capacities: SerdeOnceCell<Vec<f64>>,
    magnetic_susceptibilites: SerdeOnceCell<Vec<f64>>,
    binder_cumulant: SerdeOnceCell<Vec<f64>>,
}

impl MonteCarloResult {
    pub fn new(
        temperatures: Vec<f64>,
        sampled_magnetisations: Vec<Vec<f64>>,
        sampled_energies: Vec<Vec<f64>>,
        num_spins: usize,
        linear_system_size: usize,
        lattice_description: String,
        settings: MonteCarloSettings,
    ) -> Self {
        // TODO: handle this better? change the retrun type?
        assert_eq!(temperatures.len(), sampled_magnetisations.len());
        assert_eq!(temperatures.len(), sampled_energies.len());

        MonteCarloResult {
            temperatures,
            sampled_magnetisations,
            sampled_energies,
            num_spins,
            linear_system_size,
            lattice_description,
            settings,
            avg_abs_magnetisations: SerdeOnceCell::new(),
            avg_energies: SerdeOnceCell::new(),
            heat_capacities: SerdeOnceCell::new(),
            magnetic_susceptibilites: SerdeOnceCell::new(),
            binder_cumulant: SerdeOnceCell::new(),
        }
    }

    pub fn get_temperatures(&self) -> &Vec<f64> {
        &self.temperatures
    }

    pub fn get_sampled_magnetisations(&self) -> &Vec<Vec<f64>> {
        &self.sampled_magnetisations
    }

    pub fn get_sampled_energies(&self) -> &Vec<Vec<f64>> {
        &self.sampled_energies
    }

    pub fn get_num_spins(&self) -> usize {
        self.num_spins
    }

    pub fn get_lattice_description(&self) -> &String {
        &self.lattice_description
    }

    pub fn get_settings(&self) -> &MonteCarloSettings {
        &self.settings
    }

    pub fn get_avg_abs_magnetisations(&self) -> &Vec<f64> {
        &self
            .avg_abs_magnetisations
            .get_or_init(|| avg_of_each_abs_element(&self.sampled_magnetisations))
    }

    pub fn get_avg_energies(&self) -> &Vec<f64> {
        self.avg_energies
            .get_or_init(|| avg_of_each_element(&self.sampled_energies))
    }

    fn calc_heat_capacitites(&self) -> Vec<f64> {
        var_of_each_element(&self.sampled_energies)
            .iter()
            .zip(&self.temperatures)
            .map(|(var_e, temperature)| var_e / (temperature.pow(2) * K_BOLTZMANN))
            .collect()
    }

    pub fn get_heat_capacities(&self) -> &Vec<f64> {
        self.heat_capacities
            .get_or_init(|| self.calc_heat_capacitites())
    }

    fn calc_magnetic_susceptibilites(&self) -> Vec<f64> {
        var_of_each_element(&self.sampled_magnetisations)
            .iter()
            .zip(&self.temperatures)
            .map(|(var_m, temperature)| var_m / (temperature * (self.num_spins as f64)))
            .collect()
    }

    pub fn get_magnetic_susceptibilites(&self) -> &Vec<f64> {
        self.magnetic_susceptibilites
            .get_or_init(|| self.calc_magnetic_susceptibilites())
    }

    pub fn get_binder_cumulant(&self) -> &Vec<f64> {
        self.binder_cumulant
            .get_or_init(|| apply_to_each_element(&self.sampled_magnetisations, binder_cumulant))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::monte_carlo::single_spin_flip::SingleSpinFlipSettings;
    use approx;
    use std::iter::zip;

    fn build_test_mc_results() -> MonteCarloResult {
        let test_temperatures = vec![1., 2., 3., 4., 5.];

        let test_magnetisations = vec![
            vec![1., 2., 3.],
            vec![0.5, 0.5, 0.5],
            vec![-1.05, 0., -1.05],
            vec![-2.5, 2., 1.4],
            vec![100., 12., 200.],
        ];

        let test_energies = vec![
            vec![1., 2., 3.],
            vec![0.25, 0.25, 0.25],
            vec![-1.05, 0., -1.05],
            vec![-2.5, 2., 1.4],
            vec![100., 12., 200.],
        ];

        MonteCarloResult::new(
            test_temperatures,
            test_magnetisations,
            test_energies,
            100,
            10,
            String::from("Test Lattice"),
            MonteCarloSettings::SingleSpinflip(SingleSpinFlipSettings::default()),
        )
    }

    #[test]
    fn mc_result_get_avg_magnetisation() {
        let test_result = build_test_mc_results();
        let expected_avg_magnetisations = vec![2., 0.5, 0.7, 5.9 / 3., 104.];
        for (expected, actual) in zip(
            expected_avg_magnetisations,
            test_result.get_avg_abs_magnetisations(),
        ) {
            approx::assert_relative_eq!(expected, *actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn mc_result_get_avg_energies() {
        let test_result = build_test_mc_results();
        let expected_avg_energies = vec![2., 0.25, -0.7, 0.3, 104.];
        for (expected, actual) in zip(expected_avg_energies, test_result.get_avg_energies()) {
            approx::assert_relative_eq!(expected, *actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn mc_result_get_heat_capacities() {
        let test_result = build_test_mc_results();
        let expected_avg_energies = vec![1., 0., 0.04083333333333334, 0.373125, 353.92];
        for (expected, actual) in zip(expected_avg_energies, test_result.get_heat_capacities()) {
            approx::assert_relative_eq!(expected, *actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn mc_result_get_magnetic_susceptibilites() {
        let test_result = build_test_mc_results();
        let expected_avg_energies = vec![
            0.01,
            0.0,
            0.0012250000000000002,
            0.014924999999999999,
            17.696,
        ];
        for (expected, actual) in zip(
            expected_avg_energies,
            test_result.get_magnetic_susceptibilites(),
        ) {
            approx::assert_relative_eq!(expected, *actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn mc_result_get_binder_cumulant() {
        let test_result = build_test_mc_results();
        let expected_binder_cumulant = vec![0.499999, 0.666666, 0.499999, 0.604893, 0.323891];
        for (expected, actual) in zip(expected_binder_cumulant, test_result.get_binder_cumulant()) {
            approx::assert_relative_eq!(expected, *actual, epsilon = 1e-6);
        }
    }
}
