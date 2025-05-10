use rand;
use std::fmt::Debug;
use std::hash::Hash;

pub const K_BOLTZMANN: f64 = 1.; // using Planck units

pub mod cubic_lattice;
pub mod square_lattice;
pub mod triangular_lattice;

pub trait Lattice {
    type Idx: Copy + PartialEq + Eq + Hash + Debug;

    fn number_sites(&self) -> usize;

    fn linear_system_size(&self) -> usize;

    fn sum_neighbouring_spins(&self, flip_idx: Self::Idx) -> f64;

    fn calc_delta_energy(&self, flip_idx: Self::Idx) -> f64;

    fn flip(self, flip_idx: Self::Idx) -> Self;

    fn idx_into(&self, idx: Self::Idx) -> i8;

    fn draw_random_index<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> <Self as Lattice>::Idx;

    fn get_all_indices(&self) -> Vec<Self::Idx>;

    fn get_sum_of_spins(&self) -> i64;

    fn get_magnetisation(&self) -> f64 {
        (self.get_sum_of_spins() as f64) / (self.number_sites() as f64)
    }

    fn get_energy(&self) -> f64;

    fn get_energy_per_site(&self) -> f64 {
        self.get_energy() / self.number_sites() as f64
    }

    fn get_energy_step(&self) -> f64;

    fn describe(&self) -> String {
        String::from("Unknown lattice")
    }
}

pub trait SupportsWolfAlgorithm: Lattice {
    fn iter_neighbours(&self, index: Self::Idx) -> impl Iterator<Item = <Self as Lattice>::Idx>;

    fn get_pots_interaction(&self) -> f64;
}
