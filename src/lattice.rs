use rand;
use std::fmt::Debug;
use std::hash::Hash;

pub const K_BOLTZMANN: f64 = 1.; // using Planck units

pub mod cubic_lattice;
pub mod square_lattice;
pub mod triangular_lattice;

/// This trait defines a geometry on which we can perform monte carlo updates. It also holds
/// the current state of the system that is altered during the simulations.
/// It contains a collection of sites (or nodes in a graph) with each having a "spin". The sites interact
/// with each other according to some rule. For example, in the simple Ising case the sites form a lattice
/// and each site interacts with its neighbours depening on if their "spins" are aligned .
pub trait Lattice {
    /// This type defines the index through which the sites in the lattice can be accessed.
    /// For example, in a 1D chain of spins the index could just be an integer numbering the sites.
    type Idx: Copy + PartialEq + Eq + Hash + Debug;

    /// Return the total number of sites in the lattice.
    /// Many theoretical models assume infinite lattices but we have to make due with finite ones here.
    fn number_sites(&self) -> usize;

    fn linear_system_size(&self) -> usize;

    /// Sum the "spins" of the nearest neighbours of a given site. The site is defined by its index.
    /// This funcion basically defines the geometry of the graph we work on.
    /// For example, in a square lattice with nearest neighbour interaction only, there will always be 4 neighbouring spins.
    fn sum_neighbouring_spins(&self, flip_idx: Self::Idx) -> f64;

    /// Calculate the change in total energy of the system if the spin at `flip_idx` is flipped.
    /// Note: there are models where flipping is more complex (eg because we need a reflection plaine
    ///  like in the Heisenberg model). This case is not covered (yet).
    fn calc_delta_energy(&self, flip_idx: Self::Idx) -> f64;

    /// Modify the state of the lattice by flipping the spin at the given index.
    /// Return the new lattice.
    fn flip(self, flip_idx: Self::Idx) -> Self;

    /// Return the current value of the spin at the given index.
    /// Note: as of now, we only handle models where the state of a site can be described by a small integer.
    fn idx_into(&self, idx: Self::Idx) -> i8;

    /// Select a valid index of the lattice at random using the given rng.
    fn draw_random_index<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> <Self as Lattice>::Idx;

    /// Collect all valid indices of the lattice. They then allow to iterate through the lattice sites without having
    /// to know how they are represented internally.
    fn get_all_indices(&self) -> Vec<Self::Idx>;

    /// Calculate the sum of all spins in the lattice.
    fn get_sum_of_spins(&self) -> i64;

    /// Calculate the average (ie per site) magnetisation in the current lattice state.
    fn get_magnetisation(&self) -> f64 {
        (self.get_sum_of_spins() as f64) / (self.number_sites() as f64)
    }

    /// Calculate the total energy of the system in the current lattice state.
    fn get_energy(&self) -> f64;

    /// Calculate the avergae energy (ie per site) in the current lattice state.
    fn get_energy_per_site(&self) -> f64 {
        self.get_energy() / self.number_sites() as f64
    }

    /// Get the minimal increase in energy that can occure when flipping a spin in the lattice.
    fn get_energy_step(&self) -> f64;

    /// Give a human readable description of this lattice for displaying in plots etc.
    fn describe(&self) -> String {
        String::from("Unknown lattice")
    }
}

/// This trait is necessary if the lattice should be used with the Wolf cluster update algorithm.
/// At the moment only models that can be re-formulated as Potts models are supported, see https://en.wikipedia.org/wiki/Potts_model#Standard_Potts_model.
pub trait SupportsWolfAlgorithm: Lattice {
    /// Iterate thorugh the neighbours of the site at the given index. The neighbours are all the sites
    /// with which the current one interacts.
    fn iter_neighbours(&self, index: Self::Idx) -> impl Iterator<Item = <Self as Lattice>::Idx>;

    /// The version of the Wolf algorithm we use is designed for the Pots model. Many other models (eg the Ising model)
    /// can be formulated in terms of a Potts model. This method returns the interaction strength in the lattice when it is
    /// re-formulated as Potts model.
    fn get_potts_interaction(&self) -> f64;
}
