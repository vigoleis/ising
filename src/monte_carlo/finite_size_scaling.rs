use crate::lattice::Lattice;
use crate::monte_carlo::wolf_cluster_mc::ClusterFlipSettings;
use crate::monte_carlo::mc_result;

pub fn cluster_montecarlo_for_sizes<L: Lattice>(
    sizes: Vec<i64>,
    lattice_factory: Fn(int64) -> L,
    settings: ClusterFlipSettings,
) -> Vec<> {
    
}