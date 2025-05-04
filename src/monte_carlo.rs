use single_spin_flip::SingleSpinFlipSettings;
use wolf_cluster_mc::ClusterFlipSettings;

pub mod creutz;
pub mod mc_results;
pub mod single_spin_flip;
pub mod wolf_cluster_mc;

#[derive(Debug)]
pub enum MonteCarloSettings {
    SingleSpinflip(SingleSpinFlipSettings),
    ClusterUpdate(ClusterFlipSettings),
}
