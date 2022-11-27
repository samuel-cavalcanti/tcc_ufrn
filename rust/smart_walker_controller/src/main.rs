mod collect_simulation_data;
mod controller;
mod coppeliasimulation;
mod test_kinematic_model;
mod benchmark;

fn main() {
    // collect_simulation_data::collect_data();
    test_kinematic_model::run();
    // benchmark::run();
}
