use crate::monte_carlo::creutz::CreutzResult;
use crate::monte_carlo::mc_results::MonteCarloResult;
use crate::utils;
use plotly::common::Mode;
use plotly::layout::{Axis, AxisType, GridPattern, Layout, LayoutGrid};
use plotly::{Bar, ImageFormat, Plot, Scatter};

fn save_or_show(plot: Plot, show: bool, save_as: Option<String>) {
    if let Some(path) = save_as {
        plot.write_image(path, ImageFormat::PNG, 1280, 900, 1.0);
    }

    if show {
        plot.show();
    }
}

fn line_trace_from_func(
    func: impl Fn(f64) -> f64,
    lower: f64,
    upper: f64,
    n: usize,
) -> Box<Scatter<f64, f64>> {
    let mut xx = utils::linspace_vector(lower, upper, n-1);
    xx.push(upper);
    let yy = xx.iter().map(|x| func(*x)).collect();
    Scatter::new(xx, yy).mode(Mode::Lines)
}

fn scatter_trace_magnetisations(mc_result: &MonteCarloResult) -> Box<Scatter<f64, f64>> {
    Scatter::new(
        mc_result.get_temperatures().to_vec(),
        mc_result.get_avg_abs_magnetisations().to_vec(),
    )
    .name("Absolute Magnetisations Per Spin")
    .mode(Mode::Markers)
}

fn scatter_trace_energies(mc_result: &MonteCarloResult) -> Box<Scatter<f64, f64>> {
    Scatter::new(
        mc_result.get_temperatures().to_vec(),
        mc_result.get_avg_energies().to_vec(),
    )
    .name("Energies Per Spin")
    .mode(Mode::Markers)
}

fn scatter_trace_heat_caps(mc_result: &MonteCarloResult) -> Box<Scatter<f64, f64>> {
    Scatter::new(
        mc_result.get_temperatures().to_vec(),
        mc_result.get_heat_capacities().to_vec(),
    )
    .name("Heat Capacities")
    .mode(Mode::Markers)
}

fn scatter_trace_mag_susc(mc_result: &MonteCarloResult) -> Box<Scatter<f64, f64>> {
    Scatter::new(
        mc_result.get_temperatures().to_vec(),
        mc_result.get_magnetic_susceptibilites().to_vec(),
    )
    .name("Magnetic Susceptibilites")
    .mode(Mode::Markers)
}

pub fn plot_magnetisations(mc_result: &MonteCarloResult, show: bool, save_as: Option<String>) {
    let mags_scatter = scatter_trace_magnetisations(&mc_result);

    let layout = Layout::new()
        .title(mc_result.get_lattice_description())
        .x_axis(Axis::new().title("Temperature [1]"))
        .y_axis(Axis::new().title("Abs Magnetisation [1]"));

    let mut plot = Plot::new();
    plot.add_trace(mags_scatter);
    plot.set_layout(layout);

    save_or_show(plot, show, save_as);
}

pub fn plot_binder_cumulants(mc_result: &MonteCarloResult, show: bool, save_as: Option<String>) {
    let bind_scatter = Scatter::new(
        mc_result.get_temperatures().to_vec(),
        mc_result.get_binder_cumulant().to_vec(),
    )
    .name("Binder Cumulants")
    .mode(Mode::Markers);

    let plot_min = utils::min_from_float_vec(mc_result.get_temperatures());
    let plot_max = utils::max_from_float_vec(mc_result.get_temperatures());
    let two_thirds_line = line_trace_from_func(|c| 2. / 3., plot_min, plot_max, 3).name("2/3");
    let zero_line = line_trace_from_func(|c| 0., plot_min, plot_max, 3).name("0");

    let layout = Layout::new()
        .title(mc_result.get_lattice_description())
        .x_axis(Axis::new().title("Temperature [1]"))
        .y_axis(Axis::new().title("Binder Cumulant [1]"));

    let mut plot = Plot::new();
    plot.add_trace(bind_scatter);
    plot.add_trace(two_thirds_line);
    plot.add_trace(zero_line);
    plot.set_layout(layout);

    save_or_show(plot, show, save_as);
}

pub fn plot_summary(mc_result: &MonteCarloResult, show: bool, save_as: Option<String>) {
    let mags_scatter = scatter_trace_magnetisations(&mc_result);
    let energies_scatter = scatter_trace_energies(&mc_result).x_axis("x2").y_axis("y2");
    let heat_caps_scatter = scatter_trace_heat_caps(&mc_result)
        .x_axis("x3")
        .y_axis("y3");
    let mag_susc_scatter = scatter_trace_mag_susc(&mc_result).x_axis("x4").y_axis("y4");

    let mut plot = Plot::new();

    plot.add_trace(mags_scatter);
    plot.add_trace(energies_scatter);
    plot.add_trace(heat_caps_scatter);
    plot.add_trace(mag_susc_scatter);

    let layout = Layout::new()
        .grid(
            LayoutGrid::new()
                .rows(2)
                .columns(2)
                .y_gap(0.6)
                .pattern(GridPattern::Independent),
        )
        .title(mc_result.get_lattice_description())
        .x_axis(Axis::new().title("Temperature [1]"))
        .x_axis2(Axis::new().title("Temperature [1]"))
        .x_axis3(Axis::new().title("Temperature [1]"))
        .x_axis4(Axis::new().title("Temperature [1]"))
        .y_axis(Axis::new().title("Abs Magnetisation [1]"))
        .y_axis2(Axis::new().title("Energy [1]"))
        .y_axis3(Axis::new().title("Heat Capacity [1]"))
        .y_axis4(Axis::new().title("Mag Susceptibility [1]"));
    plot.set_layout(layout);

    save_or_show(plot, show, save_as);
}

pub fn plot_demon_energy_histogram(
    creutz_result: &CreutzResult,
    show: bool,
    save_as: Option<String>,
) {
    let demon_energy_hist = creutz_result.get_demon_energy_histogram();
    let (interscept, slope) = creutz_result.fit_to_demon_energy_histogram().unwrap();

    let hist_trace = Bar::new(
        demon_energy_hist.get_left_bin_edges().to_vec(),
        demon_energy_hist.get_counts().to_vec(),
    )
    .name("Histogram");
    let fit_trace = line_trace_from_func(
        |x| interscept.exp() * (x * slope).exp(),
        0.,
        creutz_result.get_used_settings().get_max_demon_energy(),
        10,
    )
    .name(format!(
        "Linear fit: Temperature = {:0.2}",
        creutz_result.get_temperature().unwrap()
    ));

    let mut plot = Plot::new();

    let layout = Layout::new()
        .title(creutz_result.get_lattice_description())
        .x_axis(Axis::new().title("Demon Energy [1]"))
        .y_axis(Axis::new().type_(AxisType::Log).title("Counts [1]"));

    plot.add_trace(hist_trace);
    plot.add_trace(fit_trace);
    plot.set_layout(layout);

    save_or_show(plot, show, save_as);
}
