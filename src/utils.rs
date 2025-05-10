use itertools::Itertools;
use nalgebra;
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;
use std::cell::OnceCell;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct Histogram {
    counts: Vec<u64>,
    left_bin_edges: Vec<f64>,
}

impl Histogram {
    pub fn get_counts(&self) -> &Vec<u64> {
        &self.counts
    }

    pub fn get_left_bin_edges(&self) -> &Vec<f64> {
        &self.left_bin_edges
    }
}

fn to_ordered_floats(v: &Vec<f64>) -> Vec<f64> {
    v.into_iter()
        .sorted_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|i| *i)
        .collect()
}

pub fn min_from_float_vec(v: &Vec<f64>) -> f64 {
    *v.into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .expect("Vector must not be empty")
}

pub fn max_from_float_vec(v: &Vec<f64>) -> f64 {
    *v.into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .expect("Vector must not be empty")
}

pub fn histogram(data: &Vec<f64>, left_bin_edges: &Vec<f64>) -> Histogram {
    let sorted_bins = to_ordered_floats(left_bin_edges);
    let mut hist_out = vec![0; sorted_bins.len()];
    let last_bin_idx = sorted_bins.len() - 2;
    let lower_bound = sorted_bins[0];

    'data_loop: for &point in data {
        if point <= lower_bound {
            // ignore everything left of the first bin edge
            continue 'data_loop;
        }

        'bin_loop: for (bin_idx, &bin_bound) in sorted_bins[1..].iter().enumerate() {
            if point <= bin_bound {
                hist_out[bin_idx] += 1;
                break 'bin_loop;
            }
            if bin_idx == last_bin_idx {
                // the point is bigger than all the bin bounds -> add it to the last bin
                hist_out[bin_idx + 1] += 1;
            }
        }
    }

    return Histogram {
        counts: hist_out,
        left_bin_edges: sorted_bins,
    };
}

pub fn linspace_vector(lower: f64, upper: f64, n: usize) -> Vec<f64> {
    let dx = (upper - lower) / (n - 1) as f64;
    (0..n - 1).map(|i| lower + i as f64 * dx).collect()
}

pub fn linspace_vector_from_step(start: f64, stop: f64, step: f64) -> Vec<f64> {
    assert!(stop > start, "start must be bigger than stop.");
    assert!(step > 0., "The step must be positive.");

    let n = ((stop - start) / step - 1.).ceil() as usize;
    (0..n + 1).map(|i| start + i as f64 * step).collect()
}

fn square_root_choice_bins(data: &Vec<f64>) -> Vec<f64> {
    let n_bins = (data.len() as f64).sqrt().floor() as usize;
    linspace_vector(data.min(), data.max(), n_bins + 1)
}

pub fn histogram_sqt_choice(data: &Vec<f64>) -> Histogram {
    let bins = square_root_choice_bins(data);
    histogram(data, &bins)
}

pub fn least_squares_lin_fit(x: &Vec<f64>, y: &Vec<f64>) -> Option<(f64, f64)> {
    // TODO: maybe replace this by a call to a dedicated package
    let x_vec = nalgebra::DVector::from((*x).clone());
    let y_vec = nalgebra::DVector::from((*y).clone());
    let ones = nalgebra::DVector::repeat(x.len(), 1.);
    let x_mat = nalgebra::MatrixXx2::from_columns(&[ones, x_vec]);

    let x_mat_t = x_mat.transpose();
    let fitted_params = (&x_mat_t * &x_mat).try_inverse()? * &x_mat_t * &y_vec;

    Some((fitted_params[(0, 0)], fitted_params[(1, 0)]))
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(from = "Option<T>")]
pub struct SerdeOnceCell<T: Serialize + std::fmt::Debug> {
    cell: OnceCell<T>,
}

impl<T: Serialize + std::fmt::Debug> Serialize for SerdeOnceCell<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.get() {
            Some(inner_value) => serializer.serialize_some(inner_value),
            None => serializer.serialize_none(),
        }
    }
}

impl<T: Serialize + std::fmt::Debug> From<Option<T>> for SerdeOnceCell<T> {
    fn from(value: Option<T>) -> Self {
        let new_cell = SerdeOnceCell::new();
        if let Some(inner_value) = value {
            new_cell
                .set(inner_value)
                .expect("Value of new cell should never be set.");
        }
        new_cell
    }
}

impl<T: Serialize + std::fmt::Debug> SerdeOnceCell<T> {
    pub fn new() -> Self {
        SerdeOnceCell {
            cell: OnceCell::new(),
        }
    }

    pub fn get(&self) -> Option<&T> {
        self.cell.get()
    }

    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        self.cell.get_or_init(f)
    }

    pub fn set(&self, value: T) -> Result<(), T> {
        self.cell.set(value)
    }
}

#[cfg(test)]
mod test {
    use core::f64;
    use serde_test::{assert_tokens, Token};
    use std::iter::zip;

    use super::*;
    use approx;

    #[test]
    fn histogram1() {
        let test_data = vec![10., 11., 10., 9.5, 12.];
        let test_bins = vec![0., 9., 10., 11.];
        let expected_hist = vec![0, 3, 1, 1];

        let actual_hist = histogram(&test_data, &test_bins);

        assert_eq!(*actual_hist.get_counts(), expected_hist);
        assert_eq!(*actual_hist.get_left_bin_edges(), test_bins);
    }

    #[test]
    fn histogram2() {
        let test_data = vec![-1.5, 12., -0.5, 1., 1.1, 1., 9., 3.];
        let test_bins = vec![0., 1., 2., 3.];
        let expected_hist = vec![2, 1, 1, 2];

        let actual_hist = histogram(&test_data, &test_bins);

        assert_eq!(*actual_hist.get_counts(), expected_hist);
        assert_eq!(*actual_hist.get_left_bin_edges(), test_bins);
    }

    #[test]
    fn histogram3() {
        let test_data = vec![-2.001, 100., 15.001, -1., 0., -0., -10., 1., -3.1];
        // the first bin collects everything smaller than -2
        let test_bins = vec![f64::NEG_INFINITY, -2., -1.5, 0.5, 3.3, 2.1, 15.];
        let sorted_test_bins = to_ordered_floats(&test_bins);
        let expected_hist = vec![3, 0, 3, 1, 0, 0, 2];

        let actual_hist = histogram(&test_data, &test_bins);

        assert_eq!(*actual_hist.get_counts(), expected_hist);
        assert_eq!(*actual_hist.get_left_bin_edges(), sorted_test_bins);
    }

    #[test]
    fn linspace_vector_simple() {
        assert_eq!(vec![1., 2.], linspace_vector(1., 3., 3));
    }

    #[test]
    fn linspace_vector_complex() {
        let expected_linspace = vec![0., 0.5, 1., 1.5, 2., 2.5, 3.];
        let actual_linspace = linspace_vector(0., 3.5, 8);

        assert_eq!(
            expected_linspace.len(),
            actual_linspace.len(),
            "Length of actual linspace {actual_linspace:?} not as expected."
        );
        for (expected, actual) in zip(expected_linspace, actual_linspace) {
            approx::assert_relative_eq!(expected, actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn linspace_vector_from_step_simple() {
        let extpected_linspace = vec![0., 4., 8., 12., 16.];
        let actual_linspace = linspace_vector_from_step(0., 20., 4.);

        assert_eq!(
            extpected_linspace.len(),
            actual_linspace.len(),
            "Length of resulting linspace {actual_linspace:?} not as expected."
        );
        for (expected, actual) in zip(extpected_linspace, actual_linspace) {
            approx::assert_relative_eq!(expected, actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn linspace_vector_from_step_complex() {
        let extpected_linspace = vec![
            -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.,
        ];
        let actual_linspace = linspace_vector_from_step(-1., 2.22, 0.25);

        assert_eq!(
            extpected_linspace.len(),
            actual_linspace.len(),
            "Length of linspace {actual_linspace:?} not as expected."
        );
        for (expected, actual) in zip(extpected_linspace, actual_linspace) {
            approx::assert_relative_eq!(expected, actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn square_root_choice_bins1() {
        let test_data = vec![1., 2., 5., 2., 3., 1.];
        let expected_bins = vec![1., 3.];
        let actual_bins = square_root_choice_bins(&test_data);

        assert_eq!(
            expected_bins.len(),
            actual_bins.len(),
            "Length of bins not as expected."
        );
        for (expected, actual) in zip(expected_bins, actual_bins) {
            approx::assert_relative_eq!(expected, actual, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn square_root_choice_bins2() {
        let test_data = vec![-10., 0., 1., 0., 1., 10., 9., -8., 5.];
        let expected_bins = vec![-10., -3.3333, 3.3333];
        let actual_bins = square_root_choice_bins(&test_data);

        assert_eq!(
            expected_bins.len(),
            actual_bins.len(),
            "Length of bins not as expected."
        );
        for (expected, actual) in zip(expected_bins, actual_bins) {
            approx::assert_abs_diff_eq!(expected, actual, epsilon = 0.0001);
        }
    }

    #[test]
    fn least_squares_lin_fit_extact_fit1() {
        let test_x = vec![1., 2.];
        let test_y = vec![1., 2.];
        match least_squares_lin_fit(&test_x, &test_y) {
            Some((q, m)) => {
                approx::assert_relative_eq!(q, 0., epsilon = 1e-6);
                approx::assert_relative_eq!(m, 1., epsilon = 1e-6);
            }
            None => panic!("No result from linear regression."),
        };
    }

    #[test]
    fn least_squares_lin_fit_extact_fit2() {
        let test_x = vec![1., 2., 3., 4., 5.];
        let test_y = vec![4., 5., 6., 7., 8.];
        match least_squares_lin_fit(&test_x, &test_y) {
            Some((q, m)) => {
                approx::assert_relative_eq!(q, 3., epsilon = 1e-6);
                approx::assert_relative_eq!(m, 1., epsilon = 1e-6);
            }
            None => panic!("No result from linear regression."),
        };
    }

    #[test]
    fn least_squares_lin_fit_extact_perfect_colinear() {
        let test_x = vec![1., 1., 1., 1., 1.];
        let test_y = vec![4.1, 4.2, 3.7, 4.7, 3.8];
        match least_squares_lin_fit(&test_x, &test_y) {
            Some(_) => panic!("Expected no result in perfectly colinear case."),
            None => (),
        };
    }

    #[test]
    fn once_cell_serde_int_empty() {
        let cell = SerdeOnceCell::<i32>::new();

        assert_tokens(&cell, &[Token::None]);
    }

    #[test]
    fn once_cell_serde_vec_empty() {
        let cell = SerdeOnceCell::<Vec<f64>>::new();

        assert_tokens(&cell, &[Token::None]);
    }

    #[test]
    fn once_cell_serde_i64() {
        let cell = SerdeOnceCell::<i64>::new();
        cell.set(12).unwrap();

        assert_tokens(&cell, &[Token::Some, Token::I64(12)]);
    }

    #[test]
    fn once_cell_serde_vec() {
        let cell = SerdeOnceCell::<Vec<f64>>::new();
        cell.set(vec![0., 1.2, 3., -12., 0.99, -111.22]).unwrap();

        assert_tokens(
            &cell,
            &[
                Token::Some,
                Token::Seq { len: Some(6) },
                Token::F64(0.),
                Token::F64(1.2),
                Token::F64(3.),
                Token::F64(-12.),
                Token::F64(0.99),
                Token::F64(-111.22),
                Token::SeqEnd,
            ],
        );
    }

    #[test]
    fn once_cell_serde_string() {
        let cell = SerdeOnceCell::<String>::new();
        cell.set(String::from("This is a test ...?")).unwrap();

        assert_tokens(&cell, &[Token::Some, Token::String("This is a test ...?")]);
    }
}
