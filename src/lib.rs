use pyo3::prelude::*;

mod fsm1;

#[pymodule]
fn _fsm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    fsm1::register(m)?;
    Ok(())
}
