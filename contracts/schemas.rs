use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvalStatus {
    Pass,
    CompileError,
    NumericalMismatch,
    Timeout,
    Oom,
    RuntimeError,
    Cheat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelEvalResult {
    pub run_id: String,
    pub op_name: String,
    pub level: u8,
    pub backend: String,
    pub status: EvalStatus,
    pub compiled: bool,
    pub correct: bool,
    pub cheat_detected: bool,
    pub max_abs_err: f64,
    pub eager_ms: f64,
    pub compile_ms: f64,
    pub candidate_ms: f64,
    pub speedup_vs_eager: f64,
    pub speedup_vs_compile: f64,
    pub achieved_tflops: f64,
    pub arithmetic_intensity: f64,
    pub reward: f64,
    pub error_message: String,
}
