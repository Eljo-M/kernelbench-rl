use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

const SUPERVISOR_GRACE_SECS: u64 = 15;
const METADATA_TIMEOUT_SECS: u64 = 15;

#[derive(Parser, Debug)]
#[command(
    name = "kernelbench-sandbox",
    about = "Timeout-bounded CPU development verifier (not a hostile-code security boundary)"
)]
struct Args {
    /// Operator name registered in kernels/op_registry.py
    #[arg(long)]
    op_name: String,

    /// Candidate path, resolved relative to the caller's working directory
    #[arg(long)]
    candidate_path: PathBuf,

    /// Candidate worker timeout; the supervisor allows additional cleanup time
    #[arg(long, default_value = "30", value_parser = parse_timeout)]
    timeout_secs: f64,

    /// Python interpreter, including an absolute virtual-environment path if needed
    #[arg(long, default_value = "python3")]
    python: PathBuf,

    /// Repository root; defaults to the source tree used to build this executable
    #[arg(long)]
    repo_root: Option<PathBuf>,

    /// Run identifier for telemetry logs
    #[arg(long, default_value = "local-dev")]
    run_id: String,
}

fn parse_timeout(raw: &str) -> Result<f64, String> {
    let value: f64 = raw.parse().map_err(|_| "timeout must be a number")?;
    let duration = Duration::try_from_secs_f64(value)
        .map_err(|_| "timeout must be positive, finite, and representable")?;
    if duration.is_zero()
        || duration
            .checked_add(Duration::from_secs(SUPERVISOR_GRACE_SECS))
            .is_none()
    {
        return Err("timeout must be positive and allow supervisor cleanup time".to_string());
    }
    Ok(value)
}

#[derive(Debug, Serialize, Deserialize)]
struct EvalResult {
    run_id: String,
    op_name: String,
    level: u8,
    backend: String,
    status: String,
    compiled: bool,
    correct: bool,
    cheat_detected: bool,
    max_abs_err: f64,
    eager_ms: f64,
    compile_ms: f64,
    candidate_ms: f64,
    speedup_vs_eager: f64,
    speedup_vs_compile: f64,
    #[serde(default)]
    achieved_tflops: f64,
    #[serde(default)]
    arithmetic_intensity: f64,
    reward: f64,
    #[serde(default)]
    error_message: String,
}

fn failure(args: &Args, level: u8, status: &str, message: &str) -> EvalResult {
    EvalResult {
        run_id: args.run_id.clone(),
        op_name: args.op_name.clone(),
        level,
        backend: "numpy_dev".to_string(),
        status: status.to_string(),
        compiled: false,
        correct: false,
        cheat_detected: false,
        max_abs_err: 0.0,
        eager_ms: 0.0,
        compile_ms: 0.0,
        candidate_ms: 0.0,
        speedup_vs_eager: 0.0,
        speedup_vs_compile: 0.0,
        achieved_tflops: 0.0,
        arithmetic_intensity: 0.0,
        reward: 0.0,
        error_message: message.chars().take(2000).collect(),
    }
}

fn repo_root(args: &Args) -> Result<PathBuf, String> {
    let root = args.repo_root.clone().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("Cargo manifest has a parent")
            .to_path_buf()
    });
    let root = root
        .canonicalize()
        .map_err(|err| format!("Cannot resolve repository root: {err}"))?;
    if !root.join("sandbox_runner/verifier.py").is_file()
        || !root.join("kernels/op_registry.py").is_file()
    {
        return Err("Repository root must contain sandbox_runner/verifier.py and kernels/op_registry.py".to_string());
    }
    Ok(root)
}

fn python_command(args: &Args, root: &Path) -> Command {
    let mut command = Command::new(&args.python);
    command
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command
}

async fn bounded_output(mut command: Command, limit: Duration) -> Result<Output, String> {
    let child = command
        .spawn()
        .map_err(|err| format!("Cannot start Python verifier: {err}"))?;
    timeout(limit, child.wait_with_output())
        .await
        .map_err(|_| "Supervisor timeout expired".to_string())?
        .map_err(|err| format!("Cannot collect Python verifier output: {err}"))
}

async fn operator_level(args: &Args, root: &Path) -> Result<u8, String> {
    // This imports only repository-owned metadata, before any candidate is loaded.
    let mut command = python_command(args, root);
    command
        .arg("-c")
        .arg("import json,sys; from kernels.op_registry import OP_REGISTRY; print(json.dumps({'level': OP_REGISTRY[sys.argv[1]].level}))")
        .arg(&args.op_name);
    let output = bounded_output(command, Duration::from_secs(METADATA_TIMEOUT_SECS)).await?;
    if !output.status.success() {
        return Err(format!(
            "Cannot resolve operator metadata: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(2000)
                .collect::<String>()
        ));
    }
    #[derive(Deserialize)]
    struct Metadata {
        level: u8,
    }
    let metadata: Metadata = serde_json::from_slice(&output.stdout)
        .map_err(|err| format!("Invalid operator metadata: {err}"))?;
    if !(1..=3).contains(&metadata.level) {
        return Err("Operator metadata level must be 1, 2, or 3".to_string());
    }
    Ok(metadata.level)
}

fn parse_result(output: &Output, args: &Args, level: u8) -> Result<EvalResult, String> {
    if !output.status.success() {
        return Err(format!(
            "Verifier exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(2000)
                .collect::<String>()
        ));
    }
    parse_payload(&output.stdout, args, level)
}

fn parse_payload(payload: &[u8], args: &Args, level: u8) -> Result<EvalResult, String> {
    let result: EvalResult = serde_json::from_slice(payload)
        .map_err(|err| format!("Invalid verifier JSON: {err}"))?;
    if result.op_name != args.op_name
        || result.run_id != args.run_id
        || result.level != level
        || result.backend != "numpy_dev"
    {
        return Err("Verifier result metadata does not match this CPU evaluation".to_string());
    }
    if ![
        "PASS", "COMPILE_ERROR", "NUMERICAL_MISMATCH", "TIMEOUT", "OOM", "RUNTIME_ERROR",
        "CHEAT",
    ]
    .contains(&result.status.as_str())
    {
        return Err("Verifier returned an unknown status".to_string());
    }
    if [
        result.eager_ms, result.compile_ms, result.candidate_ms,
        result.speedup_vs_eager, result.speedup_vs_compile,
        result.achieved_tflops, result.arithmetic_intensity,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
        || !result.max_abs_err.is_finite()
        || !result.reward.is_finite()
    {
        return Err("Verifier returned invalid numerical metrics".to_string());
    }
    Ok(result)
}

async fn evaluate(args: &Args, root: &Path, level: u8) -> EvalResult {
    let candidate = match args.candidate_path.canonicalize() {
        Ok(path) => path,
        Err(err) => {
            return failure(args, level, "RUNTIME_ERROR", &format!("Cannot resolve candidate path: {err}"));
        }
    };
    let mut command = python_command(args, root);
    command
        .arg("-m")
        .arg("sandbox_runner.verifier")
        .arg("--op-name")
        .arg(&args.op_name)
        .arg("--candidate-path")
        .arg(candidate)
        .arg("--run-id")
        .arg(&args.run_id)
        .arg("--timeout-secs")
        .arg(args.timeout_secs.to_string());
    let limit = Duration::try_from_secs_f64(args.timeout_secs)
        .expect("validated timeout")
        + Duration::from_secs(SUPERVISOR_GRACE_SECS);
    match bounded_output(command, limit).await {
        Ok(output) => parse_result(&output, args, level)
            .unwrap_or_else(|message| failure(args, level, "RUNTIME_ERROR", &message)),
        Err(message) => {
            let status = if message == "Supervisor timeout expired" { "TIMEOUT" } else { "RUNTIME_ERROR" };
            failure(args, level, status, &message)
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let root = match repo_root(&args) {
        Ok(root) => root,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let level = match operator_level(&args, &root).await {
        Ok(level) => level,
        Err(message) => {
            // Without trusted operator metadata, no schema-valid record can be invented.
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let result = evaluate(&args, &root, level).await;
    println!("{}", serde_json::to_string(&result).expect("finite evaluation result"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args::try_parse_from([
            "kernelbench-sandbox", "--op-name", "rope_2048", "--candidate-path", "candidate.py",
        ])
        .unwrap()
    }

    #[test]
    fn failure_is_finite_cpu_json_with_real_operator_level() {
        let result = failure(&args(), 2, "TIMEOUT", "worker timed out");
        let payload = serde_json::to_value(result).unwrap();
        assert_eq!(payload["level"], 2);
        assert_eq!(payload["backend"], "numpy_dev");
        assert_eq!(payload["status"], "TIMEOUT");
        assert_eq!(payload["max_abs_err"], 0.0);
        assert_eq!(payload["compile_ms"], 0.0);
        assert_eq!(payload["correct"], false);
    }

    #[test]
    fn timeouts_must_be_positive_and_finite() {
        for value in ["0", "-1", "NaN", "inf", "1e100"] {
            assert!(parse_timeout(value).is_err(), "accepted {value}");
        }
        assert_eq!(parse_timeout("0.25").unwrap(), 0.25);
    }

    #[test]
    fn interpreter_and_fractional_timeout_are_configurable() {
        let args = Args::try_parse_from([
            "kernelbench-sandbox", "--op-name", "rope_2048", "--candidate-path", "kernel.py",
            "--python", "C:/Python Env/python.exe", "--timeout-secs", "0.5",
        ])
        .unwrap();
        assert_eq!(args.python, PathBuf::from("C:/Python Env/python.exe"));
        assert_eq!(args.timeout_secs, 0.5);
    }

    #[test]
    fn failure_messages_are_bounded() {
        let result = failure(&args(), 3, "RUNTIME_ERROR", &"x".repeat(5000));
        assert_eq!(result.error_message.chars().count(), 2000);
    }

    #[test]
    fn results_must_match_operator_metadata_and_contract_numbers() {
        let args = args();
        let mut payload = serde_json::to_value(failure(&args, 2, "TIMEOUT", "timeout")).unwrap();
        assert!(parse_payload(&serde_json::to_vec(&payload).unwrap(), &args, 2).is_ok());
        payload["level"] = 1.into();
        assert!(parse_payload(&serde_json::to_vec(&payload).unwrap(), &args, 2).is_err());
        payload["level"] = 2.into();
        payload["candidate_ms"] = (-1).into();
        assert!(parse_payload(&serde_json::to_vec(&payload).unwrap(), &args, 2).is_err());
    }
}
