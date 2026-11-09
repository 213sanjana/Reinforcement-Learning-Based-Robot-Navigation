"""Transfer a frozen Jackal behavior-cloning actor into a PPO policy.

The command loads the Phase 6F.2 checkpoint, builds a Stable-Baselines3 PPO
model on the Jackal observation and action spaces, and copies the compatible
actor weights. It does not call ``learn``, step the Jackal, or publish a
velocity command.

Run from the repository root:

    python -m navigation.jackal_bc_ppo_transfer --config configs/jackal_ppo_transfer_6g1.yaml
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import gymnasium
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from agents.jackal_pretrained_ppo_policy import (
    PHASE,
    JackalPolicyContractEnv,
    JackalPretrainedPPOPolicy,
    audit_bc_actor_layout,
    check_sb3_version,
)
from navigation.jackal_bc_policy import (
    ARCHITECTURE,
    JackalBCPolicy,
    load_checkpoint,
    resolve_device,
)
from navigation.jackal_bc_schema import ACTION_NAMES, feature_names, require_env_actions


def file_sha256(path):
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    """Hash a tensor's dtype, shape, and little-endian bytes."""

    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_config(path):
    """Load and check a Phase 6G.1 YAML configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def validate_config(config):
    """Raise unless ``config`` is a Phase 6G.1 transfer configuration."""

    if not isinstance(config, dict) or config.get("phase") != PHASE:
        raise ValueError("Configuration phase must be 6g.1")
    source = _mapping(config, "source")
    runtime = _mapping(config, "runtime")
    ppo = _mapping(config, "ppo")
    validation = _mapping(config, "validation")
    output = _mapping(config, "output")
    protected = config.get("protected")
    if not isinstance(protected, list) or not protected:
        raise ValueError("protected must list the BC artifacts that stay unchanged")
    if not source.get("checkpoint"):
        raise ValueError("BC checkpoint path is required")
    if not isinstance(source.get("require_trained"), bool):
        raise ValueError("require_trained must be a boolean")
    for name in ("architecture", "preprocessing", "evaluation_context"):
        if source.get(name) is not None and not isinstance(source.get(name), str):
            raise ValueError(f"{name} reference must be a path")
    if runtime.get("deterministic") is not True:
        raise ValueError("Phase 6G.1 transfer is deterministic")
    _nonnegative_int(runtime.get("seed"), "seed")
    if runtime.get("device") not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if not output.get("directory"):
        raise ValueError("Output directory is required")

    _unit_interval(ppo.get("gamma"), "gamma")
    _unit_interval(ppo.get("gae_lambda"), "gae_lambda")
    _unit_interval(ppo.get("clip_range"), "clip_range")
    for name in ("learning_rate", "ent_coef", "vf_coef", "max_grad_norm"):
        _nonnegative_number(ppo.get(name), name)
    if float(ppo["learning_rate"]) <= 0.0 or float(ppo["vf_coef"]) <= 0.0:
        raise ValueError("learning_rate and vf_coef must be positive")
    _positive_int(ppo.get("n_steps"), "n_steps")
    _positive_int(ppo.get("batch_size"), "batch_size")
    _positive_int(ppo.get("n_epochs"), "n_epochs")
    if int(ppo["n_steps"]) % int(ppo["batch_size"]) != 0:
        raise ValueError("batch_size must divide n_steps")
    if ppo.get("use_sde") is not False:
        raise ValueError("Jackal transfer uses a diagonal Gaussian, not gSDE")
    if ppo.get("normalize_advantage") is not True:
        raise ValueError("normalize_advantage must stay true")
    _finite_number(ppo.get("log_std_init"), "log_std_init")

    _positive_int(validation.get("observations"), "observations")
    _nonnegative_int(validation.get("seed"), "validation seed")
    _nonnegative_number(validation.get("atol"), "atol")
    _nonnegative_number(validation.get("rtol"), "rtol")
    return config


def load_jackal_bc_for_transfer(path, device="cpu"):
    """Load a Jackal BC checkpoint and reject an incompatible actor."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    _prevalidate_payload(payload)
    policy, loaded = load_checkpoint(path, device=device)
    audit_bc_actor_layout(policy)
    if not policy.normalizer_is_ready:
        raise ValueError("Checkpoint is missing train-split normalization")
    return policy, loaded


def transfer_actor(bc_policy, ppo_policy):
    """Copy compatible BC tensors into the PPO actor and leave the critic weights alone.

    Frozen normalizer buffers are also installed on the critic so its forward
    pass applies the same train-split z-score once. Those buffers are not
    actor weights.
    """

    if not isinstance(bc_policy, JackalBCPolicy):
        raise TypeError("Source policy must be a JackalBCPolicy")
    if not isinstance(ppo_policy, JackalPretrainedPPOPolicy):
        raise TypeError("Target policy must be a JackalPretrainedPPOPolicy")
    audit_bc_actor_layout(bc_policy)
    if bc_policy.network_config() != ppo_policy.network_config:
        raise ValueError("PPO actor was built from a different network config")
    if not bc_policy.normalizer_is_ready:
        raise RuntimeError("Train-split observation normalizer is not installed")

    bc_before = _clone_state(bc_policy)
    critic_parameters = _parameter_snapshot(ppo_policy.vf_features_extractor)
    critic_parameters.update(
        {f"value_net.{name}": value for name, value in _parameter_snapshot(ppo_policy.value_net).items()}
    )
    log_std_before = ppo_policy.log_std.detach().cpu().clone()
    pairs = _transfer_pairs(bc_policy, ppo_policy)
    consumed = {pair["source"] for pair in pairs}
    if consumed != set(bc_policy.state_dict()):
        missing = sorted(set(bc_policy.state_dict()) - consumed)
        extra = sorted(consumed - set(bc_policy.state_dict()))
        raise ValueError(f"BC state dict was not fully mapped: missing={missing} extra={extra}")

    with torch.no_grad():
        for pair in pairs:
            destination = pair["destination"]
            source = pair["source_tensor"]
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError(f"Refusing to copy incompatible tensor {pair['source']}")
            destination.copy_(source.detach().to(device=destination.device))

    _assert_actor_matches(bc_policy, ppo_policy, pairs)
    _assert_parameters_unchanged(ppo_policy.vf_features_extractor, critic_parameters)
    _assert_module_parameters_unchanged(ppo_policy.value_net, critic_parameters, "value_net.")
    if not torch.equal(ppo_policy.log_std.detach().cpu(), log_std_before):
        raise ValueError("Gaussian log_std changed during actor transfer")

    _install_critic_normalizer(bc_policy, ppo_policy)
    _assert_parameters_unchanged(ppo_policy.vf_features_extractor, critic_parameters)
    _assert_module_parameters_unchanged(ppo_policy.value_net, critic_parameters, "value_net.")
    if not torch.equal(ppo_policy.log_std.detach().cpu(), log_std_before):
        raise ValueError("Gaussian log_std changed while installing the critic normalizer")
    if not _states_equal(bc_policy, bc_before):
        raise RuntimeError("Transfer modified the BC policy")
    _assert_partition(ppo_policy)
    if ppo_policy.optimizer.state:
        raise ValueError("Expected a fresh PPO optimizer with no transferred state")
    optimizer_ids = {
        id(parameter)
        for group in ppo_policy.optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in ppo_policy.parameters()}:
        raise ValueError("PPO optimizer does not reference every policy parameter")

    identical = _identical_trunk_parameters(ppo_policy)
    if len(identical) == len(list(ppo_policy.pi_features_extractor.parameters())):
        raise ValueError("Critic trunk matches the actor trunk")
    return _transfer_report(ppo_policy, pairs, identical, log_std_before)


def validate_equivalence(bc_policy, model, observations, atol, rtol):
    """Compare deterministic BC actions with the PPO Gaussian mean."""

    array = np.asarray(observations, dtype=np.float32)
    if array.ndim != 2 or int(array.shape[1]) != 40 or len(array) < 1:
        raise ValueError("Validation observations must be shaped (N, 40)")
    if not np.isfinite(array).all():
        raise ValueError("Validation observations must be finite")

    policy = model.policy
    if not isinstance(policy, JackalPretrainedPPOPolicy):
        raise TypeError("Target policy must be a JackalPretrainedPPOPolicy")
    if policy.squash_output or policy.normalize_images:
        raise ValueError("PPO would rescale actions or observations")
    policy.set_training_mode(False)
    bc_policy.eval()
    device = next(policy.parameters()).device
    if next(bc_policy.parameters()).device != device:
        raise ValueError("Compare BC and PPO on the same device")

    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
    with torch.inference_mode():
        bc_first = bc_policy(tensor)
        bc_second = bc_policy(tensor)
        if not torch.equal(bc_first, bc_second):
            raise RuntimeError("BC deterministic output is not reproducible")
        raw, mean = policy.raw_and_mean(tensor)
        if not torch.equal(mean, torch.tanh(raw)):
            raise RuntimeError("Gaussian mean is not tanh of the actor raw output")
        mode = policy.get_distribution(tensor).mode()
        if not torch.equal(mode, mean):
            raise RuntimeError("Distribution mode differs from the tanh mean")
        matched, error = _tensors_match(mean, bc_first, atol, rtol)
        if not matched:
            raise ValueError(
                f"PPO deterministic mean differs from the BC action; max abs error={error}"
            )
        values = policy.predict_values(tensor)
        if values.shape != (len(tensor), 1) or not torch.isfinite(values).all():
            raise ValueError("Critic outputs are not finite")
        generator_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(0)
            sample = policy.get_distribution(tensor).sample()
        finally:
            torch.random.set_rng_state(generator_state)
        if not torch.isfinite(sample).all():
            raise ValueError("Gaussian samples are not finite")

    mean_numpy = mean.detach().cpu().numpy()
    bc_numpy = bc_first.detach().cpu().numpy()
    predicted, _state = model.predict(array, deterministic=True)
    if predicted.shape != bc_numpy.shape or predicted.dtype != np.float32:
        raise ValueError("predict() did not return a float32 action batch")
    if not np.isfinite(predicted).all():
        raise ValueError("Deterministic PPO actions are not finite")
    require_env_actions(predicted)
    low = np.asarray(policy.action_space.low, dtype=np.float32)
    high = np.asarray(policy.action_space.high, dtype=np.float32)
    clipped = np.clip(mean_numpy, low, high)
    if not np.array_equal(predicted, clipped):
        raise ValueError("predict() did not apply only the Gymnasium box clip")
    action_matched, action_error = _arrays_match(predicted, bc_numpy, atol, rtol)
    if not action_matched:
        raise ValueError(
            f"Clipped PPO action differs from the BC action; max abs error={action_error}"
        )
    single, _state = model.predict(array[0], deterministic=True)
    if single.shape != (2,):
        raise ValueError("Unbatched deterministic action is not shaped (2,)")
    require_env_actions(single)
    # A batch-size-1 GEMM can differ by a few float32 ulps from the same row
    # inside a larger batch. The batched mean is the exactness check.
    unbatched_error = float(np.max(np.abs(
        single.astype(np.float64) - predicted[0].astype(np.float64)
    )))
    if unbatched_error > 1e-6:
        raise ValueError(
            "Unbatched deterministic action does not match the batch row; "
            f"max abs error={unbatched_error}"
        )

    stochastic, _state = model.predict(array, deterministic=False)
    if stochastic.shape != predicted.shape or not np.isfinite(stochastic).all():
        raise ValueError("Stochastic PPO actions are not finite")
    require_env_actions(stochastic)
    sample_numpy = sample.detach().cpu().numpy()
    return {
        "batch_size": int(len(array)),
        "atol": float(atol),
        "rtol": float(rtol),
        "parameter_match": "exact",
        "bc_output_reproducible": True,
        "preprocessing_identity": True,
        "second_normalization": False,
        "squash_output": False,
        "unscale_action_applied": False,
        "deterministic_mean_max_absolute_error": error,
        "deterministic_mean_exact": bool(error == 0.0 and atol == 0.0 and rtol == 0.0),
        "predict_matches_clipped_mean": True,
        "clipped_action_max_absolute_error": action_error,
        "unbatched_max_absolute_error": unbatched_error,
        "unbatched_tolerance": 1e-6,
        "unbatched_tolerance_reason": (
            "PyTorch float32 linear algebra may use a different kernel for "
            "batch size 1 than for a larger batch. The batched deterministic "
            "mean is compared exactly with the BC action."
        ),
        "actions_finite": True,
        "actions_inside_gymnasium_box": True,
        "critic_outputs_finite": True,
        "gaussian_sample_finite": True,
        "gaussian_sample_left_action_box": bool(
            np.any(sample_numpy < low) or np.any(sample_numpy > high)
        ),
        "action_contract": action_contract(),
    }


def initialize_jackal_ppo(config, output_directory):
    """Build, check, and publish one PPO initialization directory."""

    config = validate_config(config)
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_directory}")
    source_path = Path(config["source"]["checkpoint"])
    _reject_output_inside_source(output_directory, source_path)
    protected_before = _hash_existing(config["protected"])
    check_sb3_version()
    seed = int(config["runtime"]["seed"])
    device = resolve_device(config["runtime"]["device"])
    _seed_everything(seed, use_cuda=device.type == "cuda")
    source_hash = file_sha256(source_path)
    bc_policy, payload = load_jackal_bc_for_transfer(source_path, device=str(device))
    if config["source"]["require_trained"]:
        if payload.get("trained") is not True or payload.get("training_phase") != "6f.2":
            raise ValueError("Official transfer requires the trained Phase 6F.2 policy")
    references = _source_references(config, bc_policy, payload, source_hash)
    observations = _fixed_observations(
        int(config["validation"]["observations"]),
        int(config["validation"]["seed"]),
    )
    env = JackalPolicyContractEnv()
    model = None
    try:
        model = PPO(
            JackalPretrainedPPOPolicy,
            env,
            policy_kwargs={
                "network_config": bc_policy.network_config(),
                "log_std_init": float(config["ppo"]["log_std_init"]),
            },
            learning_rate=float(config["ppo"]["learning_rate"]),
            n_steps=int(config["ppo"]["n_steps"]),
            batch_size=int(config["ppo"]["batch_size"]),
            n_epochs=int(config["ppo"]["n_epochs"]),
            gamma=float(config["ppo"]["gamma"]),
            gae_lambda=float(config["ppo"]["gae_lambda"]),
            clip_range=float(config["ppo"]["clip_range"]),
            ent_coef=float(config["ppo"]["ent_coef"]),
            vf_coef=float(config["ppo"]["vf_coef"]),
            max_grad_norm=float(config["ppo"]["max_grad_norm"]),
            normalize_advantage=True,
            use_sde=False,
            seed=seed,
            device=str(device),
            verbose=0,
        )
        if model.policy.pi_features_extractor.trunk.normalizer_is_ready:
            raise RuntimeError("Actor normalizer was populated before transfer")
        transfer = transfer_actor(bc_policy, model.policy)
        verification = validate_equivalence(
            bc_policy,
            model,
            observations,
            float(config["validation"]["atol"]),
            float(config["validation"]["rtol"]),
        )
        _assert_not_trained(model, env)
        if _hash_existing(config["protected"]) != protected_before:
            raise RuntimeError("A protected BC artifact changed during transfer")
        if file_sha256(source_path) != source_hash:
            raise RuntimeError("Source BC checkpoint changed during transfer")

        def populate(directory):
            _write_run(
                directory,
                published_directory=output_directory,
                model=model,
                bc_policy=bc_policy,
                observations=observations,
                config=config,
                source_path=source_path,
                source_hash=source_hash,
                payload=payload,
                references=references,
                transfer=transfer,
                verification=verification,
                protected_before=protected_before,
                device=device,
            )

        publish_directory(output_directory.parent, output_directory.name, populate)
    finally:
        if model is not None and model.get_env() is not None:
            model.get_env().close()
        else:
            env.close()

    if _hash_existing(config["protected"]) != protected_before:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise RuntimeError("A protected BC artifact changed after publishing")
    summary = json.loads((output_directory / "initialization_summary.json").read_text())
    if summary.get("status") != "passed":
        raise RuntimeError("Published initialization did not record a passing status")
    return summary


def publish_directory(parent, name, populate):
    """Create ``parent/name`` only after ``populate`` finishes on a temporary directory."""

    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    temporary = parent / f".{name}.{os.getpid()}.partial"
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {temporary}")
    temporary.mkdir()
    try:
        populate(temporary)
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def action_contract():
    """Describe raw, distribution, and environment actions."""

    return {
        "order": list(ACTION_NAMES),
        "actor_raw_output": (
            "Unbounded pre-tanh linear output of action_net, ordered [linear, angular]."
        ),
        "distribution_mean": (
            "tanh(actor_raw_output), in (-1, 1). This is the diagonal-Gaussian mean "
            "and the deterministic distribution action."
        ),
        "distribution_sample": (
            "Normal(distribution_mean, exp(log_std)). Unbounded. log_std is a new "
            "parameter and is not taken from the BC checkpoint."
        ),
        "predict_deterministic": (
            "Distribution mean, then Stable-Baselines3 clips each component to the "
            "Gymnasium box [-1, 1]. Finite tanh values lie in (-1, 1), so the clip "
            "does not change them."
        ),
        "predict_stochastic": (
            "Gaussian sample, then the same box clip. The clipped vector is the "
            "action that would be passed to env.step."
        ),
        "env_step": "Not called. This phase does not step JackalGazeboEnv.",
        "squash_output": False,
        "unscale_action_applied": False,
        "additional_action_rescaling": False,
        "observation_normalization": (
            "The frozen BC z-score runs once inside the actor trunk. "
            "Stable-Baselines3 preprocess_obs is an identity float cast for this "
            "40-D Box because normalize_images is false and the observation is not an image."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-directory",
        help="New directory for this run. Defaults to a timestamped directory.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_directory:
        output = Path(args.output_directory)
    else:
        stamp = _timestamp()
        output = Path(config["output"]["directory"]) / f"run_{stamp}"
    summary = initialize_jackal_ppo(config, output)
    print(json.dumps({
        "status": summary["status"],
        "training_started": summary["training_started"],
        "ppo_num_timesteps": summary["ppo_num_timesteps"],
        "source_checkpoint_sha256": summary["source_checkpoint_sha256"],
        "deterministic_mean_exact": summary["deterministic_mean_exact"],
        "transferred_parameter_elements": summary["transferred_parameter_elements"],
        "output_directory": summary["output_directory"],
    }, indent=2))
    return 0


def _write_run(
    directory,
    *,
    published_directory,
    model,
    bc_policy,
    observations,
    config,
    source_path,
    source_hash,
    payload,
    references,
    transfer,
    verification,
    protected_before,
    device,
):
    directory = Path(directory)
    checkpoint_path = directory / "initialized_ppo.zip"
    observations_path = directory / "equivalence_observations.npz"
    model.save(checkpoint_path)
    _fsync(checkpoint_path)
    with observations_path.open("wb") as handle:
        np.savez_compressed(handle, observations=np.ascontiguousarray(observations))
        handle.flush()
        os.fsync(handle.fileno())

    reloaded = PPO.load(checkpoint_path, device=str(device))
    try:
        reload_verification = validate_equivalence(
            bc_policy,
            reloaded,
            observations,
            float(config["validation"]["atol"]),
            float(config["validation"]["rtol"]),
        )
        for key, value in model.policy.state_dict().items():
            restored = reloaded.policy.state_dict()[key]
            if not torch.equal(value.detach().cpu(), restored.detach().cpu()):
                raise ValueError(f"Reloaded PPO tensor changed: {key}")
        if reloaded.num_timesteps != 0 or reloaded._n_updates != 0 or reloaded.policy.optimizer.state:
            raise ValueError("Reloaded PPO is not a fresh initialization")
    finally:
        if reloaded.get_env() is not None:
            reloaded.get_env().close()

    fresh = _fresh_process_check(
        checkpoint_path,
        observations_path,
        source_path,
        float(config["validation"]["atol"]),
        float(config["validation"]["rtol"]),
    )
    versions = _versions()
    contract = action_contract()
    summary = {
        "phase": PHASE,
        "status": "passed",
        "training_started": False,
        "ppo_num_timesteps": 0,
        "ppo_updates": 0,
        "environment_steps": 0,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": source_hash,
        "source_checkpoint_preserved": True,
        "source_architecture": payload.get("architecture"),
        "source_training_phase": payload.get("training_phase"),
        "source_epoch": payload.get("epoch"),
        "source_best_epoch": payload.get("best_epoch"),
        "deterministic_mean_exact": verification["deterministic_mean_exact"],
        "deterministic_mean_max_absolute_error": verification[
            "deterministic_mean_max_absolute_error"
        ],
        "unbatched_max_absolute_error": verification["unbatched_max_absolute_error"],
        "transferred_parameter_tensors": transfer["transferred_parameter_tensors"],
        "transferred_parameter_elements": transfer["transferred_parameter_elements"],
        "transferred_buffer_tensors": transfer["transferred_buffer_tensors"],
        "newly_initialized_groups": transfer["newly_initialized_groups"],
        "intentionally_excluded": transfer["intentionally_excluded"],
        "critic_initialized_independently": True,
        "critic_parameters_unchanged_by_transfer": True,
        "critic_normalizer_buffers_copied": True,
        "gaussian_log_std_transferred": False,
        "output_directory": str(Path(published_directory).resolve()),
        "initialized_ppo": checkpoint_path.name,
        "initialized_ppo_sha256": file_sha256(checkpoint_path),
        "action_contract": contract,
    }
    _write_json(directory / "initialization_summary.json", summary)
    _write_json(directory / "transfer_mapping.json", transfer)
    _write_json(directory / "verification.json", {
        "in_process": verification,
        "reload": reload_verification,
        "fresh_process": fresh,
    })
    _write_json(directory / "framework_versions.json", versions)
    _write_json(directory / "source_references.json", references)
    _write_json(directory / "ppo_config.json", {
        "optimized": False,
        "learn_called": False,
        "runtime": config["runtime"],
        "ppo": config["ppo"],
        "validation": config["validation"],
        "policy_class": "agents.jackal_pretrained_ppo_policy.JackalPretrainedPPOPolicy",
        "net_arch": {"pi": [], "vf": []},
        "activation_fn": "ReLU",
        "ortho_init": False,
        "share_features_extractor": False,
        "normalize_images": False,
        "squash_output": False,
        "use_sde": False,
        "distribution": "DiagGaussianDistribution",
        "gaussian_mean": "tanh(action_net(bc_trunk))",
        "log_std_init": float(config["ppo"]["log_std_init"]),
        "action_order": list(ACTION_NAMES),
    })
    _write_json(directory / "protected_hashes.json", protected_before)
    commands = "\n".join((
        "python -m navigation.jackal_bc_ppo_transfer --config configs/jackal_ppo_transfer_6g1.yaml",
        "python -m pytest tests/test_jackal_bc_ppo_transfer.py tests/test_jackal_bc_architecture.py tests/test_jackal_bc_training.py tests/test_jackal_bc_evaluation.py",
        "",
    ))
    command_path = directory / "commands.txt"
    command_path.write_text(commands, encoding="utf-8")
    _fsync(command_path)
    if file_sha256(source_path) != source_hash:
        raise RuntimeError("Source BC checkpoint changed while writing the run")


def _fresh_process_check(checkpoint, observations_path, source, atol, rtol):
    code = """
import numpy as np
from stable_baselines3 import PPO
from navigation.jackal_bc_policy import load_checkpoint, predict_numpy
model = PPO.load(%(checkpoint)r, device="cpu")
bc, _payload = load_checkpoint(%(source)r, device="cpu")
observations = np.load(%(observations)r)["observations"]
predicted, _state = model.predict(observations, deterministic=True)
baseline = predict_numpy(bc, observations)
if predicted.shape != baseline.shape or not np.isfinite(predicted).all():
    raise SystemExit("fresh-process action shape or finiteness check failed")
if model.num_timesteps != 0:
    raise SystemExit("fresh-process PPO is not an initialization")
error = float(np.max(np.abs(predicted.astype(np.float64) - baseline.astype(np.float64))))
exact = bool(np.array_equal(predicted, baseline))
if %(atol)r == 0.0 and %(rtol)r == 0.0 and not exact:
    raise SystemExit("fresh-process deterministic action mismatch: %%s" %% error)
print("FRESH_RESULT", exact, error)
""" % {
        "checkpoint": str(checkpoint),
        "source": str(source),
        "observations": str(observations_path),
        "atol": float(atol),
        "rtol": float(rtol),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Fresh-process load failed: {detail}")
    result_line = ""
    for line in result.stdout.splitlines():
        if line.startswith("FRESH_RESULT"):
            result_line = line
    if not result_line:
        raise RuntimeError(f"Fresh-process load returned no result: {result.stdout}")
    _label, exact, error = result_line.split()
    return {
        "loaded": True,
        "deterministic_action_exact": exact == "True",
        "deterministic_mean_max_absolute_error": float(error),
    }


def _transfer_pairs(bc_policy, ppo_policy):
    final = audit_bc_actor_layout(bc_policy)
    actor = ppo_policy.pi_features_extractor.trunk
    pairs = []
    pairs.extend(_module_pairs(
        "lidar_encoder",
        bc_policy.lidar_encoder,
        "pi_features_extractor.trunk.lidar_encoder",
        actor.lidar_encoder,
        "lidar_encoder",
    ))
    pairs.extend(_module_pairs(
        "state_encoder",
        bc_policy.state_encoder,
        "pi_features_extractor.trunk.state_encoder",
        actor.state_encoder,
        "state_encoder",
    ))
    fusion_modules = list(bc_policy.fusion.children())
    last_index = len(fusion_modules) - 1
    if fusion_modules[last_index] is not final:
        raise ValueError("BC final linear layer is not the last fusion module")
    final_keys = {f"{last_index}.weight", f"{last_index}.bias"}
    bc_fusion = bc_policy.fusion.state_dict()
    trunk_fusion = actor.fusion_trunk.state_dict()
    if set(trunk_fusion) != set(bc_fusion) - final_keys:
        raise ValueError("Fusion trunk keys do not match the BC hidden layers")
    if set(bc_fusion) - set(trunk_fusion) != final_keys:
        raise ValueError("BC fusion contains tensors other than the trunk and action layer")
    for key in sorted(trunk_fusion):
        source = bc_fusion[key]
        destination = trunk_fusion[key]
        _check_compatible(f"fusion.{key}", source, destination)
        pairs.append(_pair(
            f"fusion.{key}",
            f"pi_features_extractor.trunk.fusion_trunk.{key}",
            source,
            destination,
            "fusion_hidden",
        ))
    action_state = ppo_policy.action_net.state_dict()
    if set(action_state) != {"weight", "bias"}:
        raise ValueError("action_net must contain weight and bias")
    for key in ("weight", "bias"):
        source = bc_fusion[f"{last_index}.{key}"]
        destination = action_state[key]
        _check_compatible(f"fusion.{last_index}.{key}", source, destination)
        pairs.append(_pair(
            f"fusion.{last_index}.{key}",
            f"action_net.{key}",
            source,
            destination,
            "action_mean_layer",
        ))
    bc_buffers = bc_policy.state_dict()
    actor_buffers = actor.state_dict()
    for key in ("obs_mean", "obs_std", "normalizer_count"):
        source = bc_buffers[key]
        destination = actor_buffers[key]
        _check_compatible(key, source, destination)
        pairs.append(_pair(
            key,
            f"pi_features_extractor.trunk.{key}",
            source,
            destination,
            "observation_normalizer",
        ))
    return pairs


def _module_pairs(source_prefix, source, target_prefix, target, group):
    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError(f"Parameter keys differ for {source_prefix}")
    pairs = []
    for key in sorted(source_state):
        value = source_state[key]
        destination = target_state[key]
        _check_compatible(f"{source_prefix}.{key}", value, destination)
        pairs.append(_pair(
            f"{source_prefix}.{key}",
            f"{target_prefix}.{key}",
            value,
            destination,
            group,
        ))
    return pairs


def _pair(source, target, source_tensor, destination, group):
    return {
        "source": source,
        "target": target,
        "group": group,
        "shape": list(source_tensor.shape),
        "elements": int(source_tensor.numel()),
        "dtype": str(source_tensor.dtype),
        "sha256": tensor_sha256(source_tensor),
        "source_tensor": source_tensor,
        "destination": destination,
    }


def _check_compatible(name, source, destination):
    if not isinstance(source, torch.Tensor) or not isinstance(destination, torch.Tensor):
        raise ValueError(f"Incompatible tensor {name}")
    if source.shape != destination.shape or source.dtype != destination.dtype:
        raise ValueError(
            f"Incompatible tensor {name}: source {tuple(source.shape)} {source.dtype} "
            f"versus target {tuple(destination.shape)} {destination.dtype}"
        )
    if not torch.isfinite(source).all():
        raise ValueError(f"Non-finite BC tensor {name}")


def _assert_actor_matches(bc_policy, ppo_policy, pairs):
    bc_state = bc_policy.state_dict()
    actor_state = ppo_policy.state_dict()
    for pair in pairs:
        source = bc_state[pair["source"]]
        restored = actor_state[pair["target"]]
        if not torch.equal(source.detach().cpu(), restored.detach().cpu()):
            raise ValueError(f"Transferred tensor is not an exact copy: {pair['source']}")
        pair["exact_match"] = True


def _install_critic_normalizer(bc_policy, ppo_policy):
    critic = ppo_policy.vf_features_extractor.trunk
    with torch.no_grad():
        critic.obs_mean.copy_(bc_policy.obs_mean.detach().to(device=critic.obs_mean.device))
        critic.obs_std.copy_(bc_policy.obs_std.detach().to(device=critic.obs_std.device))
        critic.normalizer_count.copy_(
            bc_policy.normalizer_count.detach().to(device=critic.normalizer_count.device)
        )
    if not critic.normalizer_is_ready:
        raise RuntimeError("Critic normalizer was not installed")
    if not torch.equal(critic.obs_mean.detach().cpu(), bc_policy.obs_mean.detach().cpu()):
        raise ValueError("Critic normalizer mean does not match the BC statistics")
    if not torch.equal(critic.obs_std.detach().cpu(), bc_policy.obs_std.detach().cpu()):
        raise ValueError("Critic normalizer scale does not match the BC statistics")


def _transfer_report(ppo_policy, pairs, identical, log_std_before):
    public_pairs = []
    for pair in pairs:
        public_pairs.append({
            "source": pair["source"],
            "target": pair["target"],
            "group": pair["group"],
            "role": "transferred_buffer" if pair["group"] == "observation_normalizer" else "transferred_parameter",
            "shape": pair["shape"],
            "elements": pair["elements"],
            "dtype": pair["dtype"],
            "sha256": pair["sha256"],
            "exact_match": True,
        })
    parameter_elements = sum(
        pair["elements"] for pair in public_pairs if pair["role"] == "transferred_parameter"
    )
    newly = _newly_initialized(ppo_policy, log_std_before)
    critic_normalizer = _critic_normalizer_rows(ppo_policy)
    return {
        "groups": [
            "lidar_encoder",
            "state_encoder",
            "fusion_hidden",
            "action_mean_layer",
            "observation_normalizer",
        ],
        "transferred": public_pairs,
        "transferred_parameter_tensors": sum(
            pair["role"] == "transferred_parameter" for pair in public_pairs
        ),
        "transferred_parameter_elements": parameter_elements,
        "transferred_buffer_tensors": sum(
            pair["role"] == "transferred_buffer" for pair in public_pairs
        ),
        "newly_initialized": newly,
        "newly_initialized_groups": [
            "gaussian_log_std",
            "critic_trunk",
            "value_head",
        ],
        "copied_frozen_normalizer_buffers": critic_normalizer,
        "intentionally_excluded": [
            "BC has no log_std; the diagonal-Gaussian scale is initialized separately.",
            "BC has no value head; the critic trunk and value head keep their own initialization.",
            "tanh is parameter-free and is applied to the Gaussian mean.",
            "The Stable-Baselines3 policy and value MLPs are empty.",
            "The BC optimizer state is not part of the inference checkpoint and is not copied.",
            "squash_output is false, so unscale_action is not used.",
        ],
        "critic_parameters_equal_to_actor_trunk": identical,
        "critic_parameters_unchanged_by_transfer": True,
        "log_std_unchanged_by_transfer": True,
        "optimizer_fresh": True,
    }


def _newly_initialized(ppo_policy, log_std_before):
    rows = [{
        "name": "log_std",
        "group": "gaussian_log_std",
        "role": "newly_initialized",
        "shape": list(log_std_before.shape),
        "dtype": str(log_std_before.dtype),
        "sha256": tensor_sha256(log_std_before),
        "initialization": "Stable-Baselines3 log_std_init, one value per action, not a BC tensor",
    }]
    for name, value in ppo_policy.vf_features_extractor.named_parameters():
        rows.append({
            "name": f"vf_features_extractor.{name}",
            "group": "critic_trunk",
            "role": "newly_initialized",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
            "initialization": (
                "Phase 6F.1 deterministic fan-in uniform initializer from network_config.seed. "
                "Biases start at zero. Actor weights are not copied."
            ),
        })
    for name, value in ppo_policy.value_net.named_parameters():
        rows.append({
            "name": f"value_net.{name}",
            "group": "value_head",
            "role": "newly_initialized",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
            "initialization": "PyTorch Linear default initialization because ortho_init is false",
        })
    return rows


def _critic_normalizer_rows(ppo_policy):
    critic = ppo_policy.vf_features_extractor.trunk
    rows = []
    for name in ("obs_mean", "obs_std", "normalizer_count"):
        value = critic.state_dict()[name]
        rows.append({
            "name": f"vf_features_extractor.trunk.{name}",
            "role": "copied_frozen_normalizer_buffer",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
            "exact_match_to_bc_buffer": True,
            "note": "Observation statistics, not actor weights. Applied once inside the critic trunk.",
        })
    return rows


def _identical_trunk_parameters(ppo_policy):
    actor = dict(ppo_policy.pi_features_extractor.named_parameters())
    critic = dict(ppo_policy.vf_features_extractor.named_parameters())
    if actor.keys() != critic.keys():
        raise ValueError("Critic trunk parameter names differ from the actor trunk")
    return [
        name
        for name in actor
        if torch.equal(actor[name].detach().cpu(), critic[name].detach().cpu())
    ]


def _assert_partition(ppo_policy):
    actor_ids = {id(parameter) for parameter in ppo_policy.pi_features_extractor.parameters()}
    actor_ids.update(id(parameter) for parameter in ppo_policy.action_net.parameters())
    explore_ids = {id(ppo_policy.log_std)}
    critic_ids = {id(parameter) for parameter in ppo_policy.vf_features_extractor.parameters()}
    critic_ids.update(id(parameter) for parameter in ppo_policy.value_net.parameters())
    all_ids = {id(parameter) for parameter in ppo_policy.parameters()}
    if actor_ids & critic_ids or actor_ids & explore_ids or critic_ids & explore_ids:
        raise ValueError("Actor, critic, and log_std share parameters")
    if actor_ids | explore_ids | critic_ids != all_ids:
        raise ValueError("Actor, critic, and log_std do not cover the PPO parameters")


def _parameter_snapshot(module):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _assert_parameters_unchanged(module, snapshot):
    for name, parameter in module.named_parameters():
        if not torch.equal(parameter.detach().cpu(), snapshot[name]):
            raise ValueError(f"Critic parameter changed during actor transfer: {name}")


def _assert_module_parameters_unchanged(module, snapshot, prefix):
    for name, parameter in module.named_parameters():
        key = prefix + name
        if not torch.equal(parameter.detach().cpu(), snapshot[key]):
            raise ValueError(f"Critic parameter changed during actor transfer: {key}")


def _clone_state(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _states_equal(module, snapshot):
    current = module.state_dict()
    if current.keys() != snapshot.keys():
        return False
    return all(torch.equal(current[key].detach().cpu(), snapshot[key]) for key in snapshot)


def _assert_not_trained(model, env):
    if model.num_timesteps != 0 or model._n_updates != 0 or model.rollout_buffer.pos != 0:
        raise RuntimeError("PPO learning or rollout collection started")
    unwrapped = model.get_env().envs[0].unwrapped
    if unwrapped is not env:
        raise RuntimeError("PPO replaced the Jackal space contract environment")
    if env.steps != 0 or env.resets != 0:
        raise RuntimeError("PPO stepped or reset the environment during initialization")


def _prevalidate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Unsupported Jackal BC checkpoint format")
    config = payload.get("network_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Unsupported Jackal BC checkpoint format")
    if list(payload.get("action_names") or []) != list(ACTION_NAMES):
        raise ValueError("BC action order must be [linear, angular]")
    if not _bounds_equal(payload.get("action_low"), [-1.0, -1.0]):
        raise ValueError("BC action low bound must be [-1, -1]")
    if not _bounds_equal(payload.get("action_high"), [1.0, 1.0]):
        raise ValueError("BC action high bound must be [1, 1]")
    if payload.get("action_normalization", "identity") != "identity":
        raise ValueError("BC action scaling must be identity")
    normalization = payload.get("normalization") or {}
    if list(normalization.get("feature_names") or []) != list(feature_names()):
        raise ValueError("Checkpoint feature names do not match the 40D layout")
    template = JackalBCPolicy.from_network_config(config)
    try:
        audit_bc_actor_layout(template)
        expected = template.state_dict()
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            extra = sorted(set(state) - set(expected))
            raise ValueError(f"Incompatible BC state dict: missing={missing} extra={extra}")
        for key, reference in expected.items():
            value = state[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Incompatible BC tensor {key}")
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"Incompatible BC tensor {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite BC tensor {key}")
    finally:
        del template


def _source_references(config, policy, payload, source_hash):
    source = config["source"]
    references = {
        "checkpoint": {
            "path": str(Path(source["checkpoint"]).resolve()),
            "sha256": source_hash,
        },
        "used_for_weights": True,
    }
    architecture_path = source.get("architecture")
    if architecture_path:
        document = json.loads(Path(architecture_path).read_text(encoding="utf-8"))
        if document.get("architecture") != ARCHITECTURE:
            raise ValueError("Architecture reference is not jackal_bc_v1")
        if document.get("network_config") != payload["network_config"]:
            raise ValueError("Architecture reference does not match the BC checkpoint")
        if document.get("output_activation") != "tanh":
            raise ValueError("Architecture reference output activation is not tanh")
        if list(document.get("action_names") or []) != list(ACTION_NAMES):
            raise ValueError("Architecture reference action order changed")
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        if int(document.get("parameter_count", -1)) != parameter_count:
            raise ValueError("Architecture parameter count does not match the checkpoint")
        references["architecture"] = {
            "path": str(Path(architecture_path).resolve()),
            "sha256": file_sha256(architecture_path),
            "network_config": payload["network_config"],
        }
    preprocessing_path = source.get("preprocessing")
    if preprocessing_path:
        document = json.loads(Path(preprocessing_path).read_text(encoding="utf-8"))
        normalization = document.get("observation_normalization") or {}
        scaling = document.get("action_scaling") or {}
        if scaling.get("method") != "identity":
            raise ValueError("Preprocessing action scaling is not identity")
        if list(scaling.get("order") or []) != list(ACTION_NAMES):
            raise ValueError("Preprocessing action order changed")
        if normalization.get("observation_sha256") != payload["normalization"]["observation_sha256"]:
            raise ValueError("Preprocessing observation hash does not match the checkpoint")
        mean = torch.tensor(normalization["mean"], dtype=torch.float32)
        std = torch.tensor(normalization["std"], dtype=torch.float32)
        if not torch.equal(mean, policy.obs_mean.detach().cpu()):
            raise ValueError("Preprocessing mean does not match the checkpoint normalizer")
        if not torch.equal(std, policy.obs_std.detach().cpu()):
            raise ValueError("Preprocessing scale does not match the checkpoint normalizer")
        if int(normalization["count"]) != int(policy.normalizer_count.item()):
            raise ValueError("Preprocessing count does not match the checkpoint normalizer")
        references["preprocessing"] = {
            "path": str(Path(preprocessing_path).resolve()),
            "sha256": file_sha256(preprocessing_path),
            "observation_sha256": normalization["observation_sha256"],
            "count": int(normalization["count"]),
            "method": normalization.get("method"),
            "action_scaling": "identity",
        }
    context_path = source.get("evaluation_context")
    if context_path:
        document = json.loads(Path(context_path).read_text(encoding="utf-8"))
        recorded = document.get("checkpoint_sha256")
        if recorded != source_hash:
            raise ValueError(
                "Phase 6F.3 recorded a different BC checkpoint hash than the source file"
            )
        success = document["aggregates"]["behavior_cloning"]["success"]
        references["evaluation_context"] = {
            "path": str(Path(context_path).resolve()),
            "sha256": file_sha256(context_path),
            "used_for_weights": False,
            "checkpoint_sha256_recorded": recorded,
            "checkpoint_sha256_matches_source": True,
            "behavior_cloning_success_count": success.get("count"),
            "behavior_cloning_success_attempted": success.get("attempted"),
            "behavior_cloning_success_rate": success.get("rate"),
            "note": "Phase 6F.3 closed-loop results are context only. This phase does not retrain that policy.",
        }
    return references


def _fixed_observations(count, seed):
    generator = np.random.default_rng(seed)
    observations = generator.normal(loc=0.0, scale=1.0, size=(count, 40)).astype(np.float32)
    observations[0] = 0.0
    observations[1] = 1.0
    observations[2, 36:] = np.array([0.2, -0.4, 0.1, -0.05], dtype=np.float32)
    if not np.isfinite(observations).all():
        raise RuntimeError("Validation observations are not finite")
    return np.ascontiguousarray(observations)


def _tensors_match(left, right, atol, rtol):
    error = float((left.detach() - right.detach()).abs().max().item())
    if float(atol) == 0.0 and float(rtol) == 0.0:
        return torch.equal(left, right), error
    return torch.allclose(left, right, atol=float(atol), rtol=float(rtol)), error


def _arrays_match(left, right, atol, rtol):
    error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    if float(atol) == 0.0 and float(rtol) == 0.0:
        return np.array_equal(left, right), error
    return np.allclose(left, right, atol=float(atol), rtol=float(rtol)), error


def _hash_existing(paths):
    hashes = {}
    for path in paths:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Protected artifact is missing: {candidate}")
        hashes[str(candidate.resolve())] = file_sha256(candidate)
    return hashes


def _reject_output_inside_source(output, source):
    output = Path(output).resolve()
    source = Path(source).resolve()
    if output == source or output in source.parents:
        raise ValueError("PPO output must not replace the BC checkpoint")
    checkpoint_dir = source.parent
    holds_training_artifacts = any(
        (checkpoint_dir / name).is_file()
        for name in ("best.pt", "final.pt", "preprocessing.json")
    )
    if holds_training_artifacts and (
        output == checkpoint_dir or checkpoint_dir in output.parents
    ):
        raise ValueError("PPO output must stay outside the BC checkpoint directory")


def _versions():
    return {
        "stable_baselines3": check_sb3_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "python": platform.python_version(),
    }


def _seed_everything(seed, use_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _fsync(path)


def _fsync(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _mapping(config, name):
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} configuration must be a mapping")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_number(value, name):
    number = _finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _unit_interval(value, name):
    number = _finite_number(value, name)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return number


def _bounds_equal(recorded, expected):
    if not isinstance(recorded, (list, tuple)) or len(recorded) != len(expected):
        return False
    return all(float(item) == float(target) for item, target in zip(recorded, expected))


if __name__ == "__main__":
    raise SystemExit(main())
