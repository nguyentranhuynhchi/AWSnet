"""Unit and CLI integration tests for AWSnet efficiency benchmarking."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.awsnet import AWSnet
from utils.efficiency import (
    BYTES_PER_GB,
    BYTES_PER_MB,
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    format_efficiency_report,
    measure_flops,
    measure_inference_efficiency,
    summarize_training_efficiency,
)


class AWSnetEfficiencyTests(unittest.TestCase):
    def test_awsnet_complexity_and_inference_cpu(self) -> None:
        model = AWSnet(in_channels=3, num_classes=4, base_filters=32, deep_supervision=True).float()

        params_info = count_parameters(model)
        self.assertEqual(params_info["total_params"], 8_729_116)
        self.assertEqual(params_info["trainable_params"], 8_729_116)
        self.assertEqual(params_info["trainable_params_excel"], "8.73M")

        flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device=torch.device("cpu"))
        self.assertGreater(flops_info["flops_total"], 8_000_000_000)
        self.assertGreater(flops_info["flops_g"], 8.0)
        self.assertLess(flops_info["flops_g"], 9.0)

        inf_info = measure_inference_efficiency(
            model, input_shape=(1, 3, 128, 128), device=torch.device("cpu"), warmup=2, repeats=3
        )
        self.assertGreater(inf_info["latency_ms_per_slice"], 0.0)
        self.assertEqual(inf_info["batch_size"], 1)
        self.assertEqual(inf_info["peak_vram_mb"], 0.0)
        self.assertEqual(inf_info["peak_vram_gb"], 0.0)

    def test_summarize_training_efficiency_skips_first_5_epochs(self) -> None:
        epoch_times = [50.0] * 5 + [8.0] * 295
        peak_bytes = int(2048 * BYTES_PER_MB)
        summary = summarize_training_efficiency(
            epoch_times_s=epoch_times,
            total_wall_clock_s=sum(epoch_times) + 60.0,
            peak_vram_bytes=peak_bytes,
            planned_epochs=300,
            warmup_epochs=5,
            peak_reserved_bytes=int(2.5 * BYTES_PER_GB),
        )
        self.assertEqual(summary["warmup_epochs_skipped"], 5)
        self.assertEqual(summary["measured_epoch_range"], "6..300")
        self.assertEqual(summary["time_per_epoch_s"], 8.0)
        self.assertAlmostEqual(summary["total_time_hours"], sum(epoch_times) / 3600.0, places=3)
        self.assertEqual(summary["peak_vram_mb"], 2048.0)
        self.assertEqual(summary["peak_vram_gb"], 2.0)
        self.assertEqual(summary["peak_vram_reserved_mb"], 2560.0)
        self.assertEqual(summary["peak_vram_reserved_gb"], 2.5)

    def test_rule_compliance_flags(self) -> None:
        ok, warnings, hw = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="none",
            num_workers=0,
            pin_memory=True,
            batch_size=16,
            planned_epochs=300,
        )
        self.assertFalse(ok)
        self.assertTrue(any("CPU" in w for w in warnings))
        self.assertEqual(hw["precision"], "fp32")

    def test_cli_train_efficiency_modes_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_dir = tmp_path / "Dataset"
            split_dir = tmp_path / "splits"
            save_dir = tmp_path / "save_run"
            eff_out = tmp_path / "eff" / "efficiency_metrics.json"
            split_dir.mkdir(parents=True)

            for mod in ("bSSFP", "LGE", "T2w"):
                (data_dir / mod / "train_npz").mkdir(parents=True)

            train_lines: list[str] = []
            for idx in range(2):
                name = f"case9999_slice{idx:03d}"
                train_lines.append(name)
                img = np.random.rand(128, 128).astype(np.float32)
                lbl = np.zeros((128, 128), dtype=np.uint8)
                lbl[40:80, 40:80] = 1
                lbl[50:60, 50:60] = 2
                lbl[60:70, 50:60] = 3
                for mod in ("bSSFP", "LGE", "T2w"):
                    np.savez_compressed(data_dir / mod / "train_npz" / f"{name}.npz", image=img, label=lbl)

            (split_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            # 1) Default run (no --measure_efficiency): should NOT write efficiency_metrics.json
            default_eff_out = tmp_path / "default_eff" / "efficiency_metrics.json"
            cmd_default = [
                sys.executable,
                str(REPO_ROOT / "train.py"),
                "--data_dir", str(data_dir),
                "--split_dir", str(split_dir),
                "--save_dir", str(tmp_path / "save_default"),
                "--batch_size", "2",
                "--max_epochs", "1",
                "--base_filters", "8",
                "--amp", "none",
                "--num_workers", "0",
                "--device", "cpu",
                "--efficiency_output", str(default_eff_out),
            ]
            r0 = subprocess.run(cmd_default, capture_output=True, text=True)
            self.assertEqual(r0.returncode, 0, msg=r0.stderr)
            self.assertFalse(default_eff_out.exists())

            # 2) --efficiency_only mode
            cmd_only = [
                sys.executable,
                str(REPO_ROOT / "train.py"),
                "--data_dir", str(data_dir),
                "--split_dir", str(split_dir),
                "--save_dir", str(tmp_path / "save_only"),
                "--batch_size", "2",
                "--max_epochs", "1",
                "--amp", "none",
                "--num_workers", "0",
                "--device", "cpu",
                "--efficiency_output", str(eff_out),
                "--efficiency_only",
            ]
            r_only = subprocess.run(cmd_only, capture_output=True, text=True)
            self.assertEqual(r_only.returncode, 0, msg=r_only.stderr)
            self.assertTrue(eff_out.is_file())
            payload = json.loads(eff_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["excel_summary"]["Trainable Params (M)"], 8.73)
            self.assertGreater(payload["excel_summary"]["FLOPs (G)"], 8.0)
            self.assertIsNone(payload["training"])

            # 3) Full 2-epoch training with --measure_efficiency -> simulate interruption -> --resume
            eff_train = tmp_path / "eff_train" / "efficiency_metrics.json"
            base_cmd = [
                sys.executable,
                str(REPO_ROOT / "train.py"),
                "--data_dir", str(data_dir),
                "--split_dir", str(split_dir),
                "--save_dir", str(save_dir),
                "--batch_size", "2",
                "--max_epochs", "2",
                "--base_filters", "8",
                "--amp", "none",
                "--num_workers", "0",
                "--pin_memory",
                "--device", "cpu",
                "--measure_efficiency",
                "--efficiency_output", str(eff_train),
            ]
            r1 = subprocess.run(base_cmd, capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)
            ckpt_file = save_dir / "awsnet_seed1234.pt"
            self.assertTrue(ckpt_file.is_file())
            self.assertTrue(eff_train.is_file())
            train_payload = json.loads(eff_train.read_text(encoding="utf-8"))
            self.assertEqual(train_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(train_payload["training"]["all_epoch_times_s"]), 2)
            self.assertTrue((save_dir / "efficiency_metrics.json").is_file())
            self.assertTrue((save_dir / "efficiency_summary_seed1234.txt").is_file())

            # Simulate interruption at epoch 1 and verify --resume preserves epoch_times
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            first_ep_time = ckpt["epoch_times"][0]
            ckpt["epoch"] = 1
            ckpt["completed_epochs"] = 1
            ckpt["training_complete"] = False
            ckpt["epoch_times"] = [first_ep_time]
            torch.save(ckpt, ckpt_file)
            eff_train.unlink()

            r2 = subprocess.run(base_cmd + ["--resume", str(ckpt_file)], capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, msg=r2.stderr)
            self.assertTrue(eff_train.is_file())
            resumed_payload = json.loads(eff_train.read_text(encoding="utf-8"))
            self.assertEqual(resumed_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(resumed_payload["training"]["all_epoch_times_s"]), 2)
            self.assertAlmostEqual(
                resumed_payload["training"]["all_epoch_times_s"][0],
                round(first_ep_time, 4),
                places=4,
            )

    def test_format_efficiency_report_contains_both_units(self) -> None:
        payload = build_efficiency_payload(
            model_name="AWSnet",
            model_id="awsnet",
            seed=1234,
            params_info={
                "total_params": 8729116,
                "trainable_params": 8729116,
                "total_params_m": 8.7291,
                "trainable_params_m": 8.7291,
                "trainable_params_excel": "8.73M",
                "total_params_excel": "8.73M",
            },
            flops_info={"flops_total": 8355512320, "flops_g": 8.3555},
            inference_info={
                "latency_ms_per_slice": 2.5,
                "warmup_runs": 30,
                "measure_runs": 100,
                "peak_vram_mb": 256.0,
                "peak_vram_gb": 0.25,
                "peak_vram_reserved_mb": 300.0,
            },
            training_info={
                "planned_epochs": 300,
                "completed_epochs": 300,
                "warmup_epochs_skipped": 5,
                "measured_epoch_range": "6..300",
                "time_per_epoch_s": 5.5,
                "total_time_seconds": 1650.0,
                "total_time_hours": 0.4583,
                "total_wall_clock_hours": 0.47,
                "peak_vram_mb": 2048.0,
                "peak_vram_gb": 2.0,
                "peak_vram_reserved_mb": 2200.0,
            },
            rule_compliant=True,
            compliance_warnings=[],
            hardware_info={
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "visible_gpu_count": 1,
                "pytorch_version": "2.4.0",
                "cuda_version": "12.1",
                "precision": "fp32",
                "amp_mode": "none",
                "num_workers": 0,
                "pin_memory": True,
                "batch_size": 16,
            },
        )
        txt = format_efficiency_report(payload)
        self.assertIn("2048.00 MB (2.0000 GB)", txt)
        self.assertIn("256.00 MB (0.2500 GB)", txt)
        self.assertIn("8.73M", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
