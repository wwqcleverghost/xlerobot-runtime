import json
import tempfile
import unittest
from pathlib import Path

from prune_checkpoints import prune_checkpoints


class PruneCheckpointsTest(unittest.TestCase):
    def test_retains_three_models_latest_state_and_in_progress_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            root = run / "checkpoints"
            for step in range(20000, 100001, 20000):
                checkpoint = root / f"{step:06d}"
                model = checkpoint / "pretrained_model"
                state = checkpoint / "training_state"
                model.mkdir(parents=True)
                state.mkdir()
                for name in ("model.safetensors", "config.json", "train_config.json"):
                    (model / name).write_text("saved")
                for name in ("optimizer_param_groups.json", "raw_model.safetensors",
                             "optimizer_state.safetensors", "scheduler_state.json", "rng_state.safetensors"):
                    (state / name).write_text("saved")
                (state / "training_step.json").write_text(json.dumps({"step": step}))
            (root / "last").symlink_to("100000")
            (root / "120000").mkdir()
            (root / "notes").mkdir()
            raw = root / "100000/training_state/raw_model.safetensors"
            raw.unlink()
            prune_checkpoints(run)
            self.assertTrue((root / "020000").exists())
            raw.write_text("saved")
            prune_checkpoints(run)
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["060000", "080000", "100000", "120000", "last", "notes"],
            )
            for step in ("060000", "080000", "100000"):
                self.assertTrue((root / step / "pretrained_model/model.safetensors").is_file())
                self.assertEqual((root / step / "training_state").exists(), step == "100000")
            prune_checkpoints(run)
            self.assertTrue(raw.is_file())


if __name__ == "__main__":
    unittest.main()
