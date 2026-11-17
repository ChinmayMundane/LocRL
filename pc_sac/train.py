"""Train PC-SAC under the CAPQL collection and evaluation protocol."""
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/localizationrl-matplotlib")
from common import ROOT, build_pc_agent
import hydra
from omegaconf import OmegaConf
from capql.train import run_training


@hydra.main(config_path=str(ROOT / "configs"), config_name="environment", version_base="1.1")
def main(cfg):
    directory = ROOT / "models" / str(cfg.model_file)
    if (directory / "final_model.tar").exists():
        raise FileExistsError(f"Completed model already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"method": "preference_conditioned_scalar_sac", "critic_outputs": 1,
                "reward_scale": 1.0, "config": OmegaConf.to_container(cfg, resolve=True)}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run_training(cfg, agent_factory=build_pc_agent, method_label="PC-SAC")


if __name__ == "__main__":
    main()
