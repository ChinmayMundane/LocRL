"""Evaluate the scalar conditioned policy with the canonical trajectory metrics."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/localizationrl-matplotlib")
from common import ROOT, build_pc_agent
import hydra
from capql.evaluate import evaluate


def load_policy(cfg, env):
    name = str(getattr(cfg, "model_checkpoint", "final_model")).removesuffix(".tar")
    path = ROOT / "models" / str(cfg.model_file) / f"{name}.tar"
    if not path.is_file():
        raise FileNotFoundError(path)
    agent = build_pc_agent(cfg, env, training=False)
    agent.load(str(path), load_replay_buffer=False)
    return agent


@hydra.main(config_path=str(ROOT / "configs"), config_name="environment", version_base="1.1")
def main(cfg):
    evaluate(cfg, policy_factory=load_policy, policy_label="PC-SAC", progress=True)


if __name__ == "__main__":
    main()
