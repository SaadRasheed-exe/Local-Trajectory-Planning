"""Backward-compatible entry point; the implementation lives in app.main."""
import hydra
from omegaconf import DictConfig

from app.main import run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
