import sys
from pathlib import Path

import torch

disc_path = Path(
    sys.argv[1]
)  # output/deberta-v3-base-continue-2024-02-05-16-00/RTD/discriminator/pytorch.model-100000.bin
disc_path = Path(
    sys.argv[1]
)  # output/deberta-v3-base-continue-2024-02-05-16-00/RTD/discriminator/pytorch.model-100000.bin
config_path = Path("experiments/language_model/rtd_base.json")
disc_ckpt = torch.load(disc_path, map_location="cpu")
state_dict = disc_ckpt["state_dict"]
torch.save(state_dict, "pytorch_model.bin")
# with open("pytorch_model.bin", mode='wb') as f:
#     f.write(disc_ckpt['state_dict']

# config = ModelConfig.from_json_file(config_path)
