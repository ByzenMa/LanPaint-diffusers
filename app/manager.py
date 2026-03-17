import os
import yaml



class ConfigManager:
    def __init__(self, config_path: str = "./config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @property
    def sd1_5_path(self):
        return self.config["model"]["sd1_5_path"]

    @property
    def controlnet_canny_path(self):
        return self.config["model"]["controlnet_canny_path"]

    # partial injection ranges
    @property
    def r_channel_start(self):
        return float(self.config["control_injection"]["r_channel_start"])

    @property
    def r_channel_end(self):
        return float(self.config["control_injection"]["r_channel_end"])

    @property
    def g_channel_start(self):
        return float(self.config["control_injection"]["g_channel_start"])

    @property
    def g_channel_end(self):
        return float(self.config["control_injection"]["g_channel_end"])

    @property
    def b_channel_start(self):
        return float(self.config["control_injection"]["b_channel_start"])

    @property
    def b_channel_end(self):
        return float(self.config["control_injection"]["b_channel_end"])

    # inference
    @property
    def num_inference_steps(self):
        return int(self.config["inference"]["num_inference_steps"])

    @property
    def guidance_scale(self):
        return float(self.config["inference"]["guidance_scale"])

    @property
    def seed(self):
        return int(self.config["inference"]["seed"])



