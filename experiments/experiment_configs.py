"""Named experiment presets used by run_all.py."""
SMOKE = {"scale": 0.05, "seeds": [0], "label": "smoke"}
FULL = {"scale": 1.0, "seeds": None, "label": "full"}   # seeds=None -> config.yaml seeds
