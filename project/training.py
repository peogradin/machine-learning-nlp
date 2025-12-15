
# %%
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# %%
from datasets import load_dataset

# %%

data = load_dataset("dair-ai/emotion", "split")
# %%
