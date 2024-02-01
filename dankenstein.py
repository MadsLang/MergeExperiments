import yaml
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub import HfApi
from jinja2 import Template
import os
import subprocess

HF_USERNAME = ""
OUT_MODEL_NAME = ""


### Make merge config

base_model = "mistralai/Mistral-7B-v0.1" # <- models have to be based on the same pretrained model
model1 = "HuggingFaceH4/zephyr-7b-beta" # <- finetuned on english
model2 = "danish-foundation-models/munin-7b-alpha" # <- finetuned on Danish
model3 = "timpal0l/Mistral-7B-v0.1-flashback-v2" # <- finetuned on Swedish
model4 = "NbAiLab/nb-sau-7b-4k-step100k" # <- finetuned on Norwegian bokmål

density = "0.53"
merge_methods = "dare_ties"

yaml_config = """
models:
  - model: {base_model}
    # No parameters necessary for base model
  - model: {model1}
    parameters:
      density: {density}
      weight: 0.3
  - model: {model2}
    parameters:
      density: {density}
      weight: 0.3
  - model: {model3}
    parameters:
      density: {density}
      weight: 0.2
  - model: {model4}
    parameters:
      density: {density}
      weight: 0.2
merge_method: {merge_method}
base_model: {base_model}
parameters:
  int8_mask: true
dtype: bfloat16
"""

with open('config.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_config)



### Run merge
runtime = "CPU"
trust_remote_code = False
cli_command = "mergekit-yaml config.yaml merge --copy-tokenizer"

# Additional arguments 
if runtime == "CPU":
    cli_command += " --allow-crimes --out-shard-size 1B --lazy-unpickle"
elif runtime == "GPU":
    cli_command += " --cuda --low-cpu-memory"

if trust_remote_code:
    cli_command += " --trust-remote-code"

print(f"Running cli command: {cli_command}")
subprocess.run(cli_command.split(' '))


### Make model card

template_text = """
---
license: apache-2.0
tags:
- merge
- mergekit
- lazymergekit
{%- for model in models %}
- {{ model }}
{%- endfor %}
---

# {{ model_name }}

{{ model_name }} is a merge of the following models using [mergekit](https://github.com/cg123/mergekit):

{%- for model in models %}
* [{{ model }}](https://huggingface.co/{{ model }})
{%- endfor %}

## 🧩 Configuration

\```yaml
{{- yaml_config -}}
\```
"""

# Create a Jinja template object
jinja_template = Template(template_text.strip())

# Get list of models from config
data = yaml.safe_load(yaml_config)
if "models" in data:
    models = [data["models"][i]["model"] for i in range(len(data["models"])) if "parameters" in data["models"][i]]
elif "parameters" in data:
    models = [data["slices"][0]["sources"][i]["model"] for i in range(len(data["slices"][0]["sources"]))]
elif "slices" in data:
    models = [data["slices"][i]["sources"][0]["model"] for i in range(len(data["slices"]))]
else:
    raise Exception("No models or slices found in yaml config")

# Fill the template
content = jinja_template.render(
    model_name=OUT_MODEL_NAME,
    models=models,
    yaml_config=yaml_config,
    username=HF_USERNAME,
)

# Save the model card
card = ModelCard(content)
card.save('merge/README.md')



### Upload to HF

api = HfApi(token=os.getenv("HF_TOKEN"))

api.create_repo(
    repo_id=f"{HF_USERNAME}/{OUT_MODEL_NAME}",
    repo_type="model"
)
api.upload_folder(
    repo_id=f"{HF_USERNAME}/{OUT_MODEL_NAME}",
    folder_path="merge",
)