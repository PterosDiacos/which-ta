import yaml
from openai import OpenAI


try_xpu_path = "/home/yc825/projects/try-xpu"


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)


def get_model():
    with open(f"{try_xpu_path}/cfg-serve.yaml") as fin:
        cfg = yaml.safe_load(fin)
        model = cfg['model']
        return model


_model = get_model()
def get_resp(**kwargs):
    resp = client.chat.completions.create(
        model=_model,
        messages=kwargs["messages"],
        temperature=kwargs.get("temperature", 0.1),
        top_p=kwargs.get("top_p", 0.9),
        max_tokens=kwargs.get("max_tokens", 100),
    )
    return resp