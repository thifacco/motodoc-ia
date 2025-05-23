from huggingface_hub import hf_hub_download

hf_hub_download(

    repo_id="TheBloke/gemma-7b-it-GGUF",

    filename="gemma-7b-it.Q4_K_M.gguf",

    local_dir="./models"

)