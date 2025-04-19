# This script is set up for running localsgd on shenzhen-node2 and shenzhen-nodem

I set up my environment according to the TorchTitan README and TorchFT README.

To run, first start the lighthouse on shenzhen-nodem (node 2 is not confirmed to work): 
```bash
/srv/apps/warren/torchft/.shell_scripts/run_lighthouse.sh

Then,
```bash
On node-m:source /srv/apps/warren/torchft/.shell_scripts/run_lighthouse.sh
On node-2:source /root/warren/torchft/.shell_scripts/run_server.sh
```bash

```bash
# TorchTitan setup
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" --hf_token=...

# TorchFT setup
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# Installation from github because don't have sudo
wget https://github.com/protocolbuffers/protobuf/releases/download/v30.2/protoc-30.2-linux-x86_64.zip
unzip protoc-30.2-linux-x86_64.zip -d $HOME/.local
export PATH=$HOME/.local/bin:$PATH
pip install .
```
