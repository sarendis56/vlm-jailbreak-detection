Install Llava and dependencies:
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support
pip install -e .

pip install -r requirements.txt
```

Download the model:
```bash
python download_model.py
```

Download the datasets:
```bash
python download_datasets.py
```

Verify the setup:
```bash
python verify_setup.py
```

Run the experiments (note that you should run it *outside* of the `code` directory):
```bash
mkdir results
python code/balanced_ood_kcd.py
```
