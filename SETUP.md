# Train VLM Distillation

## 1. Install

- Git clone

```bash
git clone https://github.com/DanhVinhLe/VLM_Distillation.git
```

- Install requirements, venv

```bash
apt update
cd VLM_Distillation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download Data

```bash
cd train_data
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d coco

wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip -d gqa

wget https://huggingface.co/datasets/DVLe/ocr_vqa/resolve/main/dataset.json
cd ocr_vqa
python loadDataset.py

wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip -d textvqa

mkdir -p vg
cd vg
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images2.zip
```

### Notes

- The first JSON file contains image paths and conversations.
- The subsequent ZIP files contain the image data.
- After downloading, the `train_data` folder structure should look like this:

```
train_data/
├── llava_v1_5_mix665k.json
├── coco/
│   └── train2017/
├── gqa/
│   └── images/
├── ocr_vqa/
│   ├── images/
│   ├── dataset.json
│   └── loadDataset.py
├── textvqa/
│   └── train_images/
└── vg/
    ├── VG_100K/
    └── VG_100K_2/
```

## 3. Train Script

- Each baseline lives in its own folder named after the baseline, with corresponding scripts that can be run sequentially.

> **Note:** Before running, adjust parameters such as **`PROJECT_DIR`**, **`NPROC_PER_NODE`**, **`MASTER_PORT`**, **`per_device_train_batch_size`**, and **`logging_step`** (increase if you want more verbose logging).

## 4. Eval

*(Ongoing)*
