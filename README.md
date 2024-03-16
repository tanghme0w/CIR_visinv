# VisInv4CIR
## Data Preparation
```text
.
├── clip-vit-large-patch14
├── data
│    └── fashion-iq
│           ├── images
│           ├── image_splits
│           └── json
├── third_party
│
...<scripts>

```
## Run
### train
```bash
python main.py --gpu 0 --model ViT-L/14 --source-data dress --batch-size 512
```
- gpu: the index of cuda device, omit this param if you prefer to use cpu.
- source_data: in ['dress', 'shirt', 'toptee'].
### eval
```bash
python evaluation.py --gpu 0 --source-data dress --batch-size 64
```