# Paper training settings

The public entry points expose training choices as ordinary command-line
arguments. They do not select hidden dataset-specific recipes.

The retained distillation settings were:

Example:

```bash
python -m configs.train_distillation \
  --train 'knowledge_edit/fictbio/train/**/*.xml' \
  --validation 'knowledge_edit/fictbio/validation/**/*.xml' \
  --dataset fictbio \
  --base-model llama3.1-70b \
  --run-name fictbio-llama3.1-70b \
  --checkpoint-interval 240 \
  --val-interval 320 \
  --log-interval 1
```
