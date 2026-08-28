# PyTorch Native Tensor Parallel (TP) and FSDP for Distributed Training

## Requirements
- `transformers >= 4.41.2`

## File Structure
```plaintext
.
├── training
│   ├── tensor_parallelism
│   │   ├── configs: Configs for model loading, LoRa, and FSDP
│   │   ├── datasets: Custom dataset classes (e.g., LLamaDataset)
│   │   ├── hooks: Forward hooks for debugging purposes (legacy code, not used)
│   │   ├── input_tensors: Sample input tensors in proper shape and format for testing
│   │   ├── archs: Llama models' architecture configs for fast lookup (not loaded or used in code)
│   │   ├── patching: [IMPORTANT] Monkey patching files to overwrite crucial methods/functions
│   │   ├── policies: FSDP mixed_precision settings and wrapper policies
│   │   ├── logs: Training losses and times data (refer to [README](logs/README.md) for more details)
│   │   ├── utils: Utility functions
│   │   ├── *.py: Python files (e.g., run_fsdp_tp_llama3.py)
│   │   └── training.sbatch: Sample Slurm script to allocate 2 nodes, each with 2 GPUs and init a training on multiple nodes with FSDP+TP applied.
│   │   └── tunnel.sbatch: Sample Slurm script to allocate 2 nodes, each with 4 GPUs.
```

**Note**: During runs, `checkpoints` and `out` folders may be created by the code, or you may need to create them manually to store checkpoint files or output files.

## Modifying Experiment Configurations
Experiment configurations can be set by modifying `exp_config`, which is located in the `configs/model.py` file. You can modify:
- `model_id`: HF model you want to load (tested on "meta-llama/Meta-Llama-3-8B-Instruct" and "meta-llama/Meta-Llama-3-70B-Instruct")
- `layer_name`: Specifies the layer whose weights are saved (used only when `save_weights` is set to 1 in the scripts below)
- `mixed_precision`: Whether to use mixed precision policy in FSDP. **Default**: `True`
- `use_fp16`: Whether to use `float16` (always set to `False` as we use `bfloat16`). **Default**: `False`
- `num_hidden_layers`: Number of hidden layers of the model for debugging purposes. Set to `None` to load the full model. **Default**: `None`
- `torch_dtype`: Model dtype (i.e. `bf16` or `f32`). **Default**: `bf16`
- `gradient_checkpointing`: Enable gradient checkpointing. **Default**: `True`
- `first_run`: Whether this is the first run (used only when `sharded_load` is set to `True`). **Default**: `True`. **Explanation**: When the model is too large to fit on a single GPU, we only load the necessary sharded portions of the full weights that each GPU requires to compute the output. Initially, we do not have a distributed state dictionary. Thus, in the first run, we need to load the entire model into CPU memory first, then distribute its weights to each GPU. These distributed weights are saved as what PyTorch refers to as a **distributed state dictionary**. From the second run, we only need to load this distributed state dicts into sharded models.
- `save_and_load_optimizer_state`: Whether to also save/load the optimizer state. **Default**: `True`
- `use_peft`: Whether to use the PEFT model or the base model. **Default**: `True`
- `sharded_load`: Whether to load only necessary portions of weights onto GPUs to save memory. **Default**: `True`
- `reset_adapters`: Whether to force create a new adapter even when there's already saved adapters at `adapter_dir`. **Default**: True
- `save_adapters`: Whether to save adapters at given `adapter_dir`
- `world_size`: Universally total number of GPUs used for training (not number of GPUs per node). **Default**: 8
- `tp_size`: TP dimension size. **Default**: 1
- `seed`: Random seed to initialize the samplers for Training and Validation Data. **Default**: 42
- `node_id`: Index of the current node (0-indexed). **Default**: 0
- `host_node`: Name of the host node (i.e. head node). This must be specified in multi-node training is used. **Default**: None
- `ip`: IP address of the host node. This must be specified in multi-node training is used. **Default**: None
- `multi_node`: Whether doing multi-node training. **Default**: `False`
- `use_fsdp_only`: bool or int. This is used to force FSDP-only training. Set to True or 1 to use FSDP1; set to 2 to use FSDP2. **Default**: True

## Running Experiments
Each experiment Python script contains code to perform model inference and training using different parallelism techniques, including TP only and a combination of FSDP and TP. Specifically:
- `run_tp_llama3.py`: Runs inference and training of Llama 3 models with TP applied to the model.
- `run_fsdp_tp_llama3.py`: Runs inference and training of Llama 3 models with FSDP+TP applied, i.e., 2D parallelism with TP applied across the TP dimension and FSDP across another dimension.
- `run_single_llama3.py`: Runs inference and training of Llama 3 models on a single GPU with no parallelism applied.

### 0. Arguments Breakdown:
The following is a breakdown of the arguments for the `run*.py` files:

- `-m`, `--multi_node`: Specifies whether to run on multiple nodes.
- `-w`, `--world_size`: Total number of GPUs.
- `--ngpus`: Number of GPUs on each node.
- `-s`, `--save_training_logs`: Specifies whether to save training logs, including training times and graphs of training losses, and their values in a numpy array.
- `-sw`, `--save_weights`: Specifies whether to save the weights specified by `layer_name` in `exp_config`.
- `-n`, `--num_epochs`: Number of epochs. Not necessary if `--train` is set to 0.
- `-p`, `--population`: Number of training runs. For example, if `-p 2`, the training will be performed twice with the same settings, each with `--num_epochs`. This is used to aggregate the training logs for more reliable benchmarks. Not necessary if `--train` is set to 0.
- `-nid`, `--node_id`: Node ID (not necessary when training on a single GPU, i.e., the `-m` flag is not set). **IMPORTANT**: The node ID should always be 0 on the host node (the first node in the NODELIST).
- `-hn`, `--host_node`: Name of the host node. This argument is used to manually specify the host node instead of determining it automatically by inspecting `$WORK_DIR/nodelist.txt`. This argument is used in the Slurm training script as the host node is known within the context of the Slurm script. For more information, refer to section `2. Running on Multiple Nodes` below. **Default**: None.
- `-t`, `--train`: Specifies whether the run is for training or inference. `1` for training and `0` for inference.
- `-a`, `--adapter_dir`: Path to the directory containing adapter weights. **Default**: `adapters`.
- `-an`, `--adapter_name`: Name of the adapter. This is currently not utilized as the code only works when `adapter_name` is `default`. Please use `adapter_dir` only.

### 1. Running on a Single Node
#### Tensor Parallel (TP) Only
Example:
```bash
python run_tp_llama3.py -w 4 --ngpus 4 -s 1 -sw 1 -n 2 -p 1 -t 1 -a adapters/test
```
**Note**: `w` and `ngpus` has to be the same when running on single node since world size = number of gpus per node
#### FSDP + TP
Example:
```bash
python run_fsdp_tp_llama3.py -w 4 --ngpus 2 -s 1 -sw 1 -n 2 -p 1 -t 1 -a adapters/test
```
**Note**: When running on single node:
- `w` is the number of gpus on the single node.
- `--ngpus` is the number of gpus in each DP group (i.e. Size of TP group)

### 2. Running on Multiple Nodes
After running the tunneling Slurm script `tunnel.sbatch` to allocate 2 nodes, the NODELIST is saved as a text file at `$WORK_DIR/nodelist.txt` for secondary nodes to retrieve the IP of the host node.

**NOTE**: We rely on this method when conducting experiments because the `NODELIST` environment variable is not retained in sessions after the Slurm script finishes. An example of a Slurm script for formal training, submitted as an sbatch job, is provided in the file `training.sbatch`.

#### Tensor Parallel (TP) Only
Example:
On node 0:
```bash
python run_tp_llama3.py -m -w 8 --ngpus 4 -s 1 -sw 1 -n 2 -p 1 -nid 0 -t 1 -a adapters/test
```
On node 1:
```bash
python run_tp_llama3.py -m -w 8 --ngpus 4 -s 1 -sw 1 -n 2 -p 1 -nid 1 -t 1 -a adapters/test
```

**NOTE**: Run this command on every node, assigning a unique `nid` value to each node. `-nid 0` must be set at the host node (the first node in `NODELIST`). Other nodes' indices can be chosen freely.

#### FSDP + TP
Example:
Only node 0:
```bash
python run_fsdp_tp_llama3.py -m -w 8 --ngpus 4 -s 1 -sw 1 -n 2 -p 1 -nid 0 -t 1 -a adapters/test
```
Only node 1:
```bash
python run_fsdp_tp_llama3.py -m -w 8 --ngpus 4 -s 1 -sw 1 -n 2 -p 1 -nid 1 -t 1 -a adapters/test
```

**NOTE**: Similarly, run this command on each node, using appropriate `nid` values for each.

## Testing the Inference Output and Parameter Updates (Forward/Backward Pass Validation)
### 1. Procedure:
- Run inference/training on single GPU using the given Python script.

    **NOTE**: like every other experiment, the experiment configuration is also set through `exp_config` located in the `configs/model.py` file.
- Run inference/training on multiple GPUs using multi-GPU Python scripts.
- Run `compare_output.py` to compare the weights/inference outputs.

    **NOTE**: The file `compare_output.py` contains sample code to use three functions for comparing output tensors or parameter updates. There are three functions:
    - `compare_outputs`: Given two tensors, this function checks whether they are exactly the same. If not, it calculates more detailed statistics of the two tensors (e.g., Top-k indices matching).
    - `compare_perp_llama3`: Given two output logits and the true labels, this function calculates the perplexity of each logit and makes a comparison.
    - `compare_weights`: Given two directories containing pretrained and trained weights, this function calculates the cosine similarity of the update vectors of the weight of the two models (e.g., the update vector of the weight of `model.embed_tokens` layer trained with FSDP+TP and the one trained on a single GPU). For more details, refer to the function itself.

### 2. Examples:
#### Example 1: Checking the correctness of the forward pass of a Llama 3-8B model (i.e. compare inference outputs)
- **Step 0**: Setup experiment configuration. Example config:
    ```
    model_id: str="meta-llama/Meta-Llama-3-8B-Instruct"
    mixed_precision: bool=True
    use_fp16: bool=False
    num_hidden_layers: int=None
    torch_dtype: str='bf16' # 'bf16' or 'f32'
    gradient_checkpointing: bool=True
    first_run: bool=True
    save_and_load_optimizer_state: bool=True
    save_adapters: bool=False
    use_peft: bool=False
    sharded_load: bool=True
    reset_adapters: bool=True
    ```
- **Step 1**: Run inference on single GPU:

    Command:
    ```bash
    python run_single_llama3.py -t 0
    ```
    Output:
    ```
    Output saved at "out/single/llama3_single.pt"
    ```

- **Step 2**: Run inference on multi-GPU. For example, FSDP+TP run on 2 nodes, each of which contains 4 GPUs:

    Command:
    - On node 0:
    ```bash
    python run_fsdp_tp_llama3.py -m -w 8 --ngpus 4 -t 0 -nid 0
    ```
    - On node 1:
    ```bash
    python run_fsdp_tp_llama3.py -m -w 8 --ngpus 4 -t 0 -nid 1
    ```
    Output:
    ```
    Output saved succesfully at out/fsdp_tp/2n4g/llama3_fsdp_tp.pt
    ```

- **Step 3**: Compare two outputs. In `compare_outputs.py` file, call two functions:
    ```
    compare_outputs(
        input_path_1="out/fsdp_tp/2n4g/llama3_fsdp_tp.pt",
        input_path_2="out/single/llama3_single.pt",
        device='cuda:0',
        atol=1e-8,
        top_k=5,
    )

    compare_perp_llama3(
        input_path="input_tensors/fsdp/llama3_input_padded.pt",
        output_path_1="out/fsdp_tp/2n4g/llama3_fsdp_tp.pt",
        output_path_2="out/single/llama3_single.pt",
        ignore_index=128009,
        device='cuda:0',
    )
    ```

- **Step 4**: Inspecting the returned outputs to validate the correctness of the multi-GPU inference output. For example, in the following output, we could safely declare that the forward pass is performed correctly as the MAE is acceptably small (i.e. 0.02498) and the top-K matching is high.

    ```
    Inference output is incorrect for tensor 0
    SSE in tensor 0 is 650777.125
    Percentage of different entries: 87.84%
    Average error: 0.02498
    Average entry value: torch.mean(tensor1).item()=0.17541912198066711 torch.mean(tensor2).item()=0.17531761527061462
    ---------
    Number of differences: 461478428
    ----------------- Top-5 -----------------
    Top-5 indices do not match
    Number of matching top-5 indices: 17535
    Percentage of matching top-5 indices: 85.62%
    MAE for top-5 logits: 0.02986
    ----------------- Top-4 -----------------
    Top-4 indices do not match
    Number of matching top-4 indices: 14525
    Percentage of matching top-4 indices: 88.65%
    MAE for top-4 logits: 0.03029
    ----------------- Top-3 -----------------
    Top-3 indices do not match
    Number of matching top-3 indices: 11254
    Percentage of matching top-3 indices: 91.59%
    MAE for top-3 logits: 0.03084
    ----------------- Top-2 -----------------
    Top-2 indices do not match
    Number of matching top-2 indices: 7697
    Percentage of matching top-2 indices: 93.96%
    MAE for top-2 logits: 0.03129
    ----------------- Top-1 -----------------
    Top-1 indices do not match
    Number of matching top-1 indices: 3958
    Percentage of matching top-1 indices: 96.63%
    MAE for top-1 logits: 0.03233

    ...
    Perplexity for single GPU run: 224463.71875
    Perplexity for multi-GPU run: 224141.984375
    Difference in perplexity values: 321.734375 (i.e. 0.14%)
    ```

#### Example 2: Checking the correctness of a parameter update of a PEFT Llama 3-8B model (i.e. calculate the cosine similarity of a particular weight when running the training on single vs multiple GPUs)

- **Step 0**: Setup experiment configuration.
    ```
    layer_name: str='base_model.model.model.layers.0.self_attn.v_proj.lora_B.default' # Layer whoses weights are to be saved
    first_run: bool=True
    save_and_load_optimizer_state: bool=True
    save_adapters: bool=False
    use_peft: bool=True
    sharded_load: bool=True
    reset_adapters: bool=True
    ```
    Here we set `first_run` to True as this is the first time this model is shardedly loaded and saved. Also we force the `reset_adapters` to True to freshly initialize the adapters' weights.

    **NOTE**: By default, when `reset_adapters=False`, the model would load pretrained adapters weights from `adapter_dir`, which is by default `adapters` folder. Please make sure there are pretrained weights (i.e. non-zero weights) in that folder. If not, the LoRA weights will be freshly initialized with zero weights on either of the two LoRA matrices in each layer.

    For debugging purposes, there is a monkey patching function provided that overwrites how peft initializes zero weights. The given function `reset_lora_parameters` modifies the original function to always initialize non-zero LoRA weights. By default, it is not activated. You can activate it by uncommenting the line that overwrites that function in the file `patching/peft_patching.py`. Remember to deactivate it when running formal training.

- **Step 1**: Run training on single GPU:

    Command:
    ```bash
    python run_single_llama3.py -s 1 -n 1 -t 1 -sw 1
    ```
    Output:
    ```
    Weight of base_model.model.model.layers.0.self_attn.v_proj.lora_B.default saved at iteration 0 at out/weight/base_model.model.model.layers.0.self_attn.v_proj.lora_B.default/weight_0_eps.pt

    Weight of base_model.model.model.layers.0.self_attn.v_proj.lora_B.default saved at iteration 1 at out/weight/base_model.model.model.layers.0.self_attn.v_proj.lora_B.default/weight_1_eps.pt
    ```

- **Step 2**: Run training on multi-GPU. For example, FSDP+TP run on 2 nodes:

    Command:
    - On node 0:
    ```bash
    python run_fsdp_tp_llama3.py -m -w 8 --ngpus 4 -s 1 -sw 1 -n 1 -p 1 -t 1 -nid 0
    ```
    - On node 1:
    ```bash
    python run_fsdp_tp_llama3.py -m -w 8 --ngpus 4 -s 1 -sw 1 -n 1 -p 1 -t 1 -nid 1
    ```
    Output:
    ```
    Full weight saved sucessfully at out/fsdp_tp/weight/base_model.model.model.layers.0.self_attn.v_proj.lora_B.default/2n4g/weight_fsdp_tp_0_eps.pt

    Full weight saved sucessfully at out/fsdp_tp/weight/base_model.model.model.layers.0.self_attn.v_proj.lora_B.default/2n4g/weight_fsdp_tp_1_eps.pt
    ```

- **Step 3**: Compare update vectors. In `compare_outputs.py` file, call the following function:
    ```
    layer_name = 'base_model.model.model.layers.0.self_attn.v_proj.lora_B.default'
    parallel_mode = 'fsdp_tp'
    compare_weights(
            tp_model_dir=f'out/{parallel_mode}/weight/{layer_name}/2n4g',
            single_model_dir=f'out/weight/{layer_name}',
            layer_name=layer_name,
            parallel_mode=parallel_mode,
            epoch=1,
        )
    ```

- **Step 4**: Inspecting the returned cosine similarity to validate the correctness of the multi-GPU parameter update. For example, in the following output, we can safely declare that the backward pass is performed correctly as the cosine similarity is close to 1 (i.e. 0.984375).
    ```
    ----------------- Comparing weights -----------------
    Weigth shape: torch.Size([1024, 512])
    Weight mismatch for base_model.model.model.layers.0.self_attn.v_proj.lora_B.default
            ------ Before training ------
    MAE in base_model.model.model.layers.0.self_attn.v_proj.lora_B.default is 0.0
    Percentage of different entries before training: 0.00%
    Average entry value before training: 0.0
    Average entry value before training: 0.0
            ------ After training for 1 epochs ------
    MAE in base_model.model.model.layers.0.self_attn.v_proj.lora_B.default is 1.885928213596344e-08
    Percentage of different entries after training: 35.76%
    Average entry value trained: -1.9208528101444244e-09
    Average entry value trained: -2.2992026060819626e-09
    Cosine similarity between update vectors: 0.984375
    ```
