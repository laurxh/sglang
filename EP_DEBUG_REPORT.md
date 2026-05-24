# DeepSeek-V3-AWQ EP=8 推理问题诊断报告

**日期**: 2026-05-19
**模型**: DeepSeek-V3-0324-AWQ (4-bit AWQ quantized)
**配置**: 8×H100, TP=8, EP=8
**模型路径**: `/mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/luxuhao/models/DeepSeek-V3-0324-AWQ`

---

## 一、背景与配置

### 1.1 MoE 参数

| 参数 | 值 |
|------|-----|
| n_routed_experts | 256 |
| num_experts_per_tok (topk) | 8 |
| moe_intermediate_size | 2048 |
| hidden_size | 7168 |
| routed_scaling_factor | 2.5 |
| scoring_func | sigmoid |
| norm_topk_prob | True |
| first_k_dense_replace | 3 (前3层是dense MLP) |
| moe_layer_freq | 1 (从第3层开始每层都是MoE) |
| 总层数 | 61 |

### 1.2 EP 配置

| 参数 | 值 |
|------|-----|
| tp_size | 8 |
| ep_size | 8 |
| moe_tp_size | tp_size / ep_size = 1 |
| num_local_experts | 256 / 8 = 32 |
| Expert 分配 | rank 0 → expert 0-31, rank 1 → expert 32-63, ... |

### 1.3 启动命令

```bash
python -m sglang.launch_server \
  --model-path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/luxuhao/models/DeepSeek-V3-0324-AWQ \
  --tp 8 --ep-size 8 --trust-remote-code \
  --mem-fraction-static 0.75 --cuda-graph-max-bs 16 \
  --port 30000 --disable-cuda-graph
```

---

## 二、Expert Parallelism (EP) 工作原理

### 2.1 核心思想

EP 的核心是 **"分而治之 + all-reduce 合并"**：

```
Token X 被 router 选中 8 个 expert: [3, 30, 65, 100, 130, 180, 200, 250]

Rank 0 (expert 0-31):   只计算 expert 3, 30 的贡献 → partial_output_rank0
Rank 1 (expert 32-63):  只计算 expert 65 的贡献      → partial_output_rank1
Rank 2 (expert 64-95):  只计算 expert 100 的贡献     → partial_output_rank2
Rank 3 (expert 96-127): 只计算 expert 100 的贡献     → partial_output_rank3
Rank 4 (expert 128-159):只计算 expert 130 的贡献     → partial_output_rank4
Rank 5 (expert 160-191):只计算 expert 180 的贡献     → partial_output_rank5
Rank 6 (expert 192-223):只计算 expert 200 的贡献     → partial_output_rank6
Rank 7 (expert 224-255):只计算 expert 250 的贡献     → partial_output_rank7

all-reduce(所有 rank 的 partial_output) = 完整的 MoE output
```

**"跳过非本地 expert" 是正确的** — 每个 rank 只计算自己负责的 expert，然后通过 all-reduce 把所有 rank 的部分结果加起来，得到完整结果。

### 2.2 与 Tensor Parallelism 的类比

- **TP**: 每个 rank 计算矩阵乘法的一部分列（切分隐藏维度），最后 all-reduce 合并
- **EP**: 每个 rank 计算一部分 expert 的输出（切分 expert 维度），最后 all-reduce 合并

### 2.3 all-reduce 位置

```python
# deepseek_v2.py 第 930-935 行
if self.tp_size > 1 and not should_skip_post_experts_all_reduce(...):
    final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
```

这个 all-reduce 同时服务两个目的：
1. 合并 EP 的部分 expert 结果（routed experts）
2. 合并 TP 的 shared expert 部分结果（shared_experts 的 reduce_results=False）

---

## 三、问题1: EP 非法内存访问崩溃（已修复）

### 3.1 原始错误

```
RuntimeError: CUDA error: an illegal memory access was encountered
```

### 3.2 Root Cause

**数据流**:

1. `StandardDispatcher.dispatch()` 将全局 expert ID 重映射为本地 ID:
   ```python
   # standard.py 第 174-209 行
   local_expert_mapping = torch.full((256,), -1, dtype=torch.int32)
   local_expert_mapping[rank*32 : (rank+1)*32] = torch.arange(0, 32)
   topk_ids = local_expert_mapping[topk_ids]  # 非本地→-1, 本地→0-31
   ```

2. `fused_marlin_moe()` 调用 Marlin kernel 时需要判断是否跳过 expert_id=-1 的 block:
   ```c++
   // marlin_template.h 第 382-388 行
   if (is_ep) {  // 关键判断
       for (int i = 0; i < parallel; i++) {
           if (expert_ids_ptr[i] == -1) num_valid_blocks--;
       }
   }
   ```

3. `is_ep` 的值来自 Python 层: `is_ep = expert_map is not None`（fused_marlin_moe.py 第 204 行）

4. **AWQ 的 `apply()` 方法**构建 `MarlinMoeQuantInfo` 时**没有设置 `expert_map`**:
   ```python
   # awq_kernels.py 第 244-254 行
   quant_info = MarlinMoeQuantInfo(
       w13_qweight=layer.w13_qweight,
       w2_qweight=layer.w2_qweight,
       w13_scales=layer.w13_scales,
       w2_scales=layer.w2_scales,
       w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
       w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
       w13_qzeros=layer.w13_qzeros,
       w2_qzeros=layer.w2_qzeros,
       weight_bits=self.quant_config.weight_bits,
       # 注意: 没有 expert_map！默认 None
   )
   ```

5. **结果**: `expert_map=None` → `is_ep=False` → kernel 不跳过 expert_id=-1 的 block → 用 -1 索引权重 → 非法内存访问

### 3.3 修复方案

**文件**: `python/sglang/srt/layers/moe/moe_runner/marlin.py` 第 121-128 行

```python
# 在调用 fused_marlin_moe 之前添加 EP 检测
expert_map = quant_info.expert_map
if (
    expert_map is None
    and runner_config.num_local_experts is not None
    and runner_config.num_experts is not None
    and runner_config.num_local_experts < runner_config.num_experts
):
    # 创建 dummy expert_map 来触发 is_ep=True
    expert_map = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
```

### 3.4 修复合理性分析

| 方面 | 说明 |
|------|------|
| `expert_map` 的唯一作用 | 触发 `is_ep = expert_map is not None` 布尔判断 |
| Kernel 是否使用 expert_map 内容 | 不使用，只用 `expert_ids` 数组中的 -1 判断跳过 |
| Dummy tensor 是否安全 | 安全，kernel 不解引用 expert_map |
| 副作用 | 还触发 `intermediate_cache3.zero_()`（第224行），确保跳过的 expert 输出为 0 |

### 3.5 `is_ep=True` 触发的行为

1. **Marlin kernel**: 跳过 `expert_ids[i] == -1` 的 block（不读不写）
2. **fused_marlin_moe**: 执行 `intermediate_cache3.zero_()`（第二个 GEMM 的 output buffer 清零）

---

## 四、问题2: 推理输出乱码（未解决）

### 4.1 现象

- temperature > 0: 输出随机垃圾字符
- temperature = 0: 输出全零 token (token_id=0)
- Debug 显示部分 forward pass 中 `routed_norm=0.0000`（rank 0 上）

### 4.2 已排除的原因

| # | 排查方向 | 验证方法 | 结论 |
|---|---------|---------|------|
| 1 | EP + all-reduce 数学正确性 | torchrun 8GPU 独立测试 `/tmp/test_moe_ep.py` | ✅ 正确 (relative diff < 1e-7) |
| 2 | `routed_scaling_factor` 双重应用 | 代码分析 | ✅ 不会：AWQ 路径在 `moe_sum_reduce` 内部应用一次；`forward_normal` 第887-895行对 CUDA 平台跳过 |
| 3 | 双重 all-reduce | 代码分析 | ✅ 不会：`FusedMoE.reduce_results=False`（默认值），只有 `forward_normal` 做一次 |
| 4 | 权重加载 EP 映射 | 代码分析 `_map_global_expert_id_to_local_expert_id` | ✅ 正确 |
| 5 | `moe_align_block_size` 对 -1 的处理 | CUDA kernel 源码分析 | ✅ 正确：kernel 加1（-1→slot 0），输出 `expert_ids = left-2`（→-1） |
| 6 | Marlin kernel expert_id 索引越界 | 代码分析 | ✅ 正确：output expert_ids 为 0-31，w1/w2 shape[0]=32 |
| 7 | `topk_weights` 在第二个 GEMM 中的索引 | 代码分析 | ✅ 看起来正确 |

### 4.3 核心数据流（完整 trace）

```
                    fused_marlin_moe 内部流程
                    ═══════════════════════

hidden_states [M, K=7168]
        │
        ▼
┌─ moe_align_block_size(topk_ids, block_size_m=8, num_experts=32) ─┐
│  输入 topk_ids [M, 8]: 值为 0-31（本地）或 -1（非本地）          │
│  内部映射:                                                        │
│    - CUDA kernel: expert_id = topk_ids[i] + 1                    │
│      所以 -1→0, 0→1, ..., 31→32                                  │
│    - cumsum_buffer size = num_experts+2 = 34                     │
│    - Python wrapper 传入 num_experts+1=33 给 kernel              │
│  输出:                                                            │
│    - sorted_token_ids: flat index 排序后                          │
│    - expert_ids: 每个 block 的 expert（-1=虚拟/跳过, 0-31=有效） │
│    - num_tokens_post_padded: padded 后总 token 数                │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ GEMM1: moe_wna16_marlin_gemm ─────────────────────────────┐
│  参数:                                                       │
│    输入: hidden_states [M, K]                                │
│    输出: intermediate_cache1 [M*8, 2*N]  ← torch.empty!     │
│    权重: w1 [32, 2*N_packed, K_packed]                      │
│    top_k=8, mul_topk_weights=False, is_ep=True              │
│                                                              │
│  行为:                                                       │
│    - 跳过 expert_ids=-1 的 block                            │
│    - 对有效 block: 读 hidden_states[token_row] 计算 w1@x    │
│    - 写入 intermediate_cache1[flat_idx]                      │
│    - ⚠️ 非本地 expert 对应的行保持未初始化（torch.empty）  │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ silu_and_mul(intermediate_cache1, intermediate_cache2) ─────┐
│  对 ALL M*8 行执行 SiLU(gate) * up 激活                      │
│  ⚠️ 包括未初始化的行 → cache2 对应行 = 垃圾值              │
│  （但第二个 GEMM 不会读这些垃圾行）                         │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
    intermediate_cache3.zero_()  ← 因为 expert_map is not None
        │
        ▼
┌─ GEMM2: moe_wna16_marlin_gemm ─────────────────────────────┐
│  参数:                                                       │
│    输入: intermediate_cache2 [M*8, N]                        │
│    输出: intermediate_cache3 [M*8, K] (已清零)              │
│    权重: w2 [32, K_packed, N_packed]                        │
│    top_k=1, mul_topk_weights=True, is_ep=True               │
│    size_m = M*8                                              │
│                                                              │
│  行为:                                                       │
│    - 跳过 expert_ids=-1 的 block                            │
│    - 对有效 block: 读 cache2[flat_idx] 计算 w2@x            │
│    - 乘以 topk_weights[flat_idx]                            │
│    - 写入 cache3[flat_idx]                                  │
│    - 非本地 expert 的行保持 0                               │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ moe_sum_reduce(cache3.view(M,8,K), output, 2.5) ───────────┐
│  对每个 token 的 8 个 expert slot 求和                        │
│  乘以 routed_scaling_factor=2.5                              │
│  本地 expert 贡献实际值 + 非本地 expert 贡献 0               │
│  = 该 rank 的 partial output                                 │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
    返回给 FusedMoE.forward_impl → 返回给 forward_normal
        │
        ▼
    maybe_fuse_routed_scale_and_shared_add(routed, shared, ...)
    → 对 AWQ/Marlin 路径: routed += shared（不再乘 scaling）
        │
        ▼
    tensor_model_parallel_all_reduce(final_hidden_states)
    = 合并所有 rank 的 (EP partial routed + TP partial shared)
```

### 4.4 仍在排查的可疑方向

#### 假设 A: `use_atomic_add` + `is_ep` 的交互问题

在 H100 上 `use_atomic_add=True`（compute capability >= 9）。

```c++
// marlin_template.h 第 181-184 行
use_atomic_add = (
    hidden_states.dtype == torch.half  // AWQ 用 half ✓
    or torch.cuda.get_device_capability()[0] >= 9  // H100 ✓
) and (not is_mxfp4_marlin)  // 不是 mxfp4 ✓
```

Kernel 的写入逻辑（第 584-594 行）：
```c++
if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
    // 先清零 output 对应位置，然后 atomic add
    C[sorted_row * prob_n / 8 + col] = {0, 0, 0, 0};
}
```

**潜在问题**: 当 `is_ep=True` 时 `num_valid_blocks` 减少 → `parallel` 减少 → `iters` 的计算可能导致某些有效 block 被错误处理或遗漏。

#### 假设 B: `sorted_token_ids` padding 值的影响

`moe_align_block_size` 中 `sorted_token_ids` 的 padding 值 = `numel`（即 M*topk）。在 kernel 中：
```c++
sh_rd_block_sorted_ids[threadIdx.x] = idx / top_k;
```
如果 `idx = numel = M*topk`，则 `idx / topk = M`，超出 `hidden_states` 的行数 M。

但这些 padding token 应该在 `block_num_valid_tokens` 检查中被过滤：
```c++
if (c_idx / c_gl_stride < block_num_valid_tokens) { ... }
```

#### 假设 C: Shared Expert 路径问题

- `shared_experts` 使用 TP=8 column parallel（`reduce_results=False`）
- 每个 rank 的 shared output 是完整结果的 1/8
- 通过同一个 all-reduce 与 routed output 一起合并
- 如果 shared expert 的权重加载在 EP 模式下有问题，可能导致错误

#### 假设 D: 纯 TP 也可能有问题

可能 AWQ Marlin + 该模型的组合本身有 bug，与 EP 无关。建议对比测试 `--ep-size 1`。

### 4.5 当前诊断代码

**文件**: `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`

在第二个 GEMM 前后添加了条件 debug print：
```python
# 第一个 GEMM 之后（silu_and_mul 之后）
if expert_map is not None and _debug_ep_ctr < 2 and not torch.cuda.is_current_stream_capturing():
    print(f"[DEBUG EP] M={M} topk={topk} E={E} block_size_m={block_size_m} "
          f"valid_blocks=... invalid_blocks=... "
          f"topk_ids_flat=... topk_weights=... "
          f"hidden_norm=... cache1_norm=... cache2_norm=...")

# 第二个 GEMM 之后
if expert_map is not None and _debug_ep_ctr <= 2 and not torch.cuda.is_current_stream_capturing():
    print(f"[DEBUG EP post-gemm2] cache3_norm=... cache3_nonzero=... cache3_max=...")
```

### 4.6 建议的后续排查步骤

1. **运行带 debug 的 bench_one_batch**:
   ```bash
   python -m sglang.bench_one_batch \
     --model-path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/luxuhao/models/DeepSeek-V3-0324-AWQ \
     --tp 8 --ep-size 8 --trust-remote-code --mem-fraction-static 0.75 \
     --batch-size 1 --input-len 8 --output-len 4 --correct --disable-cuda-graph
   ```
   观察 `cache3_norm` 和 `cache3_nonzero` 判断 kernel 是否产出有效结果。

2. **对比纯 TP 测试**:
   ```bash
   python -m sglang.bench_one_batch \
     --model-path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/luxuhao/models/DeepSeek-V3-0324-AWQ \
     --tp 8 --trust-remote-code --mem-fraction-static 0.75 \
     --batch-size 1 --input-len 8 --output-len 4 --correct --disable-cuda-graph
   ```
   如果纯 TP 也乱码，说明问题不在 EP 逻辑。

3. **检查 `use_atomic_add=False` 路径**: 临时强制关闭：
   ```python
   # fused_marlin_moe.py 第 181 行
   use_atomic_add = False  # 临时测试
   ```
   如果输出变正确，则问题在 atomic add + EP 的交互。

4. **单 expert 验证**: 将 `topk_ids` 全部设为 0（强制所有 token 路由到 local expert 0），看 kernel 输出是否非零。

---

## 五、文件修改清单

| 文件路径 | 修改行 | 修改内容 | 状态 |
|---------|--------|---------|------|
| `python/sglang/srt/layers/moe/moe_runner/marlin.py` | 121-128 | 添加 EP 检测，创建 dummy expert_map | **已完成（正式修复）** |
| `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` | 224+ | 添加 debug print | **临时，需清理** |
| `python/sglang/srt/models/deepseek_v2.py` | 906-921 | 之前的 debug print | **已移除** |

---

## 六、关键代码位置索引

| 功能 | 文件 | 行号 |
|------|------|------|
| StandardDispatcher.dispatch (EP remap) | `layers/moe/token_dispatcher/standard.py` | 115-218 |
| FusedMoE 构造 (num_local_experts) | `layers/moe/fused_moe_triton/layer.py` | 200-272 |
| FusedMoE.forward_impl (all-reduce 判断) | `layers/moe/fused_moe_triton/layer.py` | 1073-1098 |
| fused_marlin_moe (核心 kernel 调用) | `layers/moe/fused_moe_triton/fused_marlin_moe.py` | 51-270 |
| Marlin runner (EP fix 位置) | `layers/moe/moe_runner/marlin.py` | 76-157 |
| AWQ apply (无 expert_map) | `hardware_backend/gpu/quantization/awq_kernels.py` | 236-255 |
| moe_align_block_size Python wrapper | `layers/moe/moe_runner/triton_utils/moe_align_block_size.py` | 50-87 |
| moe_align_block_size CUDA kernel | `jit_kernel/csrc/moe/moe_align_kernel.cu` | 55-250 |
| Marlin kernel (EP skip logic) | `jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` | 382-388, 510-528 |
| Marlin kernel (atomic add write) | `jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` | 584-594 |
| Marlin kernel (topk_weights read) | `jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` | 489-501 |
| DeepseekV2MoE.forward_normal | `models/deepseek_v2.py` | ~830-940 |
| maybe_fuse_routed_scale_and_shared_add | `layers/quantization/mxfp4_flashinfer_trtllm_moe.py` | 437-469 |
| moe_sum_reduce | `sgl_kernel` (compiled) | - |

---

## 七、EP 数学验证测试

独立验证脚本 `/tmp/test_moe_ep.py`，用 `torchrun --nproc_per_node=8` 运行：
- 模拟 256 experts, EP=8, topk=8
- 每个 rank 只计算本地 expert，然后 all-reduce
- 与 rank 0 上的全量计算对比
- 结果: relative diff < 1e-7，**证明 EP + all-reduce 逻辑数学上是正确的**

---

## 八、总结

| 问题 | 状态 | Root Cause |
|------|------|-----------|
| EP 非法内存访问 | ✅ 已修复 | AWQ 未设置 expert_map → Marlin kernel 不跳过 -1 expert → 越界访问 |
| CUDA graph OOM | ✅ 已修复 | 降低 mem-fraction-static=0.75, cuda-graph-max-bs=16 |
| 推理输出乱码 | ❌ 未解决 | 待定，最可能是 Marlin kernel 在 `use_atomic_add + is_ep` 下的写入逻辑问题，或 shared expert TP 路径问题 |
