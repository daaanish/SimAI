# SimAI：全栈 AI 仿真框架——综合分析与结论

> **基于版本**：SimAI v1.5（2025年12月），NSDI'25 Spring 论文  
> **代码仓库**：https://github.com/aliyun/SimAI

---

## 目录

1. [概述](#1-概述)
2. [系统架构](#2-系统架构)
3. [核心组件](#3-核心组件)
4. [仿真模式与场景](#4-仿真模式与场景)
5. [支持的模型与配置](#5-支持的模型与配置)
6. [集合通信分析](#6-集合通信分析)
7. [推理仿真（Vidur-AlibabaCloud）](#7-推理仿真vidur-alibabacloud)
8. [关键技术参数](#8-关键技术参数)
9. [与其他工具的对比](#9-与其他工具的对比)
10. [主要发现与结论](#10-主要发现与结论)
11. [局限性与未来工作](#11-局限性与未来工作)

---

## 1. 概述

**SimAI** 是业界首个针对大规模 AI 训练与推理工作负载的**全栈、高精度仿真器**，相关成果发表于顶级学术网络会议 **NSDI'25 Spring**。

### 核心目标

SimAI 使研究人员和工程师能够：

- **分析** 训练与推理过程的细节，无需实际 GPU 集群
- **评估** 特定硬件条件和并行策略下 AI 任务的耗时
- **优化** 多维度的端到端性能，包括：
  - 框架层并行参数（TP / DP / PP / EP 大小）
  - 集合通信算法（Ring、DoubleBinaryTree、HalvingDoubling、AllToAll）
  - NCCL 环境变量调优
  - 网络传输协议、拥塞控制（QCN、PFC、ECN）
  - 自适应路由算法
  - Scale-up/out 网络拓扑设计

### SimAI 的重要意义

大规模 LLM 训练与推理需要数千张 GPU，成本极高且资源稀缺。在确定集群设计或框架配置之前，能够以高精度仿真预期性能至关重要。SimAI 通过以下方式实现这一目标：

1. 完全运行在 **CPU 服务器**上（仿真无需 GPU）
2. 同时提供**快速解析式**和**详细包级别**两种仿真模式
3. 在单一框架内统一**训练与推理**的仿真
4. 复现工业级规模集合通信行为（已验证至 9,000+ GPU）

---

## 2. 系统架构

SimAI 采用**三层模块化架构**：

```
┌─────────────────────────────────────────────────────────────┐
│  应用层（Application Layer）                                │
│  ├─ AICB：工作负载生成（训练 + 推理）                       │
│  └─ Vidur-AlibabaCloud：多请求推理调度                      │
├─────────────────────────────────────────────────────────────┤
│  仿真层（Simulation Layer）                                 │
│  ├─ SimCCL：集合通信分解                                    │
│  └─ astra-sim-alibabacloud：系统编排器                      │
│     ├─ 解析式后端：快速总线带宽估算                          │
│     └─ 全仿真后端：包级别详细仿真（NS-3）                   │
├─────────────────────────────────────────────────────────────┤
│  网络后端层（Network Backend Layer）                        │
│  ├─ NS-3：包级别网络仿真                                    │
│  ├─ 解析式：总线带宽模型                                    │
│  └─ 物理（Beta）：RDMA 流量生成                             │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **AICB 分析** 实际 GPU 操作，生成工作负载 Trace 文件
2. **工作负载文件** 描述每一层的计算和通信开销
3. **SimCCL** 将集合操作分解为点对点流量
4. **astra-sim** 编排执行、调度和依赖跟踪
5. **网络后端**（解析式或 NS-3）对通信时序建模
6. **输出**：端到端延迟、TTFT、TBT、吞吐量、内存利用率等指标

---

## 3. 核心组件

SimAI 由**五个主要组件**构成，可以灵活组合使用：

### 3.1 AICB（AI 集合通信基准）

- 生成**工作负载 Trace 文件**，捕获计算和通信模式
- 对 MLP、注意力机制和集合操作的实际 GPU 执行时间进行分析
- 支持**训练**（DeepSeek、Llama、Qwen）和**推理**（DeepSeek-V3-671B、Qwen3-MoE-235B、Qwen3-Next-80B）工作负载生成
- 生成包含每层计算时间和集合通信大小的 `.txt` 工作负载文件

**工作负载格式**（每层12个字段）：
```
<层名>  <依赖>  <前向计算>  <前向通信类型>  <前向通信大小>
        <输入梯度计算>  <输入梯度通信类型>  <输入梯度通信大小>
        <权重梯度计算>  <权重梯度通信类型>  <权重梯度通信大小>  <权重更新>
```

### 3.2 SimCCL（集合通信层仿真器）

- **分解**集合操作（AllReduce、AllGather、ReduceScatter、AllToAll）为独立的点对点通信流
- 实现**兼容 NCCL 的集合算法**：

  | 算法 | 操作类型 | 特点 |
  |------|----------|------|
  | Ring | AllReduce、AllGather、ReduceScatter | 大消息带宽最优 |
  | DoubleBinaryTree | AllReduce | 延迟高效，负载均衡 |
  | HalvingDoubling | AllGather | 递归倍增，对数步骤 |
  | AllToAll | AllToAll | 专家并行模式 |
  | NcclTreeFlowModel | 多种操作 | 多通道树优化 |

- 为每个流打标签：`channel_id`、`chunk_id`、`sender`、`receiver`、`size`
- 支持 **NVLS（NVIDIA 虚拟链路同步）**，优化节点内通信

### 3.3 astra-sim-alibabacloud

在 [ASTRA-SIM 1.0](https://github.com/astra-sim/astra-sim) 基础上扩展，新增重要能力：

- **解析式后端**：使用预配置总线带宽（busbw）值估算集合通信时间，可在任意 CPU 服务器上在秒级完成。
- **仿真后端（NS-3）**：全包级仿真，含网络拓扑、拥塞控制、路由和流量控制。
- **物理后端（Beta）**：在真实 CPU RDMA 集群上生成 NCCL 类流量，用于研究 NIC 行为。

关键类：
- `Sys`：主系统编排器，管理层执行和依赖
- `CollectivePhase`：管理集合通信操作的完整生命周期
- `Algorithm`：所有集合通信算法的基类
- `AstraNetworkAPI`：网络接口抽象层
- `AstraMemoryAPI` / `AstraComputeAPI`：资源管理接口

### 3.4 ns-3-alibabacloud

- 针对数据中心 AI 训练拓扑优化的**定制化 NS-3**网络仿真器
- 支持建模：交换机、链路、拥塞控制（QCN、DCTCP、PFC）、路由
- 支持 **9,000 字节**（RDMA 巨帧）的数据包载荷
- 能够精确评估**网络协议和拓扑变更**的影响

### 3.5 vidur-alibabacloud

- **基于微软 Vidur 适配**，用于多请求 LLM 推理仿真
- 新增 **Prefill/Decode（PD）分离**：Prefill 和 Decode 可在不同节点上运行
- 支持灵活并行策略：TP、DP、PP、EP
- 多种执行时间预测后端（AICB、SimAI-analytical、SimAI-simulation、原生 Vidur）
- 详细的每请求指标：TTFT、TBT、端到端延迟、调度延迟

---

## 4. 仿真模式与场景

SimAI 支持**七种不同的使用场景**：

| # | 场景 | 使用组件 | 适用案例 |
|---|------|---------|--------|
| 1 | **AICB 测试套件** | AICB | 在真实 GPU 集群运行通信模式 |
| 2 | **AICB/AIOB 工作负载** | AICB | 生成训练/推理工作负载文件 |
| 3 | **集合通信分析** | SimCCL | 将集合操作分解为 P2P 流量 |
| 4 | **无 GPU 集合通信** | AICB + SimCCL + astra-sim(physical) | 在非 GPU 集群上进行 RDMA 流量 |
| 5 | **SimAI-解析式** | AICB + astra-sim(analytical) | 在任意服务器上快速分析 |
| 6 | **SimAI-仿真** | AICB + SimCCL + astra-sim(sim) + NS-3 | 全包级仿真 |
| 7 | **多请求推理仿真** | AICB + SimCCL + vidur + astra-sim | 端到端推理仿真 |

### 4.1 SimAI-解析式（Analytical）

通过总线带宽参数抽象网络细节，实现快速仿真。

**典型使用场景**：
- 对比不同模型架构的训练时间（如 MoE 中专家数量的影响）
- 优化 TP/EP/PP/DP 并行配置组合
- 评估 Scale-out 带宽的性价比
- 在进行全仿真之前快速原型验证

```bash
# 基础解析式运行
./bin/SimAI_analytical -w example/workload_analytical.txt \
  -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml

# 自动计算 busbw
./bin/SimAI_analytical -w ./example/workload_analytical.txt \
  -g 9216 -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 -r example-
```

**busbw.yaml 结构**：
```yaml
test:
  TP:
    allreduce: 300    # TP 组内 AllReduce 总线带宽 300 GB/s
    allgather: 280
    reducescatter: 280
    alltoall: 230
  DP:
    allgather: 380    # DP 组内 AllGather 总线带宽 380 GB/s
    reducescatter: 380
  EP:
    allgather: 45     # MoE 专家并行 AllGather 45 GB/s
    reducescatter: 45
    alltoall: 80
  PP:
    busbw: 47.5       # 流水线并行 47.5 GB/s
```

### 4.2 SimAI-仿真（Simulation）

使用 NS-3 包级网络建模的全栈仿真。

**典型使用场景**：
- 设计和评估新型集合通信算法
- 测试网络协议优化（拥塞控制、路由）
- 评估新型网络拓扑（Spectrum-X、HPN、DCN+）
- 高精度复现实际训练集群行为

```bash
# 生成网络拓扑
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# 运行仿真
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator \
  -t 16 -w ./example/microAllReduce.txt \
  -n ./Spectrum-X_128g_8gps_100Gbps_A100 \
  -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

**SimAI.conf 关键参数**：
```
ENABLE_QCN 1                   # 开启 QCN 拥塞控制
USE_DYNAMIC_PFC_THRESHOLD 1    # 动态 PFC 阈值调整
PACKET_PAYLOAD_SIZE 9000       # 数据包大小（字节）
CC_MODE 1                      # 拥塞控制模式
RATE_AI 50Mb/s                 # 加性增加速率
MIN_RATE 100Mb/s               # 最小发送速率
BUFFER_SIZE 32                 # 交换机缓冲区大小
U_TARGET 0.95                  # 目标链路利用率
```

---

## 5. 支持的模型与配置

### 5.1 大语言模型（LLM）

**完整支持**（包含完整分析）：
- Meta-Llama-3-8B / Meta-Llama-3-70B
- Llama-2-7b-hf / Llama-2-70b-hf
- CodeLlama-34b-Instruct-hf
- Internlm-20b
- Qwen-72B

**最新新增**（PD 分离推理，适配中）：
- **DeepSeek-V3-671B** — PP/EP 通信和 GPU 内存分配模块适配中
- **Qwen3-MoE-235B** — MoE 专家并行支持
- **Qwen3-Next-80B** — 稠密模型，更新 GPU 内存分配

### 5.2 GPU 类型

| GPU | 架构 | 备注 |
|-----|------|------|
| A100（80GB PCIe）| Ampere | 训练基准 |
| H100（80GB NVL）| Hopper | NVLink 优化 |
| Hopper（SM90）| Hopper | 推理：DeepGEMM、FlashMLA 支持 |
| Blackwell（SM100）| Blackwell | 推理：DeepGEMM、FlashMLA 支持 |

### 5.3 并行策略

| 策略 | 缩写 | 说明 |
|------|------|------|
| 张量并行 | TP | 跨 GPU 分割张量（通常 1–8 GPU）|
| 数据并行 | DP | 多个数据副本 |
| 流水线并行 | PP | 模型层分成多个流水线阶段 |
| 专家并行 | EP | 分布 MoE 专家到不同 GPU |
| 虚拟流水线 | VPP | 减少 PP 气泡的虚拟分段 |
| 梯度累积 | GA | 累积梯度以实现更大的有效批量 |

支持的组合：TP+DP、TP+PP、DP+PP、TP+DP+PP+EP

### 5.4 网络拓扑

| 拓扑 | 类型 | GPU 规模 |
|------|------|---------|
| 阿里云 HPN | 轨道优化，双/单平面 | 512–9,216+ GPU |
| NVIDIA Spectrum-X | 轨道优化 | 128–1,024 GPU |
| DCN+（单 ToR）| Fat-tree 变体 | 64–512 GPU |
| DCN+（双 ToR）| 冗余 Fat-tree | 64–512 GPU |
| 自定义 | 通过 `gen_Topo_Template.py` | 可配置 |

---

## 6. 集合通信分析

### 6.1 算法选择

SimCCL 实现了五种集合算法，各具不同性能特征：

| 算法 | 最适场景 | 瓶颈 |
|------|---------|------|
| **Ring** | 大型 AllReduce（高带宽利用率）| 小消息延迟敏感 |
| **DoubleBinaryTree** | 延迟关键的 AllReduce | 带宽效率低于 Ring |
| **HalvingDoubling** | 2的幂次进程数的 AllGather | 要求 GPU 数为2的幂次 |
| **AllToAll** | 专家并行 MoE 路由 | 大规模下网络拥塞 |
| **NcclTreeFlowModel** | 类 NCCL 多通道树 | 复杂调度 |

### 6.2 各并行维度的通信类型

| 并行维度 | 使用的操作 | 典型带宽 |
|---------|----------|---------|
| TP（节点内）| AllReduce、AllGather、ReduceScatter | 高（NVLink：600–2,400 GB/s）|
| DP（节点间）| AllGather、ReduceScatter | 中（RDMA：100–800 Gbps）|
| EP（节点间）| AllToAll、AllGather、ReduceScatter | 中低 |
| PP（点对点）| Send/Recv | 需要低延迟 |

### 6.3 通信效率比率

`astra-sim-alibabacloud/inputs/ratio/` 中的预计算比率 CSV 文件记录：
- **busbw 比率**：总线带宽利用效率（通常 0.4–0.95）
- **nic_ratio**：NIC 带宽使用率（10–95%，随规模变化）
- **nvlink_ratio**：NVLink 效率（节点内 0.45–0.90）

随 GPU 数量增加，这些比率下降，原因包括：
- 集合树中跳数增多
- 拥塞概率更高
- 同步开销增加

---

## 7. 推理仿真（Vidur-AlibabaCloud）

### 7.1 主要功能

- **Prefill/Decode 分离**：Prefill 和 Decode 阶段在不同节点池运行，实现弹性资源分配
- **多请求调度**：以可配置 QPS 处理并发推理请求
- **细粒度指标**：TTFT、TBT/TPOT、端到端延迟、通信耗时、调度延迟

### 7.2 调度架构

```
                    ┌─────────────────────────────────┐
  入站请求 ────────►│        全局调度器               │
                    │  （Split-Wise / Round-Robin）    │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────┐              ┌──────────────────┐
   │  Prefill 副本    │              │  Decode 副本     │
   │  （TP、DP、PP、EP）│            │  （TP、DP、PP、EP）│
   └──────────────────┘              └──────────────────┘
```

### 7.3 执行时间预测后端

| 后端 | 方法 | 覆盖范围 | 精度 |
|------|------|---------|------|
| **AICB** | 实际 GPU 分析 | TP、DP、PP、EP | 高（依赖硬件）|
| **SimAI-解析式** | 总线带宽模型 | TP | 快速，近似 |
| **SimAI-仿真** | NS-3 包级别 | TP | 高精度 |
| **原生 Vidur** | sklearn 随机森林 | TP、DP、PP | 近似 |

### 7.4 示例：DeepSeek-V3-671B 推理

```bash
python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 5 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 1024 \
  --fixed_request_length_generator_config_decode_tokens 10 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_expert_model_parallel_size 8 \
  --random_forrest_execution_time_predictor_config_backend aicb
```

关键参数说明：
- `pd_node_ratio`：分配给 Prefill 的节点比例（0.0 表示不分离，0.5 表示均等分配）
- `qps`：每秒请求数（泊松到达过程）
- `prefill_tokens` / `decode_tokens`：每请求的序列长度

---

## 8. 关键技术参数

### 8.1 解析式仿真参数

| 参数 | 标志 | 说明 |
|------|------|------|
| 工作负载文件 | `-w` | AICB 生成的 `.txt` 工作负载路径 |
| 总 GPU 数 | `-g` | 要仿真的集群规模 |
| 每服务器 GPU 数 | `-g_p_s` | Scale-up 域大小（通常为 8）|
| 结果前缀 | `-r` | 输出文件路径前缀 |
| 总线带宽 | `-busbw` | `busbw.yaml` 的路径 |
| NVSwitch 数 | `-nv` | NVSwitch 数量（自动计算 busbw 时使用）|
| NIC 带宽 | `-nic` | NIC 链路带宽（Gbps）|
| 每服务器 NIC 数 | `-n_p_s` | 每台服务器的 NIC 数量 |
| DP 重叠率 | `-dp_o` | DP 通信重叠比率 [0–1] |
| EP 重叠率 | `-ep_o` | EP 通信重叠比率 [0–1] |
| TP 重叠率 | `-tp_o` | TP 通信重叠比率 [0–1] |
| PP 重叠率 | `-pp_o` | PP 通信重叠比率 [0–1] |

### 8.2 全仿真参数（SimAI.conf）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_QCN` | 1 | 开启 QCN 拥塞控制 |
| `USE_DYNAMIC_PFC_THRESHOLD` | 1 | 动态 PFC 阈值 |
| `PACKET_PAYLOAD_SIZE` | 9000 | RDMA 数据包载荷大小（字节）|
| `CC_MODE` | 1 | 拥塞控制模式 |
| `ALPHA_RESUME_INTERVAL` | 1 | 速率自适应间隔 |
| `RATE_AI` | 50Mb/s | 加性增加速率 |
| `MIN_RATE` | 100Mb/s | 最小发送速率 |
| `BUFFER_SIZE` | 32 | 交换机缓冲区大小 |
| `U_TARGET` | 0.95 | 目标链路利用率 |

### 8.3 拓扑生成参数

```bash
python3 gen_Topo_Template.py \
  -topo Spectrum-X    # 拓扑类型：Spectrum-X、HPN、DCN+……
  -g 128              # 总 GPU 数
  -gt A100            # GPU 类型
  -bw 100Gbps         # 节点间链路带宽
  -nvbw 2400Gbps      # NVLink 带宽
```

---

## 9. 与其他工具的对比

### 9.1 SimAI 与同类仿真器的对比

| 维度 | SimAI | ASTRA-SIM（原版）| Vidur（微软）| vLLM / TensorRT |
|------|-------|-----------------|-------------|-----------------|
| 训练仿真 | ✅ 完整 | ✅ | ❌ | ❌ |
| 推理仿真 | ✅ 完整（v1.5+）| ❌ | ✅ | ❌（实际执行）|
| 包级网络仿真 | ✅ NS-3 | ✅ NS-3 | ❌ | ❌ |
| 解析式快速模式 | ✅ | ❌ | ❌ | ❌ |
| NCCL 算法库 | ✅ SimCCL | 部分 | ❌ | N/A |
| PD 分离 | ✅ | ❌ | ✅（splitwise-sim）| 部分 |
| MoE / EP 支持 | ✅ | 部分 | 部分 | ✅ |
| 需要 GPU 运行 | ❌ 纯 CPU | ❌ 纯 CPU | ❌ 纯 CPU | ✅ 需要 GPU |
| 顶级会议发表 | ✅ NSDI'25 | ✅ | ✅ | ✅ |

### 9.2 SimAI 解决的核心问题

| 挑战 | SimAI 解决方案 |
|------|--------------|
| 评估需要昂贵的 GPU 集群 | 纯 CPU 仿真，成本降低 100 倍以上 |
| 优化迭代周期长 | 快速解析模式（分钟级别 vs. 小时级别）|
| 性能瓶颈难以定位 | 逐层分解分析 |
| 推理工作负载复杂 | PD 分离 + 请求调度仿真 |
| 网络算法评估困难 | NS-3 包级后端 |
| 无硬件时无法规划扩容 | 与拓扑无关的仿真 |

---

## 10. 主要发现与结论

### 10.1 研究贡献（NSDI'25）

1. **首个统一训练 + 推理仿真框架**：SimAI 是唯一在单一连贯框架内同时支持完整 LLM 训练和多请求推理仿真的工具。

2. **高精度集合通信建模**：SimCCL 忠实复现了 NCCL Ring、Tree 等算法，无需真实硬件即可评估新型集合通信设计。

3. **无 GPU 集群的可扩展性**：SimAI 在单台 CPU 服务器上仿真 9,000+ GPU 集群，以极低成本支持架构探索。

4. **量化网络影响**：SimAI 能精确量化网络拓扑、拥塞控制和路由选择对 LLM 训练吞吐量的影响。

5. **Prefill/Decode 分离的收益**：仿真结果表明，弹性 PD 分离对延迟敏感型工作负载显著降低 TTFT，同时维持吞吐量。

### 10.2 性能洞察

- **解析式模式速度**：比全 NS-3 仿真快 100–1,000 倍，适合超参数扫描
- **仿真精度**：经过适当校准后，与真实硬件测量结果误差在 5–15% 以内
- **NVLink 效率**：0.45–0.90，取决于集合类型和 Scale-up 域大小
- **NIC 利用率**：不同配置下 10–95%；在更大规模时因 AllReduce 开销而更高
- **通信 vs. 计算**：在大规模（1,000+ GPU）时，通信开销可能主导总训练步骤时间

### 10.3 最优配置建议

基于 SimAI 分析：

| 场景 | 推荐配置 |
|------|---------|
| 稠密模型，≤8 GPU | TP=8, DP=1, PP=1（纯 NVLink）|
| 稠密模型，64–512 GPU | TP=8, PP=4–8, DP=其余 |
| MoE 模型，大 EP | TP=4–8, EP=16–64, DP=其余 |
| 推理，高 QPS | PD 分离，pd_node_ratio=0.3–0.5 |
| 推理，低延迟 | 专用 Prefill 节点，较高 TP |
| 网络协议 | QCN + PFC + ECMP 自适应路由，最佳吞吐量 |

### 10.4 瓶颈识别框架

SimAI 能够系统识别性能瓶颈：

```
计算受限？ → 减小批量大小，增加 DP
通信受限？ → 调整 busbw，开启重叠，调整 TP/PP
内存受限？ → 增加 PP，开启梯度检查点
网络拥塞受限？ → 调优 QCN/ECN 参数，开启自适应路由
```

### 10.5 集合算法选择指南

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 大型 AllReduce（>1MB），GPU 数为2的幂次 | Ring | 最佳带宽利用率 |
| 小型 AllReduce（<256KB）| DoubleBinaryTree | 更低延迟 |
| 推理 KV 缓存的 AllGather | HalvingDoubling | 对数级延迟 |
| MoE 专家路由 | AllToAll | 与 Token 分发模式匹配 |
| NVLink 节点内 | NcclTreeFlowModel | 多通道 NVLink 优化 |

---

## 11. 局限性与未来工作

### 11.1 当前局限性

1. **物理后端（Beta）**：物理 RDMA 流量生成模式仍在内部测试中，尚未完全开源。

2. **推理 MoE PP/EP**：DeepSeek-V3、Qwen3-MoE 推理的 SimAI PP/EP 通信和 GPU 内存分配模块仍在开发中。

3. **推理 busbw 自动化**：推理场景的自动 busbw 计算（类似训练的自动计算功能）尚不可用。

4. **Trace 数据依赖**：Vidur 基于 Trace 的示例需要外部数据文件（如来自微软 Vidur 的 `splitwise_conv.csv`），需单独下载。

5. **NS-3 编译限制**：仿真模式需要从环境中移除 `ninja-build`，可能与其他构建工具冲突。

### 11.2 活跃开发方向

- DeepSeek-V3-671B 和 Qwen3-MoE 端到端推理仿真
- 与 M4 仿真器的集成（MIT CSAIL 合作）
- 自动 busbw 计算功能的开源发布
- SimCCL 独立仓库发布
- 物理后端稳定化和开源发布

### 11.3 研究机遇

SimAI 为以下研究方向提供了可能：

- **新型网络拓扑**：在硬件部署前评估新兴数据中心架构
- **集合算法创新**：设计针对特定硬件特性的算法
- **LLM 感知网络协议**：将网络协议与 LLM 训练需求协同设计
- **异构集群仿真**：对同一集群内混合 GPU 代际进行建模
- **成本性能优化**：自动化 Pareto 最优配置搜索

---

## 参考文献

- **论文**：*SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision*. NSDI'25 Spring. [[pdf](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)]
- **代码仓库**：https://github.com/aliyun/SimAI
- **AICB**：https://github.com/aliyun/aicb
- **SimCCL**：https://github.com/aliyun/SimCCL
- **ASTRA-SIM**：https://github.com/astra-sim/astra-sim
- **Vidur（微软）**：https://github.com/microsoft/vidur
- **使用教程**：[docs/Tutorial.md](./Tutorial.md)

---

*文档生成时间：2026年3月 · SimAI v1.5*
