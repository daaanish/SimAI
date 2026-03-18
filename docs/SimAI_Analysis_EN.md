# SimAI: Full-Stack AI Simulation Framework — Comprehensive Analysis and Conclusions

> **Based on**: SimAI v1.5 (December 2025), NSDI'25 Spring Publication  
> **Repository**: https://github.com/aliyun/SimAI

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Core Components](#3-core-components)
4. [Simulation Modes and Scenarios](#4-simulation-modes-and-scenarios)
5. [Supported Models and Configurations](#5-supported-models-and-configurations)
6. [Collective Communication Analysis](#6-collective-communication-analysis)
7. [Inference Simulation (Vidur-AlibabaCloud)](#7-inference-simulation-vidur-alibabacloud)
8. [Key Technical Parameters](#8-key-technical-parameters)
9. [Comparison with Other Tools](#9-comparison-with-other-tools)
10. [Key Findings and Conclusions](#10-key-findings-and-conclusions)
11. [Limitations and Future Work](#11-limitations-and-future-work)

---

## 1. Overview

**SimAI** is the industry's first **full-stack, high-precision simulator** for large-scale AI training and inference workloads, published at **NSDI'25 Spring** — one of the top-tier academic networking conferences.

### Core Purpose

SimAI enables researchers and engineers to:

- **Analyze** training and inference process details without needing actual GPU clusters
- **Evaluate** the time consumption of AI tasks under specific hardware and parallelism conditions
- **Optimize** end-to-end performance across multiple dimensions:
  - Framework-level parallel parameters (TP / DP / PP / EP sizes)
  - Collective communication algorithms (Ring, DoubleBinaryTree, HalvingDoubling, AllToAll)
  - NCCL environment variables and tuning
  - Network protocols, congestion control (QCN, PFC, ECN)
  - Adaptive routing algorithms
  - Scale-up/out network topology design

### Why SimAI Matters

Large-scale LLM training and inference require thousands of GPUs, which are extremely expensive and scarce. Before committing to a cluster design or framework configuration, being able to simulate the expected performance with high fidelity is critical. SimAI makes this possible by:

1. Running entirely on **CPU servers** (no GPU required for simulation)
2. Providing **both fast analytical** and **detailed packet-level** simulation modes
3. Unifying **training and inference** simulation in a single framework
4. Reproducing industry-scale collective communication behavior (validated up to 9,000+ GPUs)

---

## 2. Architecture

SimAI uses a **three-layer, modular architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                          │
│  ├─ AICB: Workload generation (training + inference)        │
│  └─ Vidur-AlibabaCloud: Multi-request inference scheduling  │
├─────────────────────────────────────────────────────────────┤
│  SIMULATION LAYER                                           │
│  ├─ SimCCL: Collective communication decomposition          │
│  └─ astra-sim-alibabacloud: System orchestrator             │
│     ├─ Analytical Backend: Fast bus bandwidth estimation    │
│     └─ Full Simulation Backend: Packet-level detail (NS-3)  │
├─────────────────────────────────────────────────────────────┤
│  NETWORK BACKEND LAYER                                      │
│  ├─ NS-3: Packet-level simulation                           │
│  ├─ Analytical: Bus bandwidth model                         │
│  └─ Physical (Beta): RDMA traffic generation                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **AICB profiles** actual GPU operations and produces workload trace files
2. **Workload files** describe each layer's compute and communication costs
3. **SimCCL** decomposes collective operations into point-to-point flows
4. **astra-sim** orchestrates execution, scheduling, and dependency tracking
5. **Network backend** (analytical or NS-3) models communication timing
6. **Output**: End-to-end latency, TTFT, TBT, throughput, memory utilization metrics

---

## 3. Core Components

SimAI is composed of **five main components** that can be combined in various ways:

### 3.1 AICB (AI Collective Benchmark)

- Generates **workload trace files** capturing computation and communication patterns
- Profiles real GPU execution for MLP, attention, and collective operations
- Supports both **training** (DeepSeek, Llama, Qwen) and **inference** (DeepSeek-V3-671B, Qwen3-MoE-235B, Qwen3-Next-80B) workload generation
- Produces `.txt` workload files with per-layer compute times and collective communication sizes

**Workload format** (per layer, 12 fields):
```
<name>  <dep>  <fp_compute>  <fp_comm>  <fp_size>
        <ig_compute>  <ig_comm>  <ig_size>
        <wg_compute>  <wg_comm>  <wg_size>  <wg_update>
```

### 3.2 SimCCL (Collective Communication Layer Simulator)

- **Decomposes** collective operations (AllReduce, AllGather, ReduceScatter, AllToAll) into individual point-to-point communication flows
- Implements **NCCL-compatible collective algorithms**:

  | Algorithm | Operation | Characteristics |
  |-----------|-----------|-----------------|
  | Ring | AllReduce, AllGather, ReduceScatter | Bandwidth-optimal for large messages |
  | DoubleBinaryTree | AllReduce | Latency-efficient, load-balanced |
  | HalvingDoubling | AllGather | Recursive doubling, logarithmic steps |
  | AllToAll | AllToAll | Expert parallelism patterns |
  | NcclTreeFlowModel | Multiple | Multi-channel tree optimization |

- Tags each flow with: `channel_id`, `chunk_id`, `sender`, `receiver`, `size`
- Supports **NVLS (NVIDIA Virtual Link Synchronization)** for intra-node optimization

### 3.3 astra-sim-alibabacloud

Extended from [ASTRA-SIM 1.0](https://github.com/astra-sim/astra-sim) with significant new capabilities:

- **Analytical Backend**: Estimates collective communication time using pre-configured bus bandwidth (busbw) values. Runs in seconds on any CPU server.
- **Simulation Backend (NS-3)**: Full packet-level simulation with network topology, congestion control, routing, and flow control.
- **Physical Backend (Beta)**: Generates NCCL-like traffic on real CPU RDMA clusters for NIC behavior study.

Key classes:
- `Sys`: Main system orchestrator, manages layer execution and dependencies
- `CollectivePhase`: Manages collective communication operation lifecycle
- `Algorithm`: Base class for all collective communication algorithms
- `AstraNetworkAPI`: Network interface abstraction layer
- `AstraMemoryAPI` / `AstraComputeAPI`: Resource management interfaces

### 3.4 ns-3-alibabacloud

- **Customized NS-3** network simulator optimized for datacenter AI training topologies
- Models: switches, links, congestion control (QCN, DCTCP, PFC), routing
- Supports **packet payload size**: 9,000 bytes (jumbo frames for RDMA)
- Enables precise evaluation of **network protocol and topology changes**

### 3.5 vidur-alibabacloud

- **Adapted from Microsoft Vidur** for multi-request LLM inference simulation
- Adds **Prefill/Decode (PD) disaggregation**: prefill and decode can run on separate nodes
- Supports flexible parallelism: TP, DP, PP, EP
- Multiple execution time prediction backends (AICB, SimAI-analytical, SimAI-simulation, native Vidur)
- Detailed per-request metrics: TTFT, TBT, E2E latency, scheduling delay

---

## 4. Simulation Modes and Scenarios

SimAI supports **seven distinct operational scenarios**:

| # | Scenario | Components Used | Use Case |
|---|----------|-----------------|----------|
| 1 | **AICB Test Suite** | AICB | Run communication patterns on real GPU clusters |
| 2 | **AICB/AIOB Workload** | AICB | Generate training/inference workload files |
| 3 | **Collective Comm Analysis** | SimCCL | Decompose collectives into P2P flows |
| 4 | **Collective Comm w/o GPU** | AICB + SimCCL + astra-sim(physical) | RDMA traffic on non-GPU clusters |
| 5 | **SimAI-Analytical** | AICB + astra-sim(analytical) | Rapid analysis on any server |
| 6 | **SimAI-Simulation** | AICB + SimCCL + astra-sim(sim) + NS-3 | Full packet-level simulation |
| 7 | **Multi-request Inference** | AICB + SimCCL + vidur + astra-sim | End-to-end inference simulation |

### 4.1 SimAI-Analytical

Fast simulation abstracting network details via bus bandwidth parameters.

**Typical use cases**:
- Compare training time across model architectures (e.g., effect of Expert count in MoE)
- Optimize TP/EP/PP/DP parallelism configurations
- Evaluate scale-out bandwidth cost-performance tradeoffs
- Rapid prototyping before committing to full simulation

```bash
# Basic analytical run
./bin/SimAI_analytical -w example/workload_analytical.txt \
  -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml

# With automatic busbw calculation
./bin/SimAI_analytical -w ./example/workload_analytical.txt \
  -g 9216 -nv 360 -nic 48.5 -n_p_s 8 -g_p_s 8 -r example-
```

**busbw.yaml structure**:
```yaml
test:
  TP:
    allreduce: 300    # GB/s AllReduce within TP group
    allgather: 280
    reducescatter: 280
    alltoall: 230
  DP:
    allgather: 380    # GB/s AllGather within DP group
    reducescatter: 380
  EP:
    allgather: 45     # GB/s for MoE Expert Parallelism
    reducescatter: 45
    alltoall: 80
  PP:
    busbw: 47.5       # GB/s for Pipeline Parallelism
```

### 4.2 SimAI-Simulation

Full-stack simulation with packet-level NS-3 network modeling.

**Typical use cases**:
- Design and evaluate novel collective communication algorithms
- Test network protocol optimizations (congestion control, routing)
- Evaluate new network topology designs (Spectrum-X, HPN, DCN+)
- High-fidelity reproduction of actual training cluster behavior

```bash
# Generate topology
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps

# Run simulation
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator \
  -t 16 -w ./example/microAllReduce.txt \
  -n ./Spectrum-X_128g_8gps_100Gbps_A100 \
  -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

**Key SimAI.conf parameters**:
```
ENABLE_QCN 1                   # QCN congestion control on/off
USE_DYNAMIC_PFC_THRESHOLD 1    # Dynamic PFC threshold tuning
PACKET_PAYLOAD_SIZE 9000       # Packet size (bytes)
CC_MODE 1                      # Congestion control mode
RATE_AI 50Mb/s                 # Additive increase rate
MIN_RATE 100Mb/s               # Minimum sending rate
BUFFER_SIZE 32                 # Switch buffer size
U_TARGET 0.95                  # Target link utilization
```

---

## 5. Supported Models and Configurations

### 5.1 LLM Models

**Fully supported** (with complete profiling):
- Meta-Llama-3-8B / Meta-Llama-3-70B
- Llama-2-7b-hf / Llama-2-70b-hf
- CodeLlama-34b-Instruct-hf
- Internlm-20b
- Qwen-72B

**Recently added** (inference with PD separation, adaptations in progress):
- **DeepSeek-V3-671B** — PP/EP communication and GPU memory allocation adaptations
- **Qwen3-MoE-235B** — MoE Expert Parallel support
- **Qwen3-Next-80B** — dense model with updated GPU memory allocation

### 5.2 GPU Types

| GPU | Architecture | Notes |
|-----|-------------|-------|
| A100 (80GB PCIe) | Ampere | Training baseline |
| H100 (80GB NVL) | Hopper | NVLink-optimized |
| Hopper (SM90) | Hopper | Inference: DeepGEMM, FlashMLA support |
| Blackwell (SM100) | Blackwell | Inference: DeepGEMM, FlashMLA support |

### 5.3 Parallelism Strategies

| Strategy | Abbreviation | Description |
|----------|-------------|-------------|
| Tensor Parallel | TP | Split model tensors across GPUs (1–8 GPUs typical) |
| Data Parallel | DP | Run multiple data replicas |
| Pipeline Parallel | PP | Split model layers into pipeline stages |
| Expert Parallel | EP | Distribute MoE experts across GPUs |
| Virtual Pipeline | VPP | Reduce PP bubble with virtual stages |
| Gradient Accumulation | GA | Accumulate gradients for effective larger batches |

Combinations supported: TP+DP, TP+PP, DP+PP, TP+DP+PP+EP

### 5.4 Network Topologies

| Topology | Type | GPU Scale |
|----------|------|-----------|
| AlibabaCloud HPN | Rail-optimized, dual/single plane | 512–9,216+ GPUs |
| NVIDIA Spectrum-X | Rail-optimized | 128–1,024 GPUs |
| DCN+ (Single ToR) | Fat-tree variant | 64–512 GPUs |
| DCN+ (Dual ToR) | Redundant fat-tree | 64–512 GPUs |
| Custom | Via `gen_Topo_Template.py` | Configurable |

---

## 6. Collective Communication Analysis

### 6.1 Algorithm Selection

SimCCL implements five collective algorithms, each with distinct performance profiles:

| Algorithm | Best For | Bottleneck |
|-----------|----------|------------|
| **Ring** | Large AllReduce (high bandwidth utilization) | Latency-sensitive for small messages |
| **DoubleBinaryTree** | Latency-critical AllReduce | Bandwidth efficiency lower than Ring |
| **HalvingDoubling** | AllGather with power-of-2 process counts | Requires power-of-2 GPU count |
| **AllToAll** | Expert parallelism MoE routing | Network congestion at scale |
| **NcclTreeFlowModel** | NCCL-like multi-channel trees | Complex scheduling |

### 6.2 Communication Types by Parallelism

| Parallel Dimension | Operations Used | Typical Bandwidth |
|-------------------|-----------------|-------------------|
| TP (intra-node) | AllReduce, AllGather, ReduceScatter | High (NVLink: 600–2,400 GB/s) |
| DP (inter-node) | AllGather, ReduceScatter | Medium (RDMA: 100–800 Gbps) |
| EP (inter-node) | AllToAll, AllGather, ReduceScatter | Medium–Low |
| PP (point-to-point) | Send/Recv | Low latency required |

### 6.3 Communication Efficiency Ratios

Pre-computed ratio CSV files in `astra-sim-alibabacloud/inputs/ratio/` capture:
- **busbw ratio**: Efficiency of bus bandwidth utilization (typically 0.4–0.95)
- **nic_ratio**: Fraction of NIC bandwidth utilized (10–95% depending on scale)
- **nvlink_ratio**: NVLink efficiency (0.45–0.90 for intra-node)

These ratios decrease with increasing GPU count due to:
- More hops in the collective tree
- Higher congestion probability
- Increased synchronization overhead

---

## 7. Inference Simulation (Vidur-AlibabaCloud)

### 7.1 Key Features

- **Prefill/Decode Disaggregation**: Prefill and decode phases run on separate node pools, enabling elastic resource allocation
- **Multi-request Scheduling**: Handles concurrent inference requests with configurable QPS
- **Fine-grained Metrics**: TTFT, TBT/TPOT, E2E latency, communication cost, scheduling delay

### 7.2 Scheduling Architecture

```
                    ┌─────────────────────────────────┐
Incoming Request ──►│      Global Scheduler           │
                    │  (Split-Wise / Round-Robin)      │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────┐              ┌──────────────────┐
   │  Prefill Replicas│              │  Decode Replicas │
   │  (TP, DP, PP, EP)│              │  (TP, DP, PP, EP)│
   └──────────────────┘              └──────────────────┘
```

### 7.3 Execution Time Prediction Backends

| Backend | Method | Scope | Accuracy |
|---------|--------|-------|---------|
| **AICB** | Real GPU profiling | TP, DP, PP, EP | High (hardware-dependent) |
| **SimAI-Analytical** | Bus bandwidth model | TP | Fast, approximate |
| **SimAI-Simulation** | NS-3 packet-level | TP | High fidelity |
| **Native Vidur** | sklearn RandomForest | TP, DP, PP | Approximate |

### 7.4 Example: DeepSeek-V3-671B Inference

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

Key parameters:
- `pd_node_ratio`: Fraction of nodes allocated to Prefill (0.0 = no disaggregation, 0.5 = equal split)
- `qps`: Requests per second (Poisson arrival process)
- `prefill_tokens` / `decode_tokens`: Sequence lengths per request

---

## 8. Key Technical Parameters

### 8.1 Analytical Simulation Parameters

| Parameter | Flag | Description |
|-----------|------|-------------|
| Workload file | `-w` | Path to AICB-generated `.txt` workload |
| Total GPUs | `-g` | Cluster size to simulate |
| GPUs per server | `-g_p_s` | Scale-up domain size (typically 8) |
| Result prefix | `-r` | Output file path prefix |
| Bus bandwidth | `-busbw` | Path to `busbw.yaml` |
| NVSwitches | `-nv` | Number of NVSwitches (for auto busbw) |
| NIC bandwidth | `-nic` | NIC link bandwidth (Gbps) |
| NICs per server | `-n_p_s` | Number of NICs per server |
| DP overlap | `-dp_o` | DP communication overlap ratio [0–1] |
| EP overlap | `-ep_o` | EP communication overlap ratio [0–1] |
| TP overlap | `-tp_o` | TP communication overlap ratio [0–1] |
| PP overlap | `-pp_o` | PP communication overlap ratio [0–1] |

### 8.2 Full Simulation Parameters (SimAI.conf)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_QCN` | 1 | Enable QCN congestion control |
| `USE_DYNAMIC_PFC_THRESHOLD` | 1 | Dynamic PFC threshold |
| `PACKET_PAYLOAD_SIZE` | 9000 | RDMA packet payload size (bytes) |
| `CC_MODE` | 1 | Congestion control mode |
| `ALPHA_RESUME_INTERVAL` | 1 | Rate adaptation interval |
| `RATE_AI` | 50Mb/s | Additive increase rate |
| `MIN_RATE` | 100Mb/s | Minimum sending rate |
| `BUFFER_SIZE` | 32 | Switch buffer size |
| `U_TARGET` | 0.95 | Target link utilization |

### 8.3 Topology Generation Parameters

```bash
python3 gen_Topo_Template.py \
  -topo Spectrum-X    # Topology: Spectrum-X, HPN, DCN+, ...
  -g 128              # Total GPUs
  -gt A100            # GPU type
  -bw 100Gbps         # Inter-node link bandwidth
  -nvbw 2400Gbps      # NVLink bandwidth
```

---

## 9. Comparison with Other Tools

### 9.1 SimAI vs. Similar Simulators

| Dimension | SimAI | ASTRA-SIM (original) | Vidur (Microsoft) | vLLM / TensorRT |
|-----------|-------|----------------------|-------------------|-----------------|
| Training simulation | ✅ Full | ✅ | ❌ | ❌ |
| Inference simulation | ✅ Full (v1.5+) | ❌ | ✅ | ❌ (real execution) |
| Packet-level network | ✅ NS-3 | ✅ NS-3 | ❌ | ❌ |
| Analytical fast mode | ✅ | ❌ | ❌ | ❌ |
| NCCL algorithm library | ✅ SimCCL | Partial | ❌ | N/A |
| PD disaggregation | ✅ | ❌ | ✅ (splitwise-sim) | Partial |
| MoE / EP support | ✅ | Partial | Partial | ✅ |
| GPU required to run | ❌ CPU-only | ❌ CPU-only | ❌ CPU-only | ✅ Requires GPU |
| Published at top venue | ✅ NSDI'25 | ✅ | ✅ | ✅ |

### 9.2 Problems Solved by SimAI

| Challenge | SimAI Solution |
|-----------|---------------|
| Expensive GPU cluster access for evaluation | Full CPU-based simulation, 100x+ cost reduction |
| Long iteration cycles for optimization | Fast analytical mode (minutes vs. hours) |
| Black-box performance understanding | Decomposed layer-by-layer analysis |
| Inference workload complexity | Prefill/Decode disaggregation with scheduling |
| Network algorithm evaluation | Packet-level NS-3 backend |
| Scale-out planning without hardware | Topology-independent simulation |

---

## 10. Key Findings and Conclusions

### 10.1 Research Contributions (NSDI'25)

1. **First unified training + inference simulation framework**: SimAI is the only tool that supports both full LLM training and multi-request inference simulation within a single coherent framework.

2. **High-fidelity collective communication modeling**: SimCCL faithfully reproduces NCCL Ring, Tree, and other algorithms, enabling evaluation of novel collective communication designs without real hardware.

3. **Scalability without GPU clusters**: SimAI simulates 9,000+ GPU clusters on a single CPU server, enabling architectural exploration at a fraction of the cost.

4. **Quantifiable network impact**: SimAI enables precise quantification of how network topology, congestion control, and routing choices affect LLM training throughput.

5. **Prefill/Decode disaggregation benefits**: Simulation results show that elastic PD separation significantly reduces TTFT for latency-sensitive workloads while maintaining throughput.

### 10.2 Performance Insights

- **Analytical mode speed**: 100–1,000x faster than full NS-3 simulation, suitable for hyperparameter sweeps
- **Simulation accuracy**: Within 5–15% of real hardware measurements when properly calibrated
- **NVLink efficiency**: 0.45–0.90 depending on collective type and scale-up domain size
- **NIC utilization**: 10–95% across different configurations; higher at larger scale due to AllReduce overhead
- **Communication vs. compute**: At scale (1,000+ GPUs), communication overhead can dominate total training step time

### 10.3 Optimal Configuration Recommendations

Based on SimAI analysis:

| Scenario | Recommendation |
|----------|---------------|
| Dense model, ≤8 GPUs | TP=8, DP=1, PP=1 (NVLink-only) |
| Dense model, 64–512 GPUs | TP=8, PP=4–8, DP=rest |
| MoE model, large EP | TP=4–8, EP=16–64, DP=rest |
| Inference, high QPS | PD disaggregation, pd_node_ratio=0.3–0.5 |
| Inference, low latency | Dedicated prefill nodes, higher TP |
| Network protocol | QCN + PFC + ECMP adaptive routing for best throughput |

### 10.4 Bottleneck Identification Framework

SimAI enables systematic identification of performance bottlenecks:

```
Compute-bound? → Reduce batch size, increase DP
Communication-bound? → Tune busbw, enable overlap, adjust TP/PP
Memory-bound? → Increase PP, enable gradient checkpointing
Network-congestion-bound? → Tune QCN/ECN parameters, enable adaptive routing
```

### 10.5 Collective Algorithm Selection Guide

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| Large AllReduce (>1MB), power-of-2 GPUs | Ring | Best bandwidth utilization |
| Small AllReduce (<256KB) | DoubleBinaryTree | Better latency |
| AllGather for inference KV cache | HalvingDoubling | Log-scale latency |
| MoE Expert routing | AllToAll | Pattern matches token dispatch |
| NVLink intra-node | NcclTreeFlowModel | Multi-channel NVLink optimization |

---

## 11. Limitations and Future Work

### 11.1 Current Limitations

1. **Physical Backend (Beta)**: The physical RDMA traffic generation mode is still in internal testing; not yet fully open-sourced.

2. **MoE PP/EP for inference**: SimAI PP/EP communication and GPU memory allocation modules for DeepSeek-V3, Qwen3-MoE inference are still in progress.

3. **Inference busbw automation**: Automatic busbw calculation for inference (like training's auto-busbw) is not yet available.

4. **Trace data dependency**: Vidur trace-based examples require external data files (e.g., `splitwise_conv.csv` from Microsoft Vidur) which must be downloaded separately.

5. **NS-3 compilation**: Simulation mode requires removing `ninja-build` from the environment, which may conflict with other build tools.

### 11.2 Active Development Areas

- End-to-end DeepSeek-V3-671B and Qwen3-MoE inference simulation
- Integration with M4 simulator (MIT CSAIL collaboration)
- Automatic busbw calculation open-source release
- SimCCL standalone repository release
- Physical backend stabilization and open-source release

### 11.3 Research Opportunities

SimAI opens the door for several research directions:

- **Novel network topologies**: Evaluate emerging datacenter architectures before hardware deployment
- **Collective algorithm innovation**: Design algorithms targeting specific hardware characteristics
- **LLM-aware network protocols**: Co-design network protocols with LLM training requirements
- **Heterogeneous cluster simulation**: Model mixed GPU generations within one cluster
- **Cost-performance optimization**: Automated Pareto-optimal configuration search

---

## References

- **Paper**: *SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale Large Language Model Training with Scalability and Precision*. NSDI'25 Spring. [[pdf](https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf)]
- **Repository**: https://github.com/aliyun/SimAI
- **AICB**: https://github.com/aliyun/aicb
- **SimCCL**: https://github.com/aliyun/SimCCL
- **ASTRA-SIM**: https://github.com/astra-sim/astra-sim
- **Vidur (Microsoft)**: https://github.com/microsoft/vidur
- **Tutorial**: [docs/Tutorial.md](./Tutorial.md)

---

*Document generated: March 2026 · SimAI v1.5*
