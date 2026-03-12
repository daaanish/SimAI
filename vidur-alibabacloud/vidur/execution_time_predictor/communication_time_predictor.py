import time
from vidur.entities import Batch
from enum import Enum
from vidur.config import ReplicaConfig, BaseExecutionTimePredictorConfig, BaseModelConfig
from vidur.execution_time_predictor.SimAIWorkload import SimAIWorkload, WorkItem
from typing import Dict
import subprocess
import csv
import os
import hashlib

# TPTimePredictor是一个单例
# 只需要考虑单次allreduce的时间
# TPTimePredictor is a singleton
# Only consider single allreduce time
class TPTimePredictor:
    def __init__(self,
        model_config: BaseModelConfig,
        replica_config: ReplicaConfig,
        predictor_config: BaseExecutionTimePredictorConfig
    ):
        assert model_config.num_layers % replica_config.num_pipeline_stages == 0
        self.num_layers_per_pp_stage = model_config.num_layers // replica_config.num_pipeline_stages
        self.hidden_size = model_config.embedding_dim
        self.predictor_config = predictor_config
        self.replica_config = replica_config
        # TODO ct: change to sizeof(tensor.dtype)
        # fy：得做； 动态调整dtype； 从config里面获取
        # fy: need to do; dynamically adjust dtype; get from config
        self.tensor_size = 2
        self.workload: SimAIWorkload = SimAIWorkload(
            tp_size=replica_config.tensor_parallel_size,
            ep_size=1,
            pp_size=replica_config.num_pipeline_stages,
            vpp_size=self.num_layers_per_pp_stage,
            ga_num=1,
            world_size=replica_config.world_size,
            pp_comm=0
        )
        self.simai_dir = os.path.abspath(predictor_config.simai_dir)
        self.simai_ns3_binary = f'{self.simai_dir}/bin/SimAI_simulator'
        self.simai_analytical_binary = f'{self.simai_dir}/bin/SimAI_analytical'
        # self.workload_path = '/tmp'
        self.workload_path = f'{self.simai_dir}/tmp_simai_inference_workload'
        # 确保workload_path存在，如果不存在则创建
        # Ensure workload_path exists, create if not exists
        if not os.path.exists(self.workload_path):
            os.makedirs(self.workload_path)
        else:
            assert os.path.isdir(self.workload_path), f"Workload path exists but is not a directory: {self.workload_path}"
        
        # self.workload_path = '/disk2/futianhao/software3/sim-ai-inference-n/simulator_output/tmp_simai_workload'
        self.cache: Dict[int, float] = {}
        # 记录 SimAI 无法处理的通信规模，避免重复尝试和重复打印日志
        # Track communication sizes that SimAI cannot handle, to avoid repeated attempts and log noise
        self.failed_cache_keys: set = set()

    def _interpolate_from_cache(self, all_reduce_bytes: int):
        """
        当 SimAI 不支持某一通信规模时，从缓存中插值估算延迟。
        策略：找到缓存中最近的通信规模，按比例线性缩放。
        
        When SimAI does not support a communication size, estimate latency by
        interpolating from cached results. Strategy: find the nearest cached size
        and scale the latency proportionally (linear scaling).
        
        Returns the estimated latency in ms, or None if cache is empty.
        """
        if not self.cache:
            return None
        
        # 缓存key为(hidden_size, num_tokens, tensor_size)，对应all_reduce_bytes = hidden_size * num_tokens * tensor_size
        # Cache key is (hidden_size, num_tokens, tensor_size), corresponding to
        # all_reduce_bytes = hidden_size * num_tokens * tensor_size
        cached_bytes = {
            k[0] * k[1] * k[2]: v
            for k, v in self.cache.items()
            if k[0] * k[1] * k[2] > 0
        }
        
        if not cached_bytes:
            return None
        
        # 找到最近的已知通信规模
        # Find the nearest known communication size
        nearest_bytes = min(cached_bytes.keys(), key=lambda x: abs(x - all_reduce_bytes))
        nearest_latency = cached_bytes[nearest_bytes]
        
        if nearest_latency <= 0:
            return None
        
        # 按比例线性缩放
        # Scale proportionally (linear approximation)
        estimated_latency = nearest_latency * (all_reduce_bytes / nearest_bytes)
        return estimated_latency

    # > 重写 增加两个功能 复用相同的workload 和 相同command的结果
    # > rewrite: add two features to reuse same workloads and results of same commands
    def get_execution_time(self, batch: Batch):
        self.workload.flush()
        num_tokens_in_batch = batch._total_num_tokens_rounded
        all_reduce_bytes = self.hidden_size * num_tokens_in_batch * self.tensor_size
        
        # 使用包含所有相关参数的元组作为缓存键，而不是仅仅使用all_reduce_bytes
        # Use a tuple containing all relevant parameters as cache key instead of just all_reduce_bytes
        cache_key = (self.hidden_size, num_tokens_in_batch, self.tensor_size)
        
        # 如果结果已经在缓存中，直接返回
        # If result is already in cache, return directly
        if cache_key in self.cache:
            return (self.cache[cache_key]
                + self.predictor_config.nccl_cpu_launch_overhead_ms
                + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                * self.replica_config.tensor_parallel_size**1.25)
        
        # 如果该规模已知 SimAI 无法处理，直接 fallback
        # If this size is already known to fail in SimAI, fall back immediately
        if cache_key in self.failed_cache_keys:
            return -1
        
        # 为workload和命令生成唯一标识符
        # 基于WorkItem的所有相关参数生成哈希值
        # Generate unique identifier for workload and command
        # Generate hash based on all relevant parameters of WorkItem
            
        
        # TODO: > 增加layer0 str（1） "ALLREDUCE"等其他变量对于hash的影响，目前是写成固定值的
        # TODO: > Add impact of other variables like layer0 str(1) "ALLREDUCE" to hash, currently hardcoded
            
        work_item_data = (
            "layer0" + 
            str(1) + 
            "ALLREDUCE" + 
            str(self.hidden_size) +
            str(num_tokens_in_batch) +
            str(self.tensor_size)
        )
        
        workload_identifier = hashlib.md5(work_item_data.encode()).hexdigest()
        
        
        # 检查是否已经有对应的workload文件
        # Check if corresponding workload file already exists
        workload_file = f'{self.workload_path}/allreduce_{workload_identifier}.txt'
        
        # 如果workload文件不存在，则生成
        # Generate if workload file does not exist
        if not os.path.exists(workload_file):
            # 假设每个layer同构
            # Assume each layer is homogeneous
            self.workload.append_work_item(
                WorkItem(
                    name="layer0",
                    forward_compute_time=1,
                    forward_comm="ALLREDUCE",
                    forward_comm_size=all_reduce_bytes
                )
            )
            self.workload.dump_file(workload_file)
        
        topo = os.path.abspath(self.predictor_config.simai_simulation_topo)
        conf = os.path.abspath(self.predictor_config.simai_simulation_config)
        
        # 检查是否已经有对应命令的结果缓存
        # Check if result cache for corresponding command already exists
        command_identifier = hashlib.md5(
            f'{workload_identifier}_{topo}_{conf}'.encode()
        ).hexdigest()
        result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd_{command_identifier}.csv'
        
        # 如果结果文件不存在，则运行命令
        # Run command if result file does not exist
        if not os.path.exists(result_file):
            # 调用simai
            # Call simai
            command = f'AS_SEND_LAT=6 AS_NVLS_ENABLE=1 {self.simai_ns3_binary} -t 16 -w {workload_file} -n {topo} -c {conf}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.simai_dir)
            if result.returncode != 0:
                print(f'{command} failed. ret: {result.returncode}')
            
            # 移动结果文件到带标识符的文件名
            # Move result file to identified filename
            original_result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd.csv'
            if os.path.exists(original_result_file):
                os.rename(original_result_file, result_file)
            else:
                # 尝试从缓存中插值：找到最近的已知通信规模并按比例缩放
                # Try to interpolate from cache: find the nearest known size and scale proportionally
                latency = self._interpolate_from_cache(all_reduce_bytes)
                if latency is not None:
                    self.cache[cache_key] = latency
                else:
                    # 记录该规模为已知失败，后续请求直接 fallback 到 sklearn，不再尝试 SimAI
                    # Mark this size as known-failing; future requests fall back to sklearn silently
                    self.failed_cache_keys.add(cache_key)
                    return -1
                return (self.cache[cache_key]
                    + self.predictor_config.nccl_cpu_launch_overhead_ms
                    + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                    * self.replica_config.tensor_parallel_size**1.25)
        
        # 从结果文件获取allreduce latency
        # Get allreduce latency from result file
        with open(result_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) == 0:
                # 结果文件为空，尝试从缓存插值
                # Result file is empty, try to interpolate from cache
                latency = self._interpolate_from_cache(all_reduce_bytes)
                if latency is not None:
                    self.cache[cache_key] = latency
                else:
                    # 记录该规模为已知失败，后续请求直接 fallback 到 sklearn，不再尝试 SimAI
                    # Mark this size as known-failing; future requests fall back to sklearn silently
                    self.failed_cache_keys.add(cache_key)
                    return -1
                return (self.cache[cache_key]
                    + self.predictor_config.nccl_cpu_launch_overhead_ms
                    + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                    * self.replica_config.tensor_parallel_size**1.25)
            
            # 最后一行第二列为total comm
            # simai返回us，vidur要求ms
            # Second column of last row is total comm
            # simai returns us, vidur requires ms
            latency = float(rows[-1][1]) * 1e-3
            
            self.cache[cache_key] = latency
        
        return (self.cache[cache_key]
            # TODO: chentong whether we need these?
            # can these parameters be integreted into simai?
            + self.predictor_config.nccl_cpu_launch_overhead_ms
            + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
            * self.replica_config.tensor_parallel_size**1.25)
        
    # > 重写 增加两个功能 复用相同的workload 和 相同command的结果
    # > rewrite: add two features to reuse same workloads and results of same commands
    def get_execution_time_by_simai_analytical(self, batch: Batch):
        """
        使用SimAI分析工具预测通信时间的方法
        Args: batch: Batch对象，包含需要处理的批次信息
        Returns: float: 预测的执行时间（毫秒），如果出错则返回-1
            
        The method of predicting communication time using the SimAI analysis tool
        Args: batch: Batch object, containing the batch information to be processed. 
        Returns: float: Predicted execution time (milliseconds), returns -1 if an error occurs.
        """
        self.workload.flush()
        num_tokens_in_batch = batch._total_num_tokens_rounded
        all_reduce_bytes = self.hidden_size * num_tokens_in_batch * self.tensor_size
        
        # 使用包含所有相关参数的元组作为缓存键，而不是仅仅使用all_reduce_bytes
        # Use a tuple containing all relevant parameters as cache key instead of just all_reduce_bytes
        cache_key = (self.hidden_size, num_tokens_in_batch, self.tensor_size)
        
        # 如果结果已经在缓存中，直接返回
        # If result is already in cache, return directly
        if cache_key in self.cache:
            return (self.cache[cache_key]
                + self.predictor_config.nccl_cpu_launch_overhead_ms
                + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                * self.replica_config.tensor_parallel_size**1.25)
        
        # 如果该规模已知 SimAI 无法处理，直接 fallback
        # If this size is already known to fail in SimAI, fall back immediately
        if cache_key in self.failed_cache_keys:
            return -1
        
        
        # 为workload和命令生成唯一标识符
        # 基于WorkItem的所有相关参数生成哈希值
        # TODO: > 增加layer0 str（1） "ALLREDUCE"等其他变量对于hash的影响，目前是写成固定值的
        # Generate unique identifier for workload and command
        # Generate hash based on all relevant parameters of WorkItem
        # TODO: > Add impact of other variables like layer0 str(1) "ALLREDUCE" to hash, currently hardcoded
            
        work_item_data = (
            "layer0" + 
            str(1) + 
            "ALLREDUCE" + 
            str(self.hidden_size) +
            str(num_tokens_in_batch) +
            str(self.tensor_size)
        )
        
        # 使用MD5哈希算法生成工作负载标识符
        workload_identifier = hashlib.md5(work_item_data.encode()).hexdigest()
        
        
        # 检查是否已经有对应的workload文件
        # Check if there is already a corresponding workload file
        workload_file = f'{self.workload_path}/allreduce_{workload_identifier}.txt'
        
        # 如果workload文件不存在，则生成
        # Generate if workload file does not exist
        if not os.path.exists(workload_file):
            # 假设每个layer同构
            # Assume each layer is homogeneous
            self.workload.append_work_item(
                WorkItem(
                    name="layer0",            # Layer name
                    forward_compute_time=1,    # Forward compute time
                    forward_comm="ALLREDUCE",  # Forward communication type
                    forward_comm_size=all_reduce_bytes  # Forward communication size (bytes)
                )
            )
            self.workload.dump_file(workload_file)
        

        # 获取拓扑文件和配置文件的绝对路径
        # Get absolute paths for topology file and configuration file
        topo = os.path.abspath(self.predictor_config.simai_simulation_topo)
        conf = os.path.abspath(self.predictor_config.simai_simulation_config)
        
        # 检查是否已经有对应命令的结果缓存
        # Check if result cache for corresponding command already exists
        command_identifier = hashlib.md5(
            f'{workload_identifier}_{topo}_{conf}'.encode()
        ).hexdigest()
        # result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd_{command_identifier}.csv'
        result_file = f'{self.simai_dir}/analytical_EndToEnd_{command_identifier}.csv'
        
        # 如果结果文件不存在，则运行命令
        # Run command if result file does not exist
        if not os.path.exists(result_file):
            # 调用simai
            # Call simai
            # command = f'AS_SEND_LAT=6 AS_NVLS_ENABLE=1 {self.simai_ns3_binary} -t 16 -w {workload_file} -n {topo} -c {conf}'
            # ./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
            
            cmd_g_p_s = 8          
            cmd_r= 'analytical_'    # Result file prefix
            cmd_busbw = f'{self.simai_dir}/example/busbw.yaml'
            # command = f'{self.simai_analytical_binary} -w {workload_file} -g {self.replica_config.world_size} -g_p_s {cmd_g_p_s} -r {cmd_r} -busbw {cmd_busbw}'
            command = f'{self.simai_analytical_binary} -w {workload_file} -g {self.replica_config.world_size} -g_p_s {cmd_g_p_s} -r {cmd_r}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.simai_dir)
            if result.returncode != 0:
                print(f'{command} failed. ret: {result.returncode}')
            
            # 移动结果文件到带标识符的文件名
            # Move result file to identified filename
            # original_result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd.csv'
            # original_result_file = f'{self.simai_dir}/analytical_EndToEnd.csv'
            original_result_file = f'{self.simai_dir}/results/analytical_EndToEnd.csv'
            if os.path.exists(original_result_file):
                os.rename(original_result_file, result_file)
            else:
                # 尝试从缓存中插值：找到最近的已知通信规模并按比例缩放
                # Try to interpolate from cache: find the nearest known size and scale proportionally
                latency = self._interpolate_from_cache(all_reduce_bytes)
                if latency is not None:
                    self.cache[cache_key] = latency
                else:
                    # 记录该规模为已知失败，后续请求直接 fallback 到 sklearn，不再尝试 SimAI
                    # Mark this size as known-failing; future requests fall back to sklearn silently
                    self.failed_cache_keys.add(cache_key)
                    return -1
                return (self.cache[cache_key]
                    + self.predictor_config.nccl_cpu_launch_overhead_ms
                    + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                    * self.replica_config.tensor_parallel_size**1.25)
        
        # 从结果文件获取allreduce latency
        # Get allreduce latency from result file
        with open(result_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) == 0:
                # 结果文件为空，尝试从缓存插值
                # Result file is empty, try to interpolate from cache
                latency = self._interpolate_from_cache(all_reduce_bytes)
                if latency is not None:
                    self.cache[cache_key] = latency
                else:
                    # 记录该规模为已知失败，后续请求直接 fallback 到 sklearn，不再尝试 SimAI
                    # Mark this size as known-failing; future requests fall back to sklearn silently
                    self.failed_cache_keys.add(cache_key)
                    return -1
                return (self.cache[cache_key]
                    + self.predictor_config.nccl_cpu_launch_overhead_ms
                    + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                    * self.replica_config.tensor_parallel_size**1.25)
            
            # 从索引5的位置获取延迟数据（微秒），转换为毫秒
            # Get latency data from index 5 position (microseconds), convert to milliseconds
            latency = float(rows[-1][5]) * 1e-3
            
            # 将结果存入缓存，以便后续相同参数的请求直接使用
            # Store result in cache for future requests with same parameters
            self.cache[cache_key] = latency

        
        # 返回最终预测的执行时间，包括缓存的延迟和额外的开销
        # Return final predicted execution time, including cached latency and additional overhead
        return (self.cache[cache_key]
            # TODO: chentong whether we need these?
            # can these parameters be integreted into simai?
            + self.predictor_config.nccl_cpu_launch_overhead_ms
            + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
            * self.replica_config.tensor_parallel_size**1.25)
        