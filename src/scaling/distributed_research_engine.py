"""
Distributed Research Engine for Hyperscale Scientific Discovery
Advanced distributed computing system for massive-scale autonomous research operations

This module implements:
- Distributed research task orchestration
- Multi-node breakthrough discovery coordination  
- Scalable knowledge synthesis across clusters
- Load balancing and fault tolerance
- Real-time performance optimization
- Dynamic resource allocation and auto-scaling
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Process, cpu_count
import threading
from queue import Queue as ThreadQueue
import numpy as np
from collections import deque, defaultdict
import hashlib
import pickle
import redis
import aioredis
from pathlib import Path

from ..research.autonomous_breakthrough_engine import (
    AutonomousBreakthroughEngine, ResearchHypothesis, ResearchBreakthrough
)
from ..algorithms.breakthrough_discovery import BreakthroughDiscoveryEngine
from ..monitoring.autonomous_research_monitor import AutonomousResearchMonitor
from ..utils.logging_config import setup_logging
from ..utils.error_handling import robust_execution, safe_array_operation

logger = logging.getLogger(__name__)


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system"""
    node_id: str
    hostname: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    status: str  # 'active', 'busy', 'offline', 'maintenance'
    load_percentage: float = 0.0
    current_tasks: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """Represents a research task for distributed execution"""
    task_id: str
    task_type: str  # 'hypothesis_generation', 'breakthrough_discovery', 'validation', 'synthesis'
    priority: int  # 1-10, higher = more priority
    estimated_duration_minutes: float
    resource_requirements: Dict[str, Any]
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    assigned_node: Optional[str] = None
    status: str = 'pending'  # 'pending', 'assigned', 'running', 'completed', 'failed'
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions"""
    timestamp: float
    total_nodes: int
    active_nodes: int
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_queue_time: float
    average_execution_time: float
    system_throughput: float  # tasks per minute
    resource_utilization: Dict[str, float]
    bottlenecks: List[str]


class TaskScheduler:
    """Intelligent task scheduler for distributed research operations"""
    
    def __init__(self):
        self.task_queue = deque()
        self.priority_queues = {i: deque() for i in range(1, 11)}  # Priority 1-10
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Scheduling algorithms
        self.scheduling_algorithm = 'priority_based'  # 'round_robin', 'load_balanced', 'capability_based'
        
        logger.info("TaskScheduler initialized")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution"""
        
        # Add to appropriate priority queue
        self.priority_queues[task.priority].append(task)
        self.task_queue.append(task)
        
        logger.info(f"Task submitted: {task.task_id} (priority: {task.priority})")
        return task.task_id
    
    def get_next_task(self, node: ComputeNode) -> Optional[DistributedTask]:
        """Get next task for a specific node based on scheduling algorithm"""
        
        if self.scheduling_algorithm == 'priority_based':
            return self._get_priority_task(node)
        elif self.scheduling_algorithm == 'capability_based':
            return self._get_capability_task(node)
        elif self.scheduling_algorithm == 'load_balanced':
            return self._get_load_balanced_task(node)
        else:
            return self._get_round_robin_task(node)
    
    def _get_priority_task(self, node: ComputeNode) -> Optional[DistributedTask]:
        """Get highest priority task that node can handle"""
        
        # Check from highest to lowest priority
        for priority in range(10, 0, -1):
            queue = self.priority_queues[priority]
            
            for task in list(queue):
                if self._can_node_handle_task(node, task):
                    queue.remove(task)
                    self.task_queue.remove(task)
                    return task
        
        return None
    
    def _get_capability_task(self, node: ComputeNode) -> Optional[DistributedTask]:
        """Get task best matched to node capabilities"""
        
        best_task = None
        best_score = 0.0
        
        for task in list(self.task_queue):
            if task.status != 'pending':
                continue
            
            score = self._calculate_node_task_affinity(node, task)
            if score > best_score and self._can_node_handle_task(node, task):
                best_task = task
                best_score = score
        
        if best_task:
            self.priority_queues[best_task.priority].remove(best_task)
            self.task_queue.remove(best_task)
        
        return best_task
    
    def _get_load_balanced_task(self, node: ComputeNode) -> Optional[DistributedTask]:
        """Get task considering node load balancing"""
        
        # Prefer tasks for less loaded nodes
        if node.load_percentage > 80:
            return None  # Node too busy
        
        # Get task with load balancing consideration
        for task in list(self.task_queue):
            if (task.status == 'pending' and 
                self._can_node_handle_task(node, task) and
                node.load_percentage < 70):  # Only if node has capacity
                
                self.priority_queues[task.priority].remove(task)
                self.task_queue.remove(task)
                return task
        
        return None
    
    def _get_round_robin_task(self, node: ComputeNode) -> Optional[DistributedTask]:
        """Simple round-robin task assignment"""
        
        for task in list(self.task_queue):
            if task.status == 'pending' and self._can_node_handle_task(node, task):
                self.priority_queues[task.priority].remove(task)
                self.task_queue.remove(task)
                return task
        
        return None
    
    def _can_node_handle_task(self, node: ComputeNode, task: DistributedTask) -> bool:
        """Check if node can handle the task requirements"""
        
        if node.status not in ['active']:
            return False
        
        requirements = task.resource_requirements
        
        # Check CPU requirements
        required_cpu = requirements.get('cpu_cores', 1)
        if required_cpu > node.cpu_cores:
            return False
        
        # Check memory requirements
        required_memory = requirements.get('memory_gb', 1.0)
        if required_memory > node.memory_gb:
            return False
        
        # Check GPU requirements
        required_gpu = requirements.get('gpu_count', 0)
        if required_gpu > node.gpu_count:
            return False
        
        # Check capability requirements
        required_capabilities = requirements.get('capabilities', [])
        if not all(cap in node.capabilities for cap in required_capabilities):
            return False
        
        # Check current load
        if node.load_percentage > 90:
            return False
        
        return True
    
    def _calculate_node_task_affinity(self, node: ComputeNode, task: DistributedTask) -> float:
        """Calculate how well a node matches a task"""
        
        affinity_score = 0.0
        
        # Capability matching
        required_caps = task.resource_requirements.get('capabilities', [])
        matching_caps = len([cap for cap in required_caps if cap in node.capabilities])
        if required_caps:
            affinity_score += (matching_caps / len(required_caps)) * 0.4
        
        # Resource efficiency (prefer not over-provisioning)
        required_cpu = task.resource_requirements.get('cpu_cores', 1)
        cpu_efficiency = min(1.0, required_cpu / node.cpu_cores)
        affinity_score += cpu_efficiency * 0.3
        
        # Load balancing (prefer less loaded nodes)
        load_score = 1.0 - (node.load_percentage / 100.0)
        affinity_score += load_score * 0.3
        
        return affinity_score
    
    def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None,
                          error_message: Optional[str] = None):
        """Update task status"""
        
        # Find task in running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = status
            
            if status == 'completed':
                task.completed_at = time.time()
                task.result = result
                self.completed_tasks[task_id] = task
                del self.running_tasks[task_id]
                
            elif status == 'failed':
                task.completed_at = time.time()
                task.error_message = error_message
                self.failed_tasks[task_id] = task
                del self.running_tasks[task_id]
        
        logger.info(f"Task {task_id} status updated to: {status}")
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        
        return {
            'total_tasks': len(self.task_queue) + len(self.running_tasks) + 
                          len(self.completed_tasks) + len(self.failed_tasks),
            'pending_tasks': len([t for t in self.task_queue if t.status == 'pending']),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'queue_by_priority': {
                priority: len(queue) for priority, queue in self.priority_queues.items()
            },
            'scheduling_algorithm': self.scheduling_algorithm
        }


class NodeManager:
    """Manages compute nodes in the distributed system"""
    
    def __init__(self):
        self.nodes = {}
        self.node_heartbeats = {}
        self.heartbeat_timeout = 30  # seconds
        
        # Node monitoring
        self.monitoring_interval = 5  # seconds
        self.monitoring_active = False
        
        logger.info("NodeManager initialized")
    
    def register_node(self, node: ComputeNode) -> bool:
        """Register a new compute node"""
        
        self.nodes[node.node_id] = node
        self.node_heartbeats[node.node_id] = time.time()
        
        logger.info(f"Node registered: {node.node_id} ({node.hostname})")
        return True
    
    def update_node_heartbeat(self, node_id: str, metrics: Optional[Dict[str, Any]] = None):
        """Update node heartbeat and metrics"""
        
        if node_id in self.nodes:
            self.node_heartbeats[node_id] = time.time()
            self.nodes[node_id].last_heartbeat = time.time()
            
            if metrics:
                self.nodes[node_id].load_percentage = metrics.get('load_percentage', 0.0)
                self.nodes[node_id].performance_metrics.update(metrics)
        else:
            logger.warning(f"Heartbeat from unknown node: {node_id}")
    
    def get_available_nodes(self) -> List[ComputeNode]:
        """Get list of available compute nodes"""
        
        available_nodes = []
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            # Check if node is alive (recent heartbeat)
            if (current_time - self.node_heartbeats.get(node_id, 0)) <= self.heartbeat_timeout:
                if node.status in ['active']:
                    available_nodes.append(node)
            else:
                # Mark node as offline if heartbeat timeout
                if node.status != 'offline':
                    node.status = 'offline'
                    logger.warning(f"Node marked offline due to heartbeat timeout: {node_id}")
        
        return available_nodes
    
    def get_node_by_id(self, node_id: str) -> Optional[ComputeNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from cluster"""
        
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.node_heartbeats:
                del self.node_heartbeats[node_id]
            
            logger.info(f"Node removed: {node_id}")
            return True
        
        return False
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics"""
        
        available_nodes = self.get_available_nodes()
        
        total_cores = sum(node.cpu_cores for node in available_nodes)
        total_memory = sum(node.memory_gb for node in available_nodes)
        total_gpus = sum(node.gpu_count for node in available_nodes)
        
        avg_load = np.mean([node.load_percentage for node in available_nodes]) if available_nodes else 0.0
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(available_nodes),
            'offline_nodes': len(self.nodes) - len(available_nodes),
            'total_cpu_cores': total_cores,
            'total_memory_gb': total_memory,
            'total_gpus': total_gpus,
            'average_load_percentage': avg_load,
            'node_statuses': {
                status: len([n for n in self.nodes.values() if n.status == status])
                for status in ['active', 'busy', 'offline', 'maintenance']
            }
        }


class ResultAggregator:
    """Aggregates results from distributed research tasks"""
    
    def __init__(self):
        self.aggregation_rules = {}
        self.result_cache = {}
        
        # Initialize default aggregation rules
        self._setup_default_rules()
        
        logger.info("ResultAggregator initialized")
    
    def _setup_default_rules(self):
        """Setup default aggregation rules for different task types"""
        
        self.aggregation_rules = {
            'hypothesis_generation': self._aggregate_hypotheses,
            'breakthrough_discovery': self._aggregate_discoveries,
            'validation': self._aggregate_validations,
            'synthesis': self._aggregate_synthesis
        }
    
    def aggregate_results(self, task_type: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple tasks of the same type"""
        
        if not results:
            return {}
        
        if task_type in self.aggregation_rules:
            return self.aggregation_rules[task_type](results)
        else:
            return self._default_aggregation(results)
    
    def _aggregate_hypotheses(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate hypothesis generation results"""
        
        all_hypotheses = []
        total_generated = 0
        
        for result in results:
            hypotheses = result.get('hypotheses_generated', [])
            all_hypotheses.extend(hypotheses)
            total_generated += len(hypotheses)
        
        # Rank hypotheses by novelty and confidence
        ranked_hypotheses = sorted(
            all_hypotheses, 
            key=lambda h: (h.novelty_score * h.confidence_level), 
            reverse=True
        )
        
        return {
            'total_hypotheses_generated': total_generated,
            'unique_hypotheses': len(set(h.scientific_question for h in all_hypotheses)),
            'top_hypotheses': ranked_hypotheses[:10],  # Top 10
            'average_novelty': np.mean([h.novelty_score for h in all_hypotheses]) if all_hypotheses else 0.0,
            'average_confidence': np.mean([h.confidence_level for h in all_hypotheses]) if all_hypotheses else 0.0,
            'aggregation_timestamp': time.time()
        }
    
    def _aggregate_discoveries(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate breakthrough discovery results"""
        
        all_breakthroughs = []
        total_discovered = 0
        
        for result in results:
            breakthroughs = result.get('breakthroughs_discovered', [])
            all_breakthroughs.extend(breakthroughs)
            total_discovered += len(breakthroughs)
        
        # Rank by scientific impact
        ranked_breakthroughs = sorted(
            all_breakthroughs,
            key=lambda b: b.scientific_impact_score,
            reverse=True
        )
        
        # Calculate cross-domain impact
        all_domains = set()
        for breakthrough in all_breakthroughs:
            all_domains.update(breakthrough.cross_domain_implications)
        
        return {
            'total_breakthroughs_discovered': total_discovered,
            'unique_discovery_types': len(set(b.discovery.discovery_type for b in all_breakthroughs)),
            'top_breakthroughs': ranked_breakthroughs[:5],  # Top 5
            'cross_domain_impact': len(all_domains),
            'affected_domains': list(all_domains),
            'average_impact_score': np.mean([b.scientific_impact_score for b in all_breakthroughs]) if all_breakthroughs else 0.0,
            'publication_ready_count': sum(1 for b in all_breakthroughs if b.publication_readiness > 0.8),
            'aggregation_timestamp': time.time()
        }
    
    def _aggregate_validations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate validation results"""
        
        total_validations = len(results)
        successful_validations = sum(1 for r in results if r.get('validation_successful', False))
        
        all_metrics = []
        for result in results:
            if 'validation_metrics' in result:
                all_metrics.extend(result['validation_metrics'])
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': successful_validations / total_validations if total_validations > 0 else 0.0,
            'average_confidence': np.mean(all_metrics) if all_metrics else 0.0,
            'aggregation_timestamp': time.time()
        }
    
    def _aggregate_synthesis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate synthesis results"""
        
        synthesized_insights = []
        novel_connections = 0
        
        for result in results:
            insights = result.get('synthesized_insights', [])
            synthesized_insights.extend(insights)
            novel_connections += result.get('novel_connections', 0)
        
        return {
            'total_insights_synthesized': len(synthesized_insights),
            'novel_connections_discovered': novel_connections,
            'synthesis_quality_score': np.mean([result.get('quality_score', 0.0) for result in results]),
            'aggregation_timestamp': time.time()
        }
    
    def _default_aggregation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default aggregation for unknown task types"""
        
        return {
            'total_results': len(results),
            'combined_results': results,
            'aggregation_timestamp': time.time()
        }


class AutoScaler:
    """Automatic scaling system for dynamic resource management"""
    
    def __init__(self, node_manager: NodeManager, task_scheduler: TaskScheduler):
        self.node_manager = node_manager
        self.task_scheduler = task_scheduler
        
        # Scaling policies
        self.min_nodes = 1
        self.max_nodes = 100
        self.target_cpu_utilization = 70.0  # percent
        self.scale_up_threshold = 80.0
        self.scale_down_threshold = 30.0
        
        # Scaling cooldown periods (prevent oscillation)
        self.scale_up_cooldown = 300  # 5 minutes
        self.scale_down_cooldown = 600  # 10 minutes
        self.last_scale_action = 0
        
        # Metrics collection
        self.scaling_history = deque(maxlen=100)
        
        logger.info("AutoScaler initialized")
    
    async def evaluate_scaling_decision(self) -> Optional[Dict[str, Any]]:
        """Evaluate whether scaling is needed"""
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_action < self.scale_up_cooldown:
            return None
        
        # Gather metrics
        metrics = self._collect_scaling_metrics()
        
        # Make scaling decision
        scaling_decision = self._analyze_scaling_needs(metrics)
        
        if scaling_decision:
            self.scaling_history.append({
                'timestamp': current_time,
                'decision': scaling_decision,
                'metrics': metrics
            })
        
        return scaling_decision
    
    def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect metrics for scaling decisions"""
        
        cluster_stats = self.node_manager.get_cluster_stats()
        scheduler_stats = self.task_scheduler.get_scheduling_stats()
        
        # Calculate resource utilization
        available_nodes = self.node_manager.get_available_nodes()
        cpu_utilization = np.mean([node.load_percentage for node in available_nodes]) if available_nodes else 0.0
        
        # Calculate queue times and throughput
        completed_tasks = list(self.task_scheduler.completed_tasks.values())
        
        if completed_tasks:
            queue_times = [
                (task.started_at - task.created_at) 
                for task in completed_tasks 
                if task.started_at and task.created_at
            ]
            execution_times = [
                (task.completed_at - task.started_at) 
                for task in completed_tasks 
                if task.completed_at and task.started_at
            ]
            
            avg_queue_time = np.mean(queue_times) if queue_times else 0.0
            avg_execution_time = np.mean(execution_times) if execution_times else 0.0
            
            # Calculate throughput (tasks per minute)
            recent_completions = [
                task for task in completed_tasks 
                if task.completed_at and (time.time() - task.completed_at) < 3600  # Last hour
            ]
            throughput = len(recent_completions) / 60.0 if recent_completions else 0.0
        else:
            avg_queue_time = 0.0
            avg_execution_time = 0.0
            throughput = 0.0
        
        # Identify bottlenecks
        bottlenecks = []
        if scheduler_stats['pending_tasks'] > scheduler_stats['running_tasks'] * 2:
            bottlenecks.append('task_queue_backlog')
        if cpu_utilization > 85:
            bottlenecks.append('cpu_saturation')
        if avg_queue_time > 300:  # 5 minutes
            bottlenecks.append('long_queue_times')
        
        return ScalingMetrics(
            timestamp=time.time(),
            total_nodes=cluster_stats['total_nodes'],
            active_nodes=cluster_stats['active_nodes'],
            total_tasks=scheduler_stats['total_tasks'],
            pending_tasks=scheduler_stats['pending_tasks'],
            running_tasks=scheduler_stats['running_tasks'],
            completed_tasks=scheduler_stats['completed_tasks'],
            failed_tasks=scheduler_stats['failed_tasks'],
            average_queue_time=avg_queue_time,
            average_execution_time=avg_execution_time,
            system_throughput=throughput,
            resource_utilization={'cpu': cpu_utilization},
            bottlenecks=bottlenecks
        )
    
    def _analyze_scaling_needs(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """Analyze whether scaling is needed based on metrics"""
        
        scaling_decision = None
        
        # Scale up conditions
        should_scale_up = (
            metrics.resource_utilization['cpu'] > self.scale_up_threshold or
            metrics.pending_tasks > metrics.active_nodes * 3 or  # Queue backup
            metrics.average_queue_time > 180 or  # 3 minutes
            'task_queue_backlog' in metrics.bottlenecks
        )
        
        # Scale down conditions
        should_scale_down = (
            metrics.resource_utilization['cpu'] < self.scale_down_threshold and
            metrics.pending_tasks < metrics.active_nodes and
            metrics.average_queue_time < 30 and  # 30 seconds
            metrics.active_nodes > self.min_nodes
        )
        
        if should_scale_up and metrics.active_nodes < self.max_nodes:
            # Calculate how many nodes to add
            cpu_util = metrics.resource_utilization['cpu']
            queue_pressure = min(metrics.pending_tasks / max(1, metrics.active_nodes), 10)
            
            # Scale up by 1-5 nodes based on pressure
            nodes_to_add = max(1, min(5, int(queue_pressure / 2)))
            nodes_to_add = min(nodes_to_add, self.max_nodes - metrics.active_nodes)
            
            scaling_decision = {
                'action': 'scale_up',
                'nodes_to_add': nodes_to_add,
                'reason': f"CPU: {cpu_util:.1f}%, Queue: {metrics.pending_tasks}, Avg wait: {metrics.average_queue_time:.1f}s",
                'priority': 'high' if cpu_util > 90 or metrics.average_queue_time > 300 else 'medium'
            }
        
        elif should_scale_down:
            # Calculate how many nodes to remove
            excess_capacity = (self.target_cpu_utilization - metrics.resource_utilization['cpu']) / 100
            nodes_to_remove = max(1, min(3, int(metrics.active_nodes * excess_capacity / 2)))
            nodes_to_remove = min(nodes_to_remove, metrics.active_nodes - self.min_nodes)
            
            if nodes_to_remove > 0:
                scaling_decision = {
                    'action': 'scale_down',
                    'nodes_to_remove': nodes_to_remove,
                    'reason': f"Low utilization: {metrics.resource_utilization['cpu']:.1f}%",
                    'priority': 'low'
                }
        
        return scaling_decision
    
    async def execute_scaling_action(self, scaling_decision: Dict[str, Any]) -> bool:
        """Execute scaling action"""
        
        try:
            if scaling_decision['action'] == 'scale_up':
                success = await self._scale_up_nodes(scaling_decision['nodes_to_add'])
            elif scaling_decision['action'] == 'scale_down':
                success = await self._scale_down_nodes(scaling_decision['nodes_to_remove'])
            else:
                logger.warning(f"Unknown scaling action: {scaling_decision['action']}")
                return False
            
            if success:
                self.last_scale_action = time.time()
                logger.info(f"Scaling action executed: {scaling_decision}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
            return False
    
    async def _scale_up_nodes(self, nodes_to_add: int) -> bool:
        """Scale up by adding new nodes"""
        
        # In a real implementation, this would:
        # 1. Request new compute instances from cloud provider
        # 2. Wait for instances to be ready
        # 3. Install and configure research software
        # 4. Register nodes with the cluster
        
        # For simulation, we'll create virtual nodes
        for i in range(nodes_to_add):
            node_id = f"auto_node_{int(time.time())}_{i}"
            
            # Create simulated node with random specs
            node = ComputeNode(
                node_id=node_id,
                hostname=f"compute-{node_id}",
                cpu_cores=np.random.choice([4, 8, 16, 32]),
                memory_gb=np.random.choice([8, 16, 32, 64]),
                gpu_count=np.random.choice([0, 1, 2, 4]),
                status='active',
                capabilities=['research', 'discovery', 'analysis']
            )
            
            self.node_manager.register_node(node)
        
        logger.info(f"Scaled up by {nodes_to_add} nodes")
        return True
    
    async def _scale_down_nodes(self, nodes_to_remove: int) -> bool:
        """Scale down by removing nodes"""
        
        available_nodes = self.node_manager.get_available_nodes()
        
        # Sort by load (remove least loaded nodes first)
        nodes_by_load = sorted(available_nodes, key=lambda n: n.load_percentage)
        
        removed_count = 0
        for node in nodes_by_load:
            if removed_count >= nodes_to_remove:
                break
            
            # Only remove nodes with low load and no critical tasks
            if node.load_percentage < 20 and len(node.current_tasks) == 0:
                self.node_manager.remove_node(node.node_id)
                removed_count += 1
        
        logger.info(f"Scaled down by {removed_count} nodes")
        return removed_count > 0
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling action history"""
        
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            entry for entry in self.scaling_history
            if entry['timestamp'] > cutoff_time
        ]


class DistributedResearchEngine:
    """Main distributed research engine coordinating all components"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        """Initialize distributed research engine"""
        
        # Core components
        self.task_scheduler = TaskScheduler()
        self.node_manager = NodeManager()
        self.result_aggregator = ResultAggregator()
        self.auto_scaler = AutoScaler(self.node_manager, self.task_scheduler)
        
        # Research engines for each node type
        self.research_engines = {}
        
        # Distributed coordination
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        
        # System state
        self.is_running = False
        self.coordinator_task = None
        self.scaling_task = None
        
        # Performance metrics
        self.start_time = time.time()
        self.total_tasks_processed = 0
        self.total_breakthroughs_discovered = 0
        
        logger.info("DistributedResearchEngine initialized")
    
    async def initialize(self):
        """Initialize distributed system"""
        
        try:
            # Initialize Redis connection for coordination
            self.redis_client = await aioredis.from_url(f"redis://{self.redis_host}:{self.redis_port}")
            
            # Initialize local research engine
            local_engine = AutonomousBreakthroughEngine()
            self.research_engines['local'] = local_engine
            
            # Register local node
            local_node = ComputeNode(
                node_id='master_node',
                hostname='localhost',
                cpu_cores=cpu_count(),
                memory_gb=8.0,  # Simplified
                gpu_count=0,
                status='active',
                capabilities=['research', 'coordination', 'discovery', 'synthesis']
            )
            
            self.node_manager.register_node(local_node)
            
            logger.info("Distributed research engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed engine: {e}")
            raise
    
    async def start_distributed_research(self, duration_hours: float = 24.0):
        """Start distributed research operations"""
        
        if self.is_running:
            logger.warning("Distributed research already running")
            return
        
        await self.initialize()
        
        self.is_running = True
        
        # Start coordinator and auto-scaler
        self.coordinator_task = asyncio.create_task(self._coordination_loop())
        self.scaling_task = asyncio.create_task(self._auto_scaling_loop())
        
        # Submit initial research tasks
        await self._submit_initial_research_tasks()
        
        logger.info(f"🚀 Distributed research started for {duration_hours} hours")
        
        try:
            # Run for specified duration
            await asyncio.sleep(duration_hours * 3600)
        except asyncio.CancelledError:
            logger.info("Distributed research cancelled")
        finally:
            await self.stop_distributed_research()
    
    async def stop_distributed_research(self):
        """Stop distributed research operations"""
        
        self.is_running = False
        
        # Cancel coordinator tasks
        if self.coordinator_task:
            self.coordinator_task.cancel()
        if self.scaling_task:
            self.scaling_task.cancel()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Distributed research stopped")
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        
        try:
            while self.is_running:
                cycle_start = time.time()
                
                # Update node statuses
                await self._update_node_statuses()
                
                # Assign tasks to available nodes
                await self._assign_tasks_to_nodes()
                
                # Collect and aggregate results
                await self._collect_and_aggregate_results()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep until next cycle (5 second intervals)
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, 5.0 - cycle_time)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in coordination loop: {e}")
    
    async def _auto_scaling_loop(self):
        """Auto-scaling evaluation loop"""
        
        try:
            while self.is_running:
                # Evaluate scaling every 2 minutes
                await asyncio.sleep(120)
                
                # Get scaling decision
                scaling_decision = await self.auto_scaler.evaluate_scaling_decision()
                
                if scaling_decision:
                    logger.info(f"Auto-scaling decision: {scaling_decision}")
                    
                    # Execute scaling action
                    success = await self.auto_scaler.execute_scaling_action(scaling_decision)
                    if not success:
                        logger.error("Failed to execute scaling action")
                
        except Exception as e:
            logger.error(f"Error in auto-scaling loop: {e}")
    
    async def _submit_initial_research_tasks(self):
        """Submit initial research tasks to get the system started"""
        
        # Generate diverse research tasks
        task_types = [
            'hypothesis_generation',
            'breakthrough_discovery',
            'validation',
            'synthesis'
        ]
        
        domains = ['physics', 'biology', 'chemistry', 'mathematics', 'computer_science']
        
        for i in range(20):  # Submit 20 initial tasks
            task_type = np.random.choice(task_types)
            domain = np.random.choice(domains)
            
            task = DistributedTask(
                task_id=f"init_task_{i}_{task_type}",
                task_type=task_type,
                priority=np.random.randint(1, 11),
                estimated_duration_minutes=np.random.uniform(5, 60),
                resource_requirements={
                    'cpu_cores': np.random.choice([1, 2, 4]),
                    'memory_gb': np.random.choice([1, 2, 4, 8]),
                    'gpu_count': np.random.choice([0, 0, 0, 1]),  # Most tasks don't need GPU
                    'capabilities': ['research', 'discovery']
                },
                input_data={
                    'domain': domain,
                    'research_focus': f"{task_type} in {domain}",
                    'data_size': np.random.randint(100, 1000)
                }
            )
            
            self.task_scheduler.submit_task(task)
        
        logger.info("Submitted 20 initial research tasks")
    
    async def _update_node_statuses(self):
        """Update status of all nodes"""
        
        # In a real distributed system, this would:
        # 1. Send health check requests to all nodes
        # 2. Update node metrics and status
        # 3. Handle node failures and recoveries
        
        # For simulation, we'll just update heartbeats and add some randomness
        for node in self.node_manager.nodes.values():
            if node.status == 'active':
                # Simulate varying load
                node.load_percentage = max(0, min(100, 
                    node.load_percentage + np.random.normal(0, 5)
                ))
                
                # Simulate occasional node issues
                if np.random.random() < 0.01:  # 1% chance
                    node.status = 'maintenance'
                    logger.warning(f"Node {node.node_id} entered maintenance mode")
                
                # Update heartbeat
                self.node_manager.update_node_heartbeat(
                    node.node_id, 
                    {'load_percentage': node.load_percentage}
                )
    
    async def _assign_tasks_to_nodes(self):
        """Assign pending tasks to available nodes"""
        
        available_nodes = self.node_manager.get_available_nodes()
        
        for node in available_nodes:
            # Skip busy nodes
            if node.load_percentage > 80:
                continue
            
            # Get next task for this node
            task = self.task_scheduler.get_next_task(node)
            
            if task:
                # Assign task to node
                task.assigned_node = node.node_id
                task.status = 'assigned'
                task.started_at = time.time()
                
                # Add to node's current tasks
                node.current_tasks.append(task.task_id)
                
                # Add to running tasks
                self.task_scheduler.running_tasks[task.task_id] = task
                
                # Start task execution (simulated)
                asyncio.create_task(self._execute_task(task, node))
                
                logger.info(f"Assigned task {task.task_id} to node {node.node_id}")
    
    async def _execute_task(self, task: DistributedTask, node: ComputeNode):
        """Execute a research task on a specific node"""
        
        try:
            task.status = 'running'
            
            # Simulate task execution time
            execution_time = task.estimated_duration_minutes * 60  # Convert to seconds
            # Add some randomness
            actual_time = max(1, execution_time + np.random.normal(0, execution_time * 0.2))
            
            # Simulate node load during task
            node.load_percentage = min(100, node.load_percentage + 20)
            
            await asyncio.sleep(min(actual_time, 300))  # Max 5 minutes for demo
            
            # Generate simulated results based on task type
            result = await self._generate_task_result(task)
            
            # Update task status
            task.status = 'completed'
            task.completed_at = time.time()
            task.result = result
            
            # Update scheduler
            self.task_scheduler.update_task_status(task.task_id, 'completed', result)
            
            # Remove from node's current tasks
            if task.task_id in node.current_tasks:
                node.current_tasks.remove(task.task_id)
            
            # Reduce node load
            node.load_percentage = max(0, node.load_percentage - 20)
            
            # Update metrics
            self.total_tasks_processed += 1
            
            if task.task_type == 'breakthrough_discovery':
                breakthroughs = result.get('breakthroughs_discovered', [])
                self.total_breakthroughs_discovered += len(breakthroughs)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = time.time()
            
            self.task_scheduler.update_task_status(task.task_id, 'failed', error_message=str(e))
            
            if task.task_id in node.current_tasks:
                node.current_tasks.remove(task.task_id)
            
            node.load_percentage = max(0, node.load_percentage - 20)
            
            logger.error(f"Task {task.task_id} failed: {e}")
    
    async def _generate_task_result(self, task: DistributedTask) -> Dict[str, Any]:
        """Generate simulated task result"""
        
        if task.task_type == 'hypothesis_generation':
            # Simulate hypothesis generation
            num_hypotheses = np.random.randint(1, 6)
            hypotheses = []
            
            for i in range(num_hypotheses):
                hypothesis = ResearchHypothesis(
                    hypothesis_id=f"hyp_{task.task_id}_{i}",
                    scientific_question=f"Research question {i} in {task.input_data.get('domain', 'general')}",
                    theoretical_foundation="Simulated theoretical foundation",
                    testable_predictions=["Prediction 1", "Prediction 2"],
                    experimental_design={},
                    expected_outcomes=["Outcome 1"],
                    confidence_level=np.random.uniform(0.6, 0.95),
                    novelty_score=np.random.uniform(0.5, 0.9),
                    impact_potential=np.random.uniform(0.4, 0.8),
                    interdisciplinary_connections=[]
                )
                hypotheses.append(hypothesis)
            
            return {
                'task_id': task.task_id,
                'hypotheses_generated': hypotheses,
                'execution_time_seconds': np.random.uniform(60, 300)
            }
        
        elif task.task_type == 'breakthrough_discovery':
            # Simulate breakthrough discovery
            num_breakthroughs = np.random.randint(0, 4)
            breakthroughs = []
            
            for i in range(num_breakthroughs):
                breakthrough = {
                    'discovery_id': f"disc_{task.task_id}_{i}",
                    'discovery_type': np.random.choice(['quantum_correlation', 'phase_transition', 'collective_behavior']),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'significance': np.random.uniform(0.8, 0.98),
                    'scientific_impact_score': np.random.uniform(0.6, 0.9),
                    'cross_domain_implications': np.random.choice([['physics'], ['biology'], ['physics', 'chemistry']])
                }
                breakthroughs.append(breakthrough)
            
            return {
                'task_id': task.task_id,
                'breakthroughs_discovered': breakthroughs,
                'execution_time_seconds': np.random.uniform(120, 600)
            }
        
        elif task.task_type == 'validation':
            return {
                'task_id': task.task_id,
                'validation_successful': np.random.random() > 0.2,  # 80% success rate
                'validation_metrics': [np.random.uniform(0.7, 0.95) for _ in range(3)],
                'execution_time_seconds': np.random.uniform(30, 180)
            }
        
        elif task.task_type == 'synthesis':
            return {
                'task_id': task.task_id,
                'synthesized_insights': [f"Insight {i}" for i in range(np.random.randint(1, 4))],
                'novel_connections': np.random.randint(0, 3),
                'quality_score': np.random.uniform(0.6, 0.9),
                'execution_time_seconds': np.random.uniform(90, 400)
            }
        
        else:
            return {
                'task_id': task.task_id,
                'generic_result': 'Task completed successfully',
                'execution_time_seconds': np.random.uniform(60, 300)
            }
    
    async def _collect_and_aggregate_results(self):
        """Collect and aggregate results from completed tasks"""
        
        # Group completed tasks by type
        completed_by_type = defaultdict(list)
        
        for task in self.task_scheduler.completed_tasks.values():
            if task.result:
                completed_by_type[task.task_type].append(task.result)
        
        # Aggregate results for each task type
        for task_type, results in completed_by_type.items():
            if len(results) >= 5:  # Aggregate when we have enough results
                aggregated = self.result_aggregator.aggregate_results(task_type, results)
                
                # Store aggregated results (in Redis in real implementation)
                logger.info(f"Aggregated {len(results)} {task_type} results")
    
    async def _update_performance_metrics(self):
        """Update system performance metrics"""
        
        # This would update comprehensive performance metrics
        # For now, we'll just log basic stats periodically
        
        current_time = time.time()
        uptime_hours = (current_time - self.start_time) / 3600
        
        if int(uptime_hours * 12) % 5 == 0:  # Every 5 minutes
            cluster_stats = self.node_manager.get_cluster_stats()
            scheduler_stats = self.task_scheduler.get_scheduling_stats()
            
            logger.info(f"System Performance - Uptime: {uptime_hours:.2f}h, "
                       f"Nodes: {cluster_stats['active_nodes']}, "
                       f"Tasks: {scheduler_stats['running_tasks']} running, "
                       f"{scheduler_stats['pending_tasks']} pending, "
                       f"Total processed: {self.total_tasks_processed}")
    
    def get_distributed_system_status(self) -> Dict[str, Any]:
        """Get comprehensive distributed system status"""
        
        current_time = time.time()
        uptime_hours = (current_time - self.start_time) / 3600
        
        return {
            'system_status': {
                'is_running': self.is_running,
                'uptime_hours': uptime_hours,
                'start_time': self.start_time,
                'total_tasks_processed': self.total_tasks_processed,
                'total_breakthroughs_discovered': self.total_breakthroughs_discovered
            },
            'cluster_stats': self.node_manager.get_cluster_stats(),
            'scheduling_stats': self.task_scheduler.get_scheduling_stats(),
            'scaling_history': self.auto_scaler.get_scaling_history(hours=6),
            'performance_metrics': {
                'tasks_per_hour': self.total_tasks_processed / max(0.1, uptime_hours),
                'breakthroughs_per_hour': self.total_breakthroughs_discovered / max(0.1, uptime_hours)
            }
        }


# Demonstration and testing functions
async def demo_distributed_research():
    """Demonstrate distributed research capabilities"""
    
    logger.info("🌐 Starting Distributed Research Engine Demo")
    
    # Initialize distributed engine
    engine = DistributedResearchEngine()
    
    try:
        # Start distributed research for 5 minutes (demo)
        await engine.start_distributed_research(duration_hours=0.083)  # 5 minutes
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Get final status
        status = engine.get_distributed_system_status()
        
        print("\n🌐 Distributed Research Demo Results:")
        print("=" * 50)
        print(f"Uptime: {status['system_status']['uptime_hours']:.2f} hours")
        print(f"Active Nodes: {status['cluster_stats']['active_nodes']}")
        print(f"Total Tasks Processed: {status['system_status']['total_tasks_processed']}")
        print(f"Breakthroughs Discovered: {status['system_status']['total_breakthroughs_discovered']}")
        print(f"Tasks per Hour: {status['performance_metrics']['tasks_per_hour']:.1f}")
        print(f"Scaling Actions: {len(status['scaling_history'])}")
        
        await engine.stop_distributed_research()
        
        print("\n✅ Distributed Research Demo Complete!")


if __name__ == "__main__":
    # Run distributed research demo
    asyncio.run(demo_distributed_research())