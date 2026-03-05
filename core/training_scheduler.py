import zlib
"""
Training Scheduler: Responsible for scheduling and managing training jobs

Provides intelligent scheduling, priority management, concurrency control, and job
lifecycle management for model training processes.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import time
import threading
import queue
import logging
import heapq
import random
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from core.error_handling import error_handler

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """Represents a training job with all necessary metadata"""
    
    model_id: str
    data_config: Dict[str, Any]
    training_params: Dict[str, Any]
    priority: str = "normal"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class JobStatus(Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPING = "stopping"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TrainingScheduler:
    """Intelligent training job scheduler with priority management and concurrency control"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrainingScheduler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Job storage
        self.jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> job_info
        self.job_queue = []  # Priority queue using heapq
        self.job_lock = threading.Lock()
        
        # Scheduling configuration
        self.config = {
            'max_concurrent_jobs': 3,
            'default_priority': JobPriority.NORMAL,
            'enable_auto_scheduling': True,
            'schedule_check_interval': 5,  # seconds
            'job_timeout_hours': 24,
            'enable_priority_boost': True,
            'concurrent_model_limit': {
                'language': 2,
                'vision_image': 2,
                'vision_video': 1,
                'audio': 2,
                'knowledge': 1,
                'manager': 1
            }
        }
        
        # Statistics
        self.stats = {
            'total_jobs_scheduled': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'jobs_cancelled': 0,
            'average_job_duration': 0,
            'total_training_time': 0,
            'queue_wait_times': [],
            'last_schedule_optimization': time.time()
        }
        
        # Active jobs tracking
        self.active_jobs = set()
        self.model_type_usage = {}  # Track concurrent usage per model type
        
        # Scheduling thread
        self._scheduling_active = True
        self.scheduling_thread = threading.Thread(target=self._scheduling_loop)
        self.scheduling_thread.daemon = True
        self.scheduling_thread.start()
        
        self._initialized = True
        logger.info("Training Scheduler initialized")
    
    def schedule_job(self, job: TrainingJob, job_id: str = None) -> Dict[str, Any]:
        """
        Schedule a new training job using a TrainingJob object
        
        Args:
            job: TrainingJob object containing job details
            job_id: Optional custom job ID (generated if not provided)
            
        Returns:
            Scheduling result
        """
        print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job called: job.model_id={job.model_id}, job_id={job_id}")
        import time
        
        # Generate job ID if not provided
        if job_id is None:
            # 使用随机数替代(zlib.adler32(str(job).encode('utf-8')) & 0xffffffff)，因为dataclass默认不可哈希
            random_suffix = abs((zlib.adler32(str(job.model_id).encode('utf-8')) & 0xffffffff)) % 9000 + 1000
            job_id = f"{job.model_id}_{int(time.time())}_{random_suffix:04d}"
        
        # Convert priority string to JobPriority enum
        priority_map = {
            'low': JobPriority.LOW,
            'normal': JobPriority.NORMAL,
            'high': JobPriority.HIGH,
            'critical': JobPriority.CRITICAL
        }
        priority = priority_map.get(job.priority.lower(), JobPriority.NORMAL)
        
        # Combine data_config and training_params into parameters
        parameters = job.training_params.copy()
        parameters['data_config'] = job.data_config
        
        # Call internal scheduling method
        print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: Calling _schedule_job_internal")
        try:
            result = self._schedule_job_internal(
                job_id=job_id,
                model_ids=[job.model_id],  # Single model per job for now
                parameters=parameters,
                priority=priority
            )
            print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: _schedule_job_internal returned")
            import sys
            sys.stdout.flush()
            # 安全地打印result
            if result is None:
                print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: result is None")
            else:
                print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: result type={type(result)}")
                if isinstance(result, dict):
                    result_summary = {k: v for k, v in result.items() if k in ['success', 'job_id', 'message']}
                    print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: result summary={result_summary}")
                else:
                    print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: result={str(result)[:200]}")
            return result
        except Exception as e:
            print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: EXCEPTION: {e}")
            import traceback
            print(f"[PRINT_DEBUG] TrainingScheduler.schedule_job: traceback: {traceback.format_exc()}")
            raise
    
    def _schedule_job_internal(self, job_id: str, model_ids: List[str], parameters: Dict[str, Any], 
                    priority: JobPriority = None) -> Dict[str, Any]:
        """
        Schedule a new training job
        
        Args:
            job_id: Unique job identifier
            model_ids: List of model IDs to train
            parameters: Training parameters
            priority: Job priority (defaults to config default)
            
        Returns:
            Scheduling result
        """
        print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal called: job_id={job_id}, model_ids={model_ids}")
        print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: Entering method")
        import sys
        sys.stdout.flush()
        with self.job_lock:
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: Acquired lock")
            sys.stdout.flush()
            # Check if job already exists
            if job_id in self.jobs:
                print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: Job already exists, returning error")
                return {
                    'success': False,
                    'message': f'Job {job_id} already exists',
                    'job_id': job_id
                }
            
            # Determine priority
            job_priority = priority or self.config['default_priority']
            
            # Create job record
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: Creating job_info")
            job_info = {
                'job_id': job_id,
                'model_ids': model_ids,
                'parameters': parameters,
                'priority': job_priority,
                'status': JobStatus.PENDING,
                'created_time': time.time(),
                'scheduled_time': None,
                'start_time': None,
                'end_time': None,
                'progress': 0,
                'metrics': {},
                'logs': [],
                'error': None,
                'retry_count': 0,
                'max_retries': parameters.get('max_retries', 3),
                'estimated_duration': parameters.get('estimated_duration', 3600),  # Default 1 hour
                'dependencies': parameters.get('dependencies', []),
                'resource_requirements': parameters.get('resource_requirements', {}),
                'callback': parameters.get('callback', None)
            }
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: job_info created")
            
            # Add to jobs dictionary
            self.jobs[job_id] = job_info
            
            # Calculate priority score (higher is better)
            priority_score = self._calculate_priority_score(job_priority, model_ids, parameters)
            
            # Add to priority queue
            heapq.heappush(self.job_queue, (-priority_score, job_id))  # Negative for max-heap
            
            # Update statistics
            self.stats['total_jobs_scheduled'] += 1
            
            logger.info(f"Scheduled job {job_id} with priority {job_priority.name} "
                       f"(score: {priority_score:.2f})")
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal success: job_id={job_id}")
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: Returning result dict - MARKER A")
            import sys
            sys.stdout.flush()
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: MARKER B - After flush")
            sys.stdout.flush()
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: BEFORE CREATING RESULT DICT")
            sys.stdout.flush()
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: priority_score={priority_score}, job_queue length={len(self.job_queue)}")
            sys.stdout.flush()
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: Calling _estimate_queue_time")
            sys.stdout.flush()
            estimated_time = self._estimate_queue_time(priority_score)
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: _estimate_queue_time returned: {estimated_time}")
            sys.stdout.flush()
            result_dict = {
                'success': True,
                'job_id': job_id,
                'priority': job_priority.name,
                'priority_score': priority_score,
                'estimated_queue_time': estimated_time,
                'queue_position': len(self.job_queue)
            }
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: BEFORE PRINT RESULT DICT")
            sys.stdout.flush()
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: result_dict={result_dict}")
            sys.stdout.flush()
            print(f"[PRINT_DEBUG] TrainingScheduler._schedule_job_internal: AFTER PRINT RESULT DICT")
            sys.stdout.flush()
            return result_dict
    
    def _calculate_priority_score(self, priority: JobPriority, model_ids: List[str], 
                                 parameters: Dict[str, Any]) -> float:
        """Calculate priority score for job scheduling"""
        # Base score from priority enum
        base_score = priority.value * 10
        
        # Boost for urgent jobs
        if parameters.get('urgent', False):
            base_score += 20
        
        # Adjust based on model types (some models are more critical)
        model_weights = {
            'manager': 5,
            'language': 4,
            'knowledge': 4,
            'vision_image': 3,
            'vision_video': 3,
            'audio': 2,
            'sensor': 1,
            'spatial': 1
        }
        
        model_score = sum(model_weights.get(model_id, 1) for model_id in model_ids)
        base_score += model_score * 0.5
        
        # Adjust based on estimated duration (shorter jobs get slight boost)
        estimated_duration = parameters.get('estimated_duration', 3600)
        if estimated_duration < 1800:  # Less than 30 minutes
            base_score += 5
        elif estimated_duration > 7200:  # More than 2 hours
            base_score -= 3
        
        return max(1.0, base_score)
    
    def _estimate_queue_time(self, priority_score: float) -> float:
        """Estimate queue waiting time based on priority score"""
        # Simple estimation based on current queue and active jobs
        # Note: This method should be called while holding self.job_lock
        print(f"[PRINT_DEBUG] TrainingScheduler._estimate_queue_time called: priority_score={priority_score}")
        queue_length = len(self.job_queue)
        active_count = len(self.active_jobs)
        print(f"[PRINT_DEBUG] TrainingScheduler._estimate_queue_time: queue_length={queue_length}, active_count={active_count}")
        
        if active_count >= self.config['max_concurrent_jobs']:
            # If at capacity, estimate based on job durations
            avg_duration = self.stats['average_job_duration'] or 3600
            print(f"[PRINT_DEBUG] TrainingScheduler._estimate_queue_time: At capacity, avg_duration={avg_duration}")
            estimated_wait = (queue_length * avg_duration) / self.config['max_concurrent_jobs']
        else:
            # If not at capacity, minimal wait
            estimated_wait = max(10, queue_length * 60)  # At least 10 seconds
            print(f"[PRINT_DEBUG] TrainingScheduler._estimate_queue_time: Not at capacity, estimated_wait={estimated_wait}")
        
        # Adjust based on priority (higher priority = shorter wait)
        priority_factor = 1.0 / (priority_score * 0.1 + 1.0)
        print(f"[PRINT_DEBUG] TrainingScheduler._estimate_queue_time: priority_factor={priority_factor}")
        estimated_wait *= priority_factor
        
        print(f"[PRINT_DEBUG] TrainingScheduler._estimate_queue_time: Final estimated_wait={estimated_wait}")
        return estimated_wait
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a scheduled or running job
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            Cancellation result
        """
        with self.job_lock:
            if job_id not in self.jobs:
                return {
                    'success': False,
                    'message': f'Job {job_id} not found'
                }
            
            job_info = self.jobs[job_id]
            current_status = job_info['status']
            
            # Update job status
            if current_status == JobStatus.RUNNING:
                job_info['status'] = JobStatus.STOPPING
                # Signal to stop (actual stopping handled by executor)
                job_info['should_stop'] = True
            elif current_status in [JobStatus.PENDING, JobStatus.SCHEDULED]:
                job_info['status'] = JobStatus.CANCELLED
                job_info['end_time'] = time.time()
                
                # Remove from queue if still there
                # Note: This is simplified; in production would need to search and remove
                new_queue = []
                for score, q_job_id in self.job_queue:
                    if q_job_id != job_id:
                        heapq.heappush(new_queue, (score, q_job_id))
                self.job_queue = new_queue
                
                # Remove from active jobs if present
                if job_id in self.active_jobs:
                    self.active_jobs.remove(job_id)
            else:
                return {
                    'success': False,
                    'message': f'Cannot cancel job in status {current_status.value}'
                }
            
            # Update statistics
            self.stats['jobs_cancelled'] += 1
            
            logger.info(f"Cancelled job {job_id} (was {current_status.value})")
            
            return {
                'success': True,
                'job_id': job_id,
                'previous_status': current_status.value,
                'new_status': job_info['status'].value
            }
    
    def get_scheduled_jobs(self, status_filter: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """
        Get list of scheduled jobs, optionally filtered by status
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of job information dictionaries
        """
        with self.job_lock:
            if status_filter:
                jobs = [job for job in self.jobs.values() if job['status'] == status_filter]
            else:
                jobs = list(self.jobs.values())
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x['created_time'], reverse=True)
            
            return jobs
    
    def get_job_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status and statistics
        
        Returns:
            Queue status information
        """
        with self.job_lock:
            # Count jobs by status
            status_counts = {}
            for job in self.jobs.values():
                status = job['status'].value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Get queue info
            queue_info = []
            for score, job_id in self.job_queue[:10]:  # Top 10 in queue
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    queue_info.append({
                        'job_id': job_id,
                        'priority_score': -score,  # Convert back to positive
                        'status': job['status'].value,
                        'model_ids': job['model_ids'],
                        'wait_time': time.time() - job['created_time']
                    })
            
            return {
                'total_jobs': len(self.jobs),
                'active_jobs': len(self.active_jobs),
                'queued_jobs': len(self.job_queue),
                'status_counts': status_counts,
                'queue_info': queue_info,
                'max_concurrent_jobs': self.config['max_concurrent_jobs'],
                'concurrent_usage': self.model_type_usage,
                'stats': self.stats.copy()
            }
    
    def update_job_priority(self, job_id: str, new_priority: JobPriority) -> Dict[str, Any]:
        """
        Update priority of a pending job
        
        Args:
            job_id: Job ID to update
            new_priority: New priority level
            
        Returns:
            Update result
        """
        with self.job_lock:
            if job_id not in self.jobs:
                return {
                    'success': False,
                    'message': f'Job {job_id} not found'
                }
            
            job_info = self.jobs[job_id]
            current_status = job_info['status']
            
            # Can only update priority for pending/scheduled jobs
            if current_status not in [JobStatus.PENDING, JobStatus.SCHEDULED]:
                return {
                    'success': False,
                    'message': f'Cannot update priority for job in status {current_status.value}'
                }
            
            # Update priority
            old_priority = job_info['priority']
            job_info['priority'] = new_priority
            
            # Recalculate priority score and re-add to queue
            # First, remove from queue (simplified approach)
            new_queue = []
            for score, q_job_id in self.job_queue:
                if q_job_id != job_id:
                    heapq.heappush(new_queue, (score, q_job_id))
            self.job_queue = new_queue
            
            # Re-add with new priority
            new_score = self._calculate_priority_score(
                new_priority, job_info['model_ids'], job_info['parameters']
            )
            heapq.heappush(self.job_queue, (-new_score, job_id))
            
            logger.info(f"Updated job {job_id} priority from {old_priority.name} to {new_priority.name}")
            
            return {
                'success': True,
                'job_id': job_id,
                'old_priority': old_priority.name,
                'new_priority': new_priority.name,
                'new_priority_score': new_score,
                'queue_position': len(self.job_queue)
            }
    
    def estimate_completion_time(self, job_id: str) -> Dict[str, Any]:
        """
        Estimate completion time for a job
        
        Args:
            job_id: Job ID to estimate
            
        Returns:
            Completion time estimation
        """
        with self.job_lock:
            if job_id not in self.jobs:
                return {
                    'success': False,
                    'message': f'Job {job_id} not found'
                }
            
            job_info = self.jobs[job_id]
            current_status = job_info['status']
            
            if current_status == JobStatus.COMPLETED:
                return {
                    'success': True,
                    'job_id': job_id,
                    'status': 'completed',
                    'actual_completion_time': job_info['end_time'],
                    'message': 'Job already completed'
                }
            elif current_status == JobStatus.FAILED:
                return {
                    'success': True,
                    'job_id': job_id,
                    'status': 'failed',
                    'message': 'Job failed'
                }
            elif current_status == JobStatus.CANCELLED:
                return {
                    'success': True,
                    'job_id': job_id,
                    'status': 'cancelled',
                    'message': 'Job was cancelled'
                }
            
            # Estimate based on current status
            now = time.time()
            
            if current_status == JobStatus.RUNNING:
                if job_info['start_time']:
                    elapsed = now - job_info['start_time']
                    progress = job_info.get('progress', 0)
                    
                    if progress > 0:
                        estimated_total = elapsed / (progress / 100.0)
                        remaining = max(0, estimated_total - elapsed)
                    else:
                        # No progress yet, use estimated duration
                        remaining = job_info.get('estimated_duration', 3600)
                    
                    return {
                        'success': True,
                        'job_id': job_id,
                        'status': 'running',
                        'elapsed_time': elapsed,
                        'estimated_remaining': remaining,
                        'estimated_completion': now + remaining,
                        'progress': progress
                    }
            
            elif current_status in [JobStatus.PENDING, JobStatus.SCHEDULED]:
                # Estimate queue time + job duration
                queue_time = self._estimate_queue_time(
                    self._calculate_priority_score(
                        job_info['priority'], 
                        job_info['model_ids'], 
                        job_info['parameters']
                    )
                )
                
                total_time = queue_time + job_info.get('estimated_duration', 3600)
                
                return {
                    'success': True,
                    'job_id': job_id,
                    'status': current_status.value,
                    'estimated_queue_time': queue_time,
                    'estimated_job_duration': job_info.get('estimated_duration', 3600),
                    'estimated_total_time': total_time,
                    'estimated_completion': now + total_time,
                    'queue_position': len([j for j in self.job_queue if j[1] == job_id])
                }
            
            # For other statuses
            return {
                'success': True,
                'job_id': job_id,
                'status': current_status.value,
                'message': 'Cannot estimate completion time for current status'
            }
    
    def optimize_schedule(self) -> Dict[str, Any]:
        """
        Optimize the current job schedule
        
        Returns:
            Optimization result
        """
        with self.job_lock:
            # Record optimization time
            optimization_time = time.time()
            
            # Simple optimization: re-sort queue based on updated priorities
            # In a more advanced implementation, this could consider:
            # - Resource dependencies
            # - Job dependencies
            # - Fairness constraints
            # - Urgency factors
            
            # Get all queued jobs
            queued_jobs = []
            for score, job_id in self.job_queue:
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    queued_jobs.append((job_id, job))
            
            # Recalculate scores and rebuild queue
            new_queue = []
            for job_id, job in queued_jobs:
                # Recalculate priority score (may consider current system state)
                new_score = self._calculate_priority_score(
                    job['priority'], job['model_ids'], job['parameters']
                )
                heapq.heappush(new_queue, (-new_score, job_id))
            
            self.job_queue = new_queue
            
            # Update statistics
            self.stats['last_schedule_optimization'] = optimization_time
            
            logger.info(f"Schedule optimized: {len(new_queue)} jobs re-sorted")
            
            return {
                'success': True,
                'optimized_jobs': len(new_queue),
                'optimization_time': optimization_time,
                'queue_size': len(new_queue)
            }
    
    def start_job_execution(self, job_id: str, executor_func: Callable) -> bool:
        """
        Start execution of a job (called by scheduler when job is dequeued)
        
        Args:
            job_id: Job ID to start
            executor_func: Function to execute the job
            
        Returns:
            True if job started successfully
        """
        with self.job_lock:
            if job_id not in self.jobs:
                logger.error(f"Cannot start non-existent job: {job_id}")
                return False
            
            job_info = self.jobs[job_id]
            
            # Check if job can be started
            if job_info['status'] not in [JobStatus.PENDING, JobStatus.SCHEDULED]:
                logger.warning(f"Job {job_id} in status {job_info['status'].value} cannot be started")
                return False
            
            # Check concurrent limits for model types
            model_counts = self.model_type_usage.copy()
            for model_id in job_info['model_ids']:
                current_count = model_counts.get(model_id, 0)
                limit = self.config['concurrent_model_limit'].get(model_id, 2)
                
                if current_count >= limit:
                    logger.warning(f"Cannot start job {job_id}: model {model_id} at capacity "
                                  f"({current_count}/{limit})")
                    return False
            
            # Update job status
            job_info['status'] = JobStatus.RUNNING
            job_info['start_time'] = time.time()
            job_info['scheduled_time'] = time.time()
            
            # Update active jobs and model usage
            self.active_jobs.add(job_id)
            for model_id in job_info['model_ids']:
                self.model_type_usage[model_id] = self.model_type_usage.get(model_id, 0) + 1
            
            # Start execution in separate thread
            def job_wrapper():
                try:
                    # Execute the job
                    result = executor_func(job_info)
                    
                    # Update job completion
                    with self.job_lock:
                        job_info['status'] = JobStatus.COMPLETED
                        job_info['end_time'] = time.time()
                        job_info['metrics'].update(result.get('metrics', {}))
                        
                        # Update statistics
                        self.stats['jobs_completed'] += 1
                        duration = job_info['end_time'] - job_info['start_time']
                        self.stats['total_training_time'] += duration
                        
                        # Update average duration
                        if self.stats['average_job_duration'] == 0:
                            self.stats['average_job_duration'] = duration
                        else:
                            self.stats['average_job_duration'] = (
                                self.stats['average_job_duration'] * 0.7 + duration * 0.3
                            )
                        
                        # Record wait time
                        wait_time = job_info['start_time'] - job_info['created_time']
                        self.stats['queue_wait_times'].append(wait_time)
                        if len(self.stats['queue_wait_times']) > 100:
                            self.stats['queue_wait_times'] = self.stats['queue_wait_times'][-100:]
                        
                        # Update active jobs and model usage
                        self.active_jobs.remove(job_id)
                        for model_id in job_info['model_ids']:
                            self.model_type_usage[model_id] = max(
                                0, self.model_type_usage.get(model_id, 1) - 1
                            )
                        
                        logger.info(f"Job {job_id} completed successfully in {duration:.1f}s")
                        
                except Exception as e:
                    # Handle job failure
                    with self.job_lock:
                        job_info['status'] = JobStatus.FAILED
                        job_info['end_time'] = time.time()
                        job_info['error'] = str(e)
                        
                        # Update statistics
                        self.stats['jobs_failed'] += 1
                        
                        # Update active jobs and model usage
                        self.active_jobs.remove(job_id)
                        for model_id in job_info['model_ids']:
                            self.model_type_usage[model_id] = max(
                                0, self.model_type_usage.get(model_id, 1) - 1
                            )
                        
                        logger.error(f"Job {job_id} failed: {str(e)}")
                        
                        # Check retry logic
                        if job_info['retry_count'] < job_info['max_retries']:
                            job_info['retry_count'] += 1
                            job_info['status'] = JobStatus.PENDING
                            job_info['error'] = None
                            
                            # Re-add to queue with priority boost
                            priority_boost = JobPriority.HIGH if self.config['enable_priority_boost'] else job_info['priority']
                            new_score = self._calculate_priority_score(
                                priority_boost, job_info['model_ids'], job_info['parameters']
                            )
                            heapq.heappush(self.job_queue, (-new_score, job_id))
                            
                            logger.info(f"Job {job_id} scheduled for retry {job_info['retry_count']}/"
                                       f"{job_info['max_retries']}")
            
            # Start job thread
            job_thread = threading.Thread(target=job_wrapper, name=f"job_{job_id}")
            job_thread.daemon = True
            job_thread.start()
            
            logger.info(f"Started execution of job {job_id}")
            return True
    
    def _scheduling_loop(self):
        """Main scheduling loop - monitors queue and starts jobs when resources available"""
        while self._scheduling_active:
            try:
                with self.job_lock:
                    # Check if we can start more jobs
                    if len(self.active_jobs) < self.config['max_concurrent_jobs'] and self.job_queue:
                        # Get highest priority job
                        score, job_id = heapq.heappop(self.job_queue)
                        
                        # Check if job still exists and is pending
                        if job_id in self.jobs and self.jobs[job_id]['status'] == JobStatus.PENDING:
                            # Mark as scheduled
                            self.jobs[job_id]['status'] = JobStatus.SCHEDULED
                            
                            # In actual implementation, would call start_job_execution here
                            # For now, just log
                            logger.debug(f"Scheduled job {job_id} from queue")
                        else:
                            # Job no longer valid, skip
                            logger.warning(f"Skipping invalid job {job_id} from queue")
                
                # Sleep before next check
                time.sleep(self.config['schedule_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                time.sleep(self.config['schedule_check_interval'])
    
    def update_job_progress(self, job_id: str, progress: float, metrics: Dict[str, Any] = None):
        """
        Update progress and metrics for a running job
        
        Args:
            job_id: Job ID to update
            progress: Progress percentage (0-100)
            metrics: Additional metrics to record
        """
        with self.job_lock:
            if job_id in self.jobs and self.jobs[job_id]['status'] == JobStatus.RUNNING:
                self.jobs[job_id]['progress'] = progress
                if metrics:
                    self.jobs[job_id]['metrics'].update(metrics)
    
    def mark_job_completed(self, job_id: str, metrics: Dict[str, Any] = None) -> bool:
        """
        Mark a job as completed
        
        Args:
            job_id: Job ID to mark as completed
            metrics: Optional final metrics to record
            
        Returns:
            True if job was marked as completed, False otherwise
        """
        print(f"[PRINT_DEBUG] TrainingScheduler.mark_job_completed called: job_id={job_id}, metrics={metrics}")
        with self.job_lock:
            print(f"[PRINT_DEBUG] TrainingScheduler.mark_job_completed: Acquired lock, checking job existence")
            if job_id not in self.jobs:
                logger.warning(f"Cannot mark job {job_id} as completed: job not found")
                print(f"[PRINT_DEBUG] TrainingScheduler.mark_job_completed: Job {job_id} not found")
                return False
            
            job_info = self.jobs[job_id]
            print(f"[PRINT_DEBUG] TrainingScheduler.mark_job_completed: Found job, current status={job_info.get('status')}, start_time={job_info.get('start_time')}")
            job_info['status'] = JobStatus.COMPLETED
            job_info['end_time'] = time.time()
            job_info['progress'] = 100.0
            
            if metrics:
                job_info['metrics'].update(metrics)
            
            # Update statistics
            self.stats['jobs_completed'] += 1
            if job_info['start_time']:
                duration = job_info['end_time'] - job_info['start_time']
                self.stats['total_training_time'] += duration
                
                # Update average duration
                if self.stats['average_job_duration'] == 0:
                    self.stats['average_job_duration'] = duration
                else:
                    self.stats['average_job_duration'] = (
                        self.stats['average_job_duration'] * 0.7 + duration * 0.3
                    )
            
            # Remove from active jobs
            if job_id in self.active_jobs:
                self.active_jobs.remove(job_id)
            
            # Update model type usage
            for model_id in job_info['model_ids']:
                if model_id in self.model_type_usage and self.model_type_usage[model_id] > 0:
                    self.model_type_usage[model_id] -= 1
            
            logger.info(f"Marked job {job_id} as completed")
            print(f"[PRINT_DEBUG] TrainingScheduler.mark_job_completed: Successfully marked job {job_id} as completed")
            return True
    
    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific job
        
        Args:
            job_id: Job ID to get info for
            
        Returns:
            Job information or None if not found
        """
        with self.job_lock:
            if job_id in self.jobs:
                return self.jobs[job_id].copy()
            return None
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Clean up old completed/failed/cancelled jobs
        
        Args:
            max_age_hours: Maximum age in hours to keep jobs
        """
        with self.job_lock:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            jobs_to_remove = []
            for job_id, job_info in self.jobs.items():
                if job_info['status'] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    job_age = current_time - job_info.get('end_time', job_info['created_time'])
                    if job_age > max_age_seconds:
                        jobs_to_remove.append(job_id)
            
            # Remove old jobs
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
            
            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self._scheduling_active = False
        if self.scheduling_thread and self.scheduling_thread.is_alive():
            self.scheduling_thread.join(timeout=5)
        
        # Clean up all jobs
        self.cleanup_old_jobs(max_age_hours=0)
        
        logger.info("Training Scheduler shutdown complete")


# Global instance for easy access
training_scheduler = TrainingScheduler()
