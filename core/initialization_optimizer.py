"""
Model Initialization Optimizer
=============================

Optimizes model initialization time through:
1. Lazy loading of non-critical components
2. Parallel initialization of independent components
3. Background loading of expensive resources
4. Caching and reuse of initialized components
"""

import time
import threading
import concurrent.futures
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os


class InitializationPriority(Enum):
    """Initialization priority levels"""
    CRITICAL = "critical"      # Required for basic operation
    ESSENTIAL = "essential"    # Required for common operations
    IMPORTANT = "important"    # Important but not immediately required
    BACKGROUND = "background"  # Can be loaded in background
    OPTIONAL = "optional"      # Only load if specifically requested


@dataclass
class ComponentProfile:
    """Profile of a component's initialization characteristics"""
    name: str
    init_func: Callable
    priority: InitializationPriority = InitializationPriority.IMPORTANT
    estimated_init_time: float = 0.1  # Estimated initialization time in seconds
    dependencies: List[str] = field(default_factory=list)
    thread_safe: bool = True
    can_lazy_load: bool = True
    memory_footprint_mb: float = 10.0
    is_shared: bool = False  # Can be shared across model instances


@dataclass
class InitializationMetrics:
    """Metrics for initialization performance"""
    total_time: float = 0.0
    component_times: Dict[str, float] = field(default_factory=dict)
    parallel_savings: float = 0.0
    lazy_savings: float = 0.0
    memory_peak_mb: float = 0.0
    success_count: int = 0
    failure_count: int = 0


class InitializationProfiler:
    """Profiles model initialization for optimization analysis"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.current_profile = None
        self.start_time = None
    
    def start_profiling(self, model_name: str):
        """Start profiling a model initialization"""
        self.current_profile = {
            'model': model_name,
            'start_time': time.time(),
            'components': {},
            'phases': {},
            'dependencies': {}
        }
        self.start_time = time.time()
        self.logger.info(f"Starting initialization profiling for {model_name}")
    
    def record_component_start(self, component_name: str):
        """Record start of component initialization"""
        if self.current_profile and component_name not in self.current_profile['components']:
            self.current_profile['components'][component_name] = {
                'start_time': time.time(),
                'end_time': None,
                'duration': None
            }
    
    def record_component_end(self, component_name: str):
        """Record end of component initialization"""
        if self.current_profile and component_name in self.current_profile['components']:
            comp_data = self.current_profile['components'][component_name]
            comp_data['end_time'] = time.time()
            comp_data['duration'] = comp_data['end_time'] - comp_data['start_time']
    
    def record_phase(self, phase_name: str):
        """Record a phase of initialization"""
        if self.current_profile:
            self.current_profile['phases'][phase_name] = time.time()
    
    def add_dependency(self, from_component: str, to_component: str):
        """Record a dependency between components"""
        if self.current_profile:
            if 'dependencies' not in self.current_profile:
                self.current_profile['dependencies'] = {}
            
            if from_component not in self.current_profile['dependencies']:
                self.current_profile['dependencies'][from_component] = []
            
            self.current_profile['dependencies'][from_component].append(to_component)
    
    def end_profiling(self) -> Dict[str, Any]:
        """End profiling and return results"""
        if not self.current_profile:
            return {}
        
        profile = self.current_profile
        profile['total_time'] = time.time() - profile['start_time']
        
        # Calculate component statistics
        component_times = []
        for comp_name, comp_data in profile['components'].items():
            if comp_data['duration'] is not None:
                component_times.append((comp_name, comp_data['duration']))
        
        profile['component_statistics'] = {
            'total_components': len(profile['components']),
            'slow_components': [(n, t) for n, t in component_times if t > 0.1],
            'total_component_time': sum(t for _, t in component_times),
            'avg_component_time': sum(t for _, t in component_times) / len(component_times) if component_times else 0
        }
        
        # Store profile
        model_name = profile['model']
        self.profiles[model_name] = profile
        
        # Generate recommendations
        recommendations = self._generate_recommendations(profile)
        profile['recommendations'] = recommendations
        
        self.logger.info(f"Completed initialization profiling for {model_name}: {profile['total_time']:.2f}s")
        
        self.current_profile = None
        return profile
    
    def _generate_recommendations(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on profile"""
        recommendations = []
        
        # Check for slow components (>100ms)
        slow_components = profile['component_statistics']['slow_components']
        if slow_components:
            lazy_candidates = []
            for comp_name, comp_time in slow_components:
                # Skip components that might be essential
                if comp_time > 0.5:  # Very slow components
                    lazy_candidates.append((comp_name, comp_time))
            
            if lazy_candidates:
                recommendations.append({
                    'type': 'lazy_loading',
                    'components': [name for name, _ in lazy_candidates],
                    'estimated_savings': sum(time for _, time in lazy_candidates),
                    'priority': 'high'
                })
        
        # Check for sequential bottlenecks
        total_time = profile['total_time']
        component_sum = profile['component_statistics']['total_component_time']
        sequential_ratio = component_sum / total_time if total_time > 0 else 0
        
        if sequential_ratio > 0.7:  # Highly sequential
            recommendations.append({
                'type': 'parallel_initialization',
                'current_sequential_ratio': f"{sequential_ratio:.1%}",
                'estimated_savings': total_time * 0.4,  # Estimate 40% savings
                'priority': 'medium'
            })
        
        # Check dependencies for optimization opportunities
        dependencies = profile.get('dependencies', {})
        if dependencies:
            # Look for independent components that could be parallelized
            independent_comps = []
            for comp_name in profile['components']:
                if comp_name not in dependencies or not dependencies[comp_name]:
                    if comp_name in [c[0] for c in slow_components if c[1] > 0.2]:
                        independent_comps.append(comp_name)
            
            if independent_comps:
                recommendations.append({
                    'type': 'dependency_optimization',
                    'independent_components': independent_comps,
                    'estimated_savings': sum(
                        comp_data['duration'] 
                        for comp_name, comp_data in profile['components'].items()
                        if comp_name in independent_comps and comp_data['duration']
                    ) * 0.3,  # Estimate 30% savings from better scheduling
                    'priority': 'low'
                })
        
        return recommendations
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive initialization optimization report"""
        report_lines = [
            "# Model Initialization Optimization Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total models profiled: {len(self.profiles)}",
            ""
        ]
        
        for model_name, profile in self.profiles.items():
            report_lines.extend([
                f"## {model_name}",
                f"**Total initialization time**: {profile['total_time']:.2f}s",
                f"**Components initialized**: {profile['component_statistics']['total_components']}",
                f"**Total component time**: {profile['component_statistics']['total_component_time']:.2f}s",
                f"**Average component time**: {profile['component_statistics']['avg_component_time']:.3f}s",
                ""
            ])
            
            # Slow components
            slow_comps = profile['component_statistics']['slow_components']
            if slow_comps:
                report_lines.append("### Slow Components (>100ms):")
                for comp_name, comp_time in slow_comps:
                    report_lines.append(f"- {comp_name}: {comp_time:.3f}s")
                report_lines.append("")
            
            # Recommendations
            recommendations = profile.get('recommendations', [])
            if recommendations:
                report_lines.append("### Optimization Recommendations:")
                for rec in recommendations:
                    report_lines.append(f"- **{rec['type'].replace('_', ' ').title()}**")
                    report_lines.append(f"  - Priority: {rec['priority']}")
                    report_lines.append(f"  - Estimated savings: {rec['estimated_savings']:.2f}s")
                    
                    if 'components' in rec:
                        report_lines.append(f"  - Components: {', '.join(rec['components'])}")
                    if 'independent_components' in rec:
                        report_lines.append(f"  - Independent components: {', '.join(rec['independent_components'])}")
                    
                    report_lines.append("")
            
            report_lines.append("---\n")
        
        # Summary statistics
        total_init_time = sum(p['total_time'] for p in self.profiles.values())
        avg_init_time = total_init_time / len(self.profiles) if self.profiles else 0
        
        report_lines.extend([
            "## Summary Statistics",
            f"**Total profiling time**: {total_init_time:.2f}s",
            f"**Average model initialization**: {avg_init_time:.2f}s",
            f"**Total slow components found**: {sum(len(p['component_statistics']['slow_components']) for p in self.profiles.values())}",
            ""
        ])
        
        # Aggregate recommendations
        all_recommendations = []
        for profile in self.profiles.values():
            all_recommendations.extend(profile.get('recommendations', []))
        
        if all_recommendations:
            report_lines.append("## Aggregate Recommendations")
            
            # Group by type
            by_type = {}
            for rec in all_recommendations:
                rec_type = rec['type']
                if rec_type not in by_type:
                    by_type[rec_type] = []
                by_type[rec_type].append(rec)
            
            for rec_type, recs in by_type.items():
                total_savings = sum(r['estimated_savings'] for r in recs)
                high_priority = sum(1 for r in recs if r['priority'] == 'high')
                
                report_lines.append(f"### {rec_type.replace('_', ' ').title()}")
                report_lines.append(f"- **Total potential savings**: {total_savings:.2f}s")
                report_lines.append(f"- **Recommendations**: {len(recs)}")
                report_lines.append(f"- **High priority**: {high_priority}")
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report


class LazyComponent:
    """Lazy loading wrapper for components"""
    
    def __init__(self, init_func: Callable, name: str = None, 
                 dependencies: List[str] = None, thread_safe: bool = True):
        self.init_func = init_func
        self.name = name or init_func.__name__
        self.dependencies = dependencies or []
        self.thread_safe = thread_safe
        self._value = None
        self._initialized = False
        self._lock = threading.RLock() if thread_safe else None
        self._init_time = 0.0
    
    def __call__(self):
        """Initialize and return the component"""
        return self.get()
    
    def get(self):
        """Get the component value, initializing if necessary"""
        if self._initialized:
            return self._value
        
        if self.thread_safe and self._lock:
            with self._lock:
                return self._initialize()
        else:
            return self._initialize()
    
    def _initialize(self):
        """Initialize the component"""
        if self._initialized:
            return self._value
        
        start_time = time.time()
        try:
            self._value = self.init_func()
            self._initialized = True
            self._init_time = time.time() - start_time
            return self._value
        except Exception as e:
            # Log error but don't crash - component might be optional
            logging.getLogger(__name__).warning(f"Failed to initialize {self.name}: {e}")
            raise
    
    def is_initialized(self):
        """Check if component is initialized"""
        return self._initialized
    
    def reset(self):
        """Reset the component (for testing)"""
        if self.thread_safe and self._lock:
            with self._lock:
                self._value = None
                self._initialized = False
        else:
            self._value = None
            self._initialized = False


class ParallelInitializer:
    """Manages parallel initialization of components"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, os.cpu_count() or 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.tasks: Dict[str, concurrent.futures.Future] = {}
        self.results: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def submit_component(self, component_name: str, init_func: Callable, 
                        dependencies: List[str] = None) -> concurrent.futures.Future:
        """Submit a component for parallel initialization"""
        # Check if already initialized
        if component_name in self.results:
            future = concurrent.futures.Future()
            future.set_result(self.results[component_name])
            return future
        
        # Check if already submitted
        if component_name in self.tasks:
            return self.tasks[component_name]
        
        # Create wrapper that respects dependencies
        def init_with_deps():
            # Wait for dependencies if any
            if dependencies:
                for dep in dependencies:
                    if dep in self.tasks:
                        self.tasks[dep].result(timeout=30)
            
            # Initialize component
            return init_func()
        
        # Submit task
        future = self.executor.submit(init_with_deps)
        self.tasks[component_name] = future
        
        # Store result when complete
        future.add_done_callback(lambda f: self._store_result(component_name, f))
        
        return future
    
    def _store_result(self, component_name: str, future: concurrent.futures.Future):
        """Store the result of a completed initialization"""
        try:
            result = future.result()
            self.results[component_name] = result
        except Exception as e:
            self.logger.error(f"Failed to initialize {component_name}: {e}")
            self.results[component_name] = None
    
    def wait_for_components(self, component_names: List[str], 
                           timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for specific components to initialize"""
        results = {}
        
        # Submit any components not already submitted
        for name in component_names:
            if name not in self.tasks:
                # Create a dummy future for already initialized components
                if name in self.results:
                    future = concurrent.futures.Future()
                    future.set_result(self.results[name])
                    self.tasks[name] = future
                else:
                    self.logger.warning(f"Component {name} not submitted for initialization")
        
        # Wait for completion
        futures = {name: self.tasks[name] for name in component_names if name in self.tasks}
        
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                self.logger.warning(f"Timeout initializing {name}")
                results[name] = None
            except Exception as e:
                self.logger.error(f"Error initializing {name}: {e}")
                results[name] = None
        
        return results
    
    def initialize_graph(self, components: Dict[str, ComponentProfile], 
                        essential_only: bool = False) -> Dict[str, Any]:
        """Initialize components based on dependency graph"""
        # Filter components if essential_only
        if essential_only:
            components = {k: v for k, v in components.items() 
                         if v.priority in [InitializationPriority.CRITICAL, 
                                          InitializationPriority.ESSENTIAL]}
        
        # Topological sort for dependencies
        sorted_components = self._topological_sort(components)
        
        # Submit components in dependency order
        for comp_name in sorted_components:
            if comp_name in components:
                comp = components[comp_name]
                self.submit_component(comp_name, comp.init_func, comp.dependencies)
        
        # Wait for all components
        return self.wait_for_components(list(components.keys()), timeout=60)
    
    def _topological_sort(self, components: Dict[str, ComponentProfile]) -> List[str]:
        """Topological sort of components based on dependencies"""
        graph = {name: set(comp.dependencies) for name, comp in components.items()}
        
        # Kahn's algorithm
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        # Queue of nodes with no incoming edges
        queue = [node for node in graph if in_degree[node] == 0]
        sorted_nodes = []
        
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        # Check for cycles
        if len(sorted_nodes) != len(graph):
            self.logger.warning("Dependency cycle detected, using original order")
            return list(components.keys())
        
        return sorted_nodes
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)


class ModelInitializationOptimizer:
    """Main optimizer class for model initialization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.profiler = InitializationProfiler()
        self.parallel_initializer = ParallelInitializer(
            max_workers=self.config.get('max_parallel_workers', 4)
        )
        
        # State
        self.optimized_models: Dict[str, Any] = {}
        self.component_cache: Dict[str, Any] = {}
        self.warm_up_tasks: Dict[str, threading.Thread] = {}
        
        # Configuration
        self.enable_lazy_loading = self.config.get('enable_lazy_loading', True)
        self.enable_parallel_init = self.config.get('enable_parallel_init', True)
        self.enable_profiling = self.config.get('enable_profiling', True)
        self.enable_warm_up = self.config.get('enable_warm_up', True)
        
        self.logger.info("ModelInitializationOptimizer initialized")
    
    def optimize_model_initialization(self, model_class, model_id: str, 
                                     config: Optional[Dict[str, Any]] = None) -> Any:
        """Create an optimized instance of a model"""
        start_time = time.time()
        
        # Start profiling
        if self.enable_profiling:
            self.profiler.start_profiling(model_id)
        
        try:
            # Apply optimization configuration
            optimized_config = config or {}
            optimized_config.update({
                'optimized_initialization': True,
                'test_mode': optimized_config.get('test_mode', False)
            })
            
            # Create model instance with minimal initialization
            self.profiler.record_phase('create_instance')
            model = model_class(optimized_config)
            
            # Store reference
            self.optimized_models[model_id] = model
            
            # Analyze model components for optimization
            if hasattr(model, '_analyze_components_for_optimization'):
                component_profiles = model._analyze_components_for_optimization()
            else:
                component_profiles = self._analyze_model_components(model)
            
            # Initialize components based on optimization strategy
            if self.enable_parallel_init:
                self._initialize_components_parallel(model, component_profiles)
            else:
                self._initialize_components_sequential(model, component_profiles)
            
            # Start background warm-up if enabled
            if self.enable_warm_up and not optimized_config.get('test_mode', False):
                self._start_warm_up(model, model_id)
            
            # End profiling
            if self.enable_profiling:
                profile = self.profiler.end_profiling()
                init_time = time.time() - start_time
                self.logger.info(f"Optimized initialization of {model_id} completed in {init_time:.2f}s")
                
                # Log recommendations
                for rec in profile.get('recommendations', []):
                    self.logger.info(f"Recommendation for {model_id}: {rec['type']} "
                                   f"(est. savings: {rec['estimated_savings']:.2f}s)")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to optimize initialization of {model_id}: {e}")
            
            # Fallback to normal initialization
            self.logger.info(f"Falling back to normal initialization for {model_id}")
            return model_class(config or {})
    
    def _analyze_model_components(self, model) -> Dict[str, ComponentProfile]:
        """Analyze model components for optimization opportunities"""
        components = {}
        
        # Common components to check
        common_attrs = [
            'external_api_service',
            'stream_manager', 
            'data_processor',
            'multi_modal_processor',
            'agi_self_learning',
            'agi_emotion_awareness',
            'unified_cognitive_arch',
            'context_memory',
            'meta_learning_system',
            'from_scratch_trainer'
        ]
        
        for attr_name in common_attrs:
            if hasattr(model, attr_name):
                # Check if it's already initialized
                attr_value = getattr(model, attr_name, None)
                
                if attr_value is None:
                    # Not initialized yet - good candidate for lazy loading
                    components[attr_name] = ComponentProfile(
                        name=attr_name,
                        init_func=lambda m=model, a=attr_name: getattr(m, a),
                        priority=self._determine_priority(attr_name),
                        can_lazy_load=True
                    )
        
        return components
    
    def _determine_priority(self, component_name: str) -> InitializationPriority:
        """Determine initialization priority for a component"""
        priority_map = {
            'data_processor': InitializationPriority.CRITICAL,
            'stream_manager': InitializationPriority.ESSENTIAL,
            'external_api_service': InitializationPriority.IMPORTANT,
            'multi_modal_processor': InitializationPriority.IMPORTANT,
            'agi_self_learning': InitializationPriority.BACKGROUND,
            'agi_emotion_awareness': InitializationPriority.BACKGROUND,
            'unified_cognitive_arch': InitializationPriority.BACKGROUND,
            'context_memory': InitializationPriority.IMPORTANT,
            'meta_learning_system': InitializationPriority.BACKGROUND,
            'from_scratch_trainer': InitializationPriority.OPTIONAL
        }
        
        return priority_map.get(component_name, InitializationPriority.IMPORTANT)
    
    def _initialize_components_parallel(self, model, component_profiles: Dict[str, ComponentProfile]):
        """Initialize components in parallel"""
        self.profiler.record_phase('start_parallel_init')
        
        # Separate components by priority
        critical_essential = {k: v for k, v in component_profiles.items() 
                             if v.priority in [InitializationPriority.CRITICAL, 
                                              InitializationPriority.ESSENTIAL]}
        
        background_optional = {k: v for k, v in component_profiles.items() 
                              if v.priority in [InitializationPriority.BACKGROUND,
                                               InitializationPriority.OPTIONAL]}
        
        # Initialize critical/essential components first
        if critical_essential:
            self.logger.debug(f"Initializing {len(critical_essential)} critical/essential components")
            results = self.parallel_initializer.initialize_graph(critical_essential)
            
            # Apply results to model
            for comp_name, value in results.items():
                if value is not None:
                    setattr(model, comp_name, value)
        
        # Start background/optional components asynchronously
        if background_optional:
            self.logger.debug(f"Starting {len(background_optional)} background components")
            
            for comp_name, profile in background_optional.items():
                if profile.can_lazy_load and self.enable_lazy_loading:
                    # Wrap in lazy loader
                    lazy_component = LazyComponent(profile.init_func, comp_name)
                    setattr(model, comp_name, lazy_component)
                else:
                    # Submit for background initialization
                    self.parallel_initializer.submit_component(
                        comp_name, profile.init_func, profile.dependencies
                    )
    
    def _initialize_components_sequential(self, model, component_profiles: Dict[str, ComponentProfile]):
        """Initialize components sequentially (fallback)"""
        self.profiler.record_phase('start_sequential_init')
        
        # Sort by priority
        sorted_components = sorted(
            component_profiles.items(),
            key=lambda x: (
                0 if x[1].priority == InitializationPriority.CRITICAL else
                1 if x[1].priority == InitializationPriority.ESSENTIAL else
                2 if x[1].priority == InitializationPriority.IMPORTANT else
                3 if x[1].priority == InitializationPriority.BACKGROUND else 4
            )
        )
        
        # Initialize in priority order
        for comp_name, profile in sorted_components:
            if profile.priority in [InitializationPriority.CRITICAL, 
                                   InitializationPriority.ESSENTIAL]:
                # Initialize immediately
                self.profiler.record_component_start(comp_name)
                try:
                    value = profile.init_func()
                    setattr(model, comp_name, value)
                    self.profiler.record_component_end(comp_name)
                except Exception as e:
                    self.logger.warning(f"Failed to initialize {comp_name}: {e}")
            elif profile.can_lazy_load and self.enable_lazy_loading:
                # Wrap in lazy loader
                lazy_component = LazyComponent(profile.init_func, comp_name)
                setattr(model, comp_name, lazy_component)
    
    def _start_warm_up(self, model, model_id: str):
        """Start background warm-up of model"""
        def warm_up_task():
            try:
                self.logger.info(f"Starting warm-up for {model_id}")
                start_time = time.time()
                
                # Common warm-up operations
                warm_up_operations = [
                    ('forward', lambda: model.forward(torch.zeros(1, 10)) if hasattr(model, 'forward') else None),
                    ('process', lambda: model.process("test") if hasattr(model, 'process') else None),
                    ('generate', lambda: model.generate("test") if hasattr(model, 'generate') else None),
                ]
                
                for op_name, op_func in warm_up_operations:
                    try:
                        op_func()
                        self.logger.debug(f"Warmed up {op_name} for {model_id}")
                    except Exception as e:
                        self.logger.debug(f"Warm-up {op_name} failed for {model_id}: {e}")
                
                warm_up_time = time.time() - start_time
                self.logger.info(f"Completed warm-up for {model_id} in {warm_up_time:.2f}s")
                
            except Exception as e:
                self.logger.warning(f"Warm-up failed for {model_id}: {e}")
        
        # Start warm-up in background thread
        thread = threading.Thread(target=warm_up_task, daemon=True)
        thread.start()
        self.warm_up_tasks[model_id] = thread
    
    def wait_for_warm_up(self, model_id: str, timeout: float = 30.0):
        """Wait for model warm-up to complete"""
        if model_id in self.warm_up_tasks:
            thread = self.warm_up_tasks[model_id]
            thread.join(timeout=timeout)
            if thread.is_alive():
                self.logger.warning(f"Warm-up timeout for {model_id}")
                return False
            return True
        return True  # No warm-up task means nothing to wait for
    
    def generate_optimization_report(self, output_path: Optional[str] = None) -> str:
        """Generate optimization report"""
        return self.profiler.generate_report(output_path)
    
    def shutdown(self):
        """Shutdown optimizer resources"""
        self.parallel_initializer.shutdown()
        
        # Wait for warm-up tasks
        for model_id, thread in self.warm_up_tasks.items():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.logger.info("ModelInitializationOptimizer shutdown complete")


# Global optimizer instance
_global_optimizer: Optional[ModelInitializationOptimizer] = None

def get_optimizer(config: Optional[Dict[str, Any]] = None) -> ModelInitializationOptimizer:
    """Get global optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ModelInitializationOptimizer(config)
    return _global_optimizer

def create_optimized_model(model_class, model_id: str, 
                          config: Optional[Dict[str, Any]] = None) -> Any:
    """Create an optimized model instance using global optimizer"""
    optimizer = get_optimizer()
    return optimizer.optimize_model_initialization(model_class, model_id, config)

def generate_initialization_report(output_path: Optional[str] = None) -> str:
    """Generate initialization optimization report"""
    optimizer = get_optimizer()
    return optimizer.generate_optimization_report(output_path)