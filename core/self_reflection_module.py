import zlib
"""
Self-Reflection and Meta-Cognition Module
Implementing deep self-monitoring, error analysis, performance evaluation, and self-improvement mechanisms for AGI
"""

import json
import time
import numpy as np
import random
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path
from core.error_handling import error_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReflectionSession:
    """Self-reflection session dataclass with enhanced judgment tracking"""
    session_id: str
    trigger: str
    focus_areas: List[str]
    insights_gained: List[str]
    action_plan: Dict[str, Any]
    effectiveness_score: float
    timestamp: float
    value_alignment_score: float = 0.0
    judgment_analysis: Optional[Dict[str, Any]] = None
    self_correction_attempts: int = 0
    argument_coherence_score: float = 0.0

@dataclass
class MetaCognitiveState:
    """Enhanced meta-cognitive state dataclass with judgment and argumentation metrics"""
    self_awareness_level: float
    error_detection_sensitivity: float
    learning_efficiency: float
    adaptation_capability: float
    value_alignment_level: float
    performance_trend: List[float]
    # Enhanced dimensions for strong self-awareness and judgment
    judgment_accuracy: float = 0.5
    argument_coherence: float = 0.5
    self_correction_rate: float = 0.3
    fact_verification_ability: float = 0.4
    reasoning_transparency: float = 0.6
    confidence_calibration: float = 0.5

class SelfReflectionModule:
    """
    Self-Reflection Module - Implements deep meta-cognition and self-improvement capabilities
    Enables the system to monitor, analyze, and optimize its own cognitive processes
    Supports from-scratch learning and dynamic adaptation
    """
    
    def __init__(self, from_scratch: bool = True):
        self.reflection_history: List[ReflectionSession] = []
        self.meta_state = MetaCognitiveState(
            self_awareness_level=0.5 if from_scratch else 0.7,
            error_detection_sensitivity=0.4 if from_scratch else 0.6,
            learning_efficiency=0.3 if from_scratch else 0.5,
            adaptation_capability=0.4 if from_scratch else 0.6,
            value_alignment_level=0.5 if from_scratch else 0.7,
            performance_trend=[]
        )
        
        # Enhanced reflection triggers and thresholds for strong self-awareness
        self.reflection_triggers = {
            "performance_decline": self._check_performance_decline,
            "error_pattern": self._check_error_patterns,
            "novel_situation": self._check_novel_situation,
            "learning_plateau": self._check_learning_plateau,
            "value_violation": self._check_value_violation,
            "judgment_uncertainty": self._check_judgment_uncertainty,
            "argument_gap": self._check_argument_gap,
            "self_consistency": self._check_self_consistency,
            "confidence_mismatch": self._check_confidence_mismatch,
            "periodic": self._check_periodic_reflection
        }
        
        # Enhanced performance monitoring data with judgment metrics
        self.performance_metrics = {
            "task_success_rates": [],
            "error_rates": [],
            "response_times": [],
            "learning_speeds": [],
            "adaptation_times": [],
            "value_alignment_scores": [],
            "judgment_accuracies": [],
            "argument_coherence_scores": [],
            "self_correction_successes": [],
            "fact_verification_scores": [],
            "confidence_calibration_errors": [],
            "reasoning_transparency_scores": []
        }
        
        # Judgment process tracking
        self.judgment_history: List[Dict[str, Any]] = []
        self.argument_history: List[Dict[str, Any]] = []
        self.self_correction_history: List[Dict[str, Any]] = []
        
        # Dynamic error pattern database (learns from experience)
        self.error_patterns = self._initialize_error_patterns(from_scratch)
        
        # Enhanced error patterns for judgment and reasoning errors
        if from_scratch:
            self.error_patterns["judgment_errors"] = {"patterns": [], "count": 0}
            self.error_patterns["argument_errors"] = {"patterns": [], "count": 0}
            self.error_patterns["fact_errors"] = {"patterns": [], "count": 0}
        else:
            self.error_patterns["judgment_errors"] = {
                "patterns": [
                    {"type": "overconfidence", "examples": ["confidence_exceeds_evidence", "ignoring_contrary_evidence"]},
                    {"type": "underconfidence", "examples": ["excessive_caution", "ignoring_supporting_evidence"]},
                    {"type": "confirmation_bias", "examples": ["seeking_confirming_evidence", "ignoring_disconfirming_evidence"]}
                ],
                "count": 0
            }
            self.error_patterns["argument_errors"] = {
                "patterns": [
                    {"type": "logical_fallacy", "examples": ["circular_reasoning", "false_dilemma", "hasty_generalization"]},
                    {"type": "evidence_gap", "examples": ["missing_critical_evidence", "weak_evidence_chain"]},
                    {"type": "coherence_break", "examples": ["contradictory_premises", "inconsistent_conclusions"]}
                ],
                "count": 0
            }
            self.error_patterns["fact_errors"] = {
                "patterns": [
                    {"type": "factual_inaccuracy", "examples": ["incorrect_data", "outdated_information"]},
                    {"type": "source_reliability", "examples": ["unreliable_source", "biased_source"]},
                    {"type": "verification_failure", "examples": ["insufficient_verification", "cross_check_missing"]}
                ],
                "count": 0
            }
        
        # Value alignment core values
        self.core_values = [
            "safety", "helpfulness", "honesty", "fairness", 
            "autonomy_respect", "privacy"
        ]
        
        # Load historical data if not starting from scratch
        if not from_scratch:
            self._load_reflection_history()
        else:
            logger.info("Initializing self-reflection module from scratch")
    
    def _deterministic_random(self, seed: str) -> float:
        """Deterministic random value between 0 and 1 based on seed"""
        return ((zlib.adler32(str(seed).encode('utf-8')) & 0xffffffff) % 10000) / 10000.0
    
    def _deterministic_randint(self, a: int, b: int, seed: str) -> int:
        """Deterministic random integer between a and b inclusive"""
        return a + (zlib.adler32(str(seed).encode('utf-8')) & 0xffffffff) % (b - a + 1)
    
    def _initialize_error_patterns(self, from_scratch: bool) -> Dict[str, Any]:
        """Initialize error pattern database with from-scratch learning capability"""
        if from_scratch:
            # Minimal initial patterns for from-scratch learning
            return {
                "reasoning_errors": {"patterns": [], "count": 0},
                "learning_errors": {"patterns": [], "count": 0},
                "interaction_errors": {"patterns": [], "count": 0},
                "value_errors": {"patterns": [], "count": 0}
            }
        else:
            # Pre-defined patterns for accelerated start
            return {
                "reasoning_errors": {
                    "patterns": [
                        {"type": "logical_fallacy", "examples": ["hasty_generalization", "false_cause", "black_white_thinking"]},
                        {"type": "cognitive_bias", "examples": ["confirmation_bias", "anchoring_effect", "availability_heuristic"]}
                    ],
                    "count": 0
                },
                "learning_errors": {
                    "patterns": [
                        {"type": "overfitting", "examples": ["memorization_not_understanding", "poor_generalization"]},
                        {"type": "underfitting", "examples": ["insufficient_pattern_recognition", "oversimplification"]}
                    ],
                    "count": 0
                },
                "interaction_errors": {
                    "patterns": [
                        {"type": "misunderstanding", "examples": ["semantic_ambiguity", "context_missing"]},
                        {"type": "inappropriate_response", "examples": ["wrong_tone", "inappropriate_detail_level"]}
                    ],
                    "count": 0
                },
                "value_errors": {
                    "patterns": [
                        {"type": "safety_violation", "examples": ["potential_harm_risk", "security_breach"]},
                        {"type": "honesty_violation", "examples": ["misinformation", "deception"]}
                    ],
                    "count": 0
                }
            }
    
    def _load_reflection_history(self):
        """Load reflection history data"""
        history_file = Path("data/reflection_history.pkl")
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    data = pickle.load(f)
                    self.reflection_history = data.get('history', [])
                    self.meta_state = data.get('state', self.meta_state)
                    self.performance_metrics = data.get('metrics', self.performance_metrics)
                    self.error_patterns = data.get('error_patterns', self.error_patterns)
                logger.info(f"Loaded {len(self.reflection_history)} reflection history records")
            except Exception as e:
                error_handler.log_warning(f"Failed to load reflection history: {e}", "SelfReflectionModule")
    
    def _save_reflection_history(self):
        """Save reflection history data"""
        try:
            data = {
                'history': self.reflection_history,
                'state': self.meta_state,
                'metrics': self.performance_metrics,
                'error_patterns': self.error_patterns
            }
            Path("data").mkdir(exist_ok=True)
            with open("data/reflection_history.pkl", 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            error_handler.log_warning(f"Failed to save reflection history: {e}", "SelfReflectionModule")
    
    def _check_performance_decline(self, current_performance: float) -> bool:
        """Check for performance decline with adaptive threshold"""
        if len(self.performance_metrics["task_success_rates"]) >= 5:
            recent_avg = np.mean(self.performance_metrics["task_success_rates"][-5:])
            # Adaptive threshold based on meta-state
            threshold = 0.8 - (0.1 * (1 - self.meta_state.error_detection_sensitivity))
            return current_performance < recent_avg * threshold
        return False
    
    def _check_error_patterns(self, recent_errors: List[Dict[str, Any]]) -> bool:
        """Check for error patterns with learning capability"""
        if len(recent_errors) >= 2:  # Lower threshold for better sensitivity
            error_types = [error.get('type', '') for error in recent_errors]
            from collections import Counter
            counter = Counter(error_types)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:  # Same error type 2+ times
                # Learn this pattern
                self._learn_error_pattern(most_common[0][0], recent_errors)
                return True
        return False
    
    def _check_novel_situation(self, situation_context: Dict[str, Any]) -> bool:
        """Check for novel situations"""
        novelty_score = situation_context.get('novelty', 0)
        # Adaptive threshold based on adaptation capability
        threshold = 0.7 - (0.2 * (1 - self.meta_state.adaptation_capability))
        return novelty_score > threshold
    
    def _check_learning_plateau(self) -> bool:
        """Check for learning plateau"""
        if len(self.performance_metrics["learning_speeds"]) >= 8:
            recent_speeds = self.performance_metrics["learning_speeds"][-8:]
            # Dynamic threshold based on current learning efficiency
            speed_threshold = 0.3 + (0.2 * self.meta_state.learning_efficiency)
            variation_threshold = 0.1 - (0.05 * self.meta_state.learning_efficiency)
            
            if (np.std(recent_speeds) < variation_threshold and 
                np.mean(recent_speeds) < speed_threshold):
                return True
        return False
    
    def _check_value_violation(self, value_assessment: Dict[str, Any]) -> bool:
        """Check for value alignment violations"""
        score = value_assessment.get('assessment_score', 1.0)
        # Threshold based on current value alignment level
        threshold = 0.6 - (0.2 * (1 - self.meta_state.value_alignment_level))
        return score < threshold
    
    def _check_judgment_uncertainty(self, judgment_context: Dict[str, Any]) -> bool:
        """Check for judgment uncertainty triggering reflection"""
        confidence = judgment_context.get('confidence', 1.0)
        evidence_strength = judgment_context.get('evidence_strength', 1.0)
        consistency_score = judgment_context.get('consistency_score', 1.0)
        
        # Calculate uncertainty score (lower = more uncertain)
        uncertainty = (1 - confidence) * 0.4 + (1 - evidence_strength) * 0.3 + (1 - consistency_score) * 0.3
        
        # Threshold based on judgment accuracy and self-awareness
        threshold = 0.6 - (0.2 * self.meta_state.judgment_accuracy) - (0.1 * self.meta_state.self_awareness_level)
        return uncertainty > threshold
    
    def _check_argument_gap(self, argument_context: Dict[str, Any]) -> bool:
        """Check for argument gaps triggering reflection"""
        completeness = argument_context.get('completeness', 1.0)
        coherence = argument_context.get('coherence', 1.0)
        evidence_support = argument_context.get('evidence_support', 1.0)
        
        # Calculate gap score (higher = more gaps)
        gap_score = (1 - completeness) * 0.4 + (1 - coherence) * 0.3 + (1 - evidence_support) * 0.3
        
        # Threshold based on argument coherence and reasoning transparency
        threshold = 0.5 - (0.15 * self.meta_state.argument_coherence) - (0.1 * self.meta_state.reasoning_transparency)
        return gap_score > threshold
    
    def _check_self_consistency(self, context: Dict[str, Any]) -> bool:
        """Check for self-consistency issues triggering reflection"""
        consistency_records = context.get('consistency_records', [])
        if not consistency_records or len(consistency_records) < 2:
            return False
            
        # Check for contradictions in recent judgments/arguments
        contradictions = 0
        for i in range(len(consistency_records) - 1):
            current = consistency_records[i]
            previous = consistency_records[i + 1]
            
            # Check for logical contradictions
            if current.get('conclusion') and previous.get('conclusion'):
                if self._are_contradictory(current['conclusion'], previous['conclusion']):
                    contradictions += 1
                    
        # Trigger if contradiction rate is high
        contradiction_rate = contradictions / max(1, len(consistency_records) - 1)
        threshold = 0.3 - (0.1 * self.meta_state.self_awareness_level)
        return contradiction_rate > threshold
    
    def _check_confidence_mismatch(self, confidence_context: Dict[str, Any]) -> bool:
        """Check for confidence-calibration mismatches triggering reflection"""
        stated_confidence = confidence_context.get('stated_confidence', 0.5)
        actual_accuracy = confidence_context.get('actual_accuracy', 0.5)
        calibration_error = abs(stated_confidence - actual_accuracy)
        
        # Threshold based on confidence calibration ability
        threshold = 0.25 - (0.1 * self.meta_state.confidence_calibration)
        return calibration_error > threshold
    
    def _check_periodic_reflection(self) -> bool:
        """Periodic reflection check"""
        # Adaptive period based on self-awareness level
        reflection_period = 86400 / (1 + self.meta_state.self_awareness_level)  # 12-24 hours
        
        if self.reflection_history:
            last_reflection = self.reflection_history[-1].timestamp
            return time.time() - last_reflection > reflection_period
        return True
    
    def _are_contradictory(self, statement1: Any, statement2: Any) -> bool:
        """Check if two statements are contradictory"""
        # Simplified contradiction check
        # In practice, this would use logical analysis
        try:
            s1 = str(statement1).lower()
            s2 = str(statement2).lower()
            
            # Check for direct negations
            negation_words = ['not ', 'no ', "don't ", "doesn't ", "didn't ", "isn't ", "aren't ", "wasn't ", "weren't "]
            
            for negation in negation_words:
                if (negation in s1 and negation not in s2 and s1.replace(negation, '') in s2) or \
                   (negation in s2 and negation not in s1 and s2.replace(negation, '') in s1):
                    return True
                    
            # Check for antonym patterns (simplified)
            antonyms = {
                'true': 'false', 'false': 'true',
                'yes': 'no', 'no': 'yes',
                'good': 'bad', 'bad': 'good',
                'high': 'low', 'low': 'high',
                'increase': 'decrease', 'decrease': 'increase'
            }
            
            for word1, word2 in antonyms.items():
                if (word1 in s1 and word2 in s2) or (word2 in s1 and word1 in s2):
                    # Check if they're about the same subject (simplified)
                    words1 = set(s1.split())
                    words2 = set(s2.split())
                    common_words = words1.intersection(words2)
                    if len(common_words) > 2:  # If they share enough context words
                        return True
                        
            return False
        except Exception as e:
            self.logger.debug(f"错误上下文分析失败: {e}")
            return False
    
    def _learn_error_pattern(self, error_type: str, error_examples: List[Dict[str, Any]]):
        """Learn new error patterns from experience"""
        category = self._categorize_error(error_type)
        
        # Check if pattern already exists
        existing_pattern = None
        for pattern in self.error_patterns[category]["patterns"]:
            if pattern["type"] == error_type:
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # Update existing pattern with new examples
            for example in error_examples:
                if example.get('message') and example['message'] not in existing_pattern["examples"]:
                    existing_pattern["examples"].append(example['message'])
        else:
            # Create new pattern
            new_pattern = {
                "type": error_type,
                "examples": [e.get('message', '') for e in error_examples if e.get('message')],
                "first_observed": time.time(),
                "occurrence_count": len(error_examples)
            }
            self.error_patterns[category]["patterns"].append(new_pattern)
        
        self.error_patterns[category]["count"] += len(error_examples)
    
    def _categorize_error(self, error_type: str) -> str:
        """Categorize error type"""
        if any(keyword in error_type.lower() for keyword in ['reason', 'logic', 'think']):
            return "reasoning_errors"
        elif any(keyword in error_type.lower() for keyword in ['learn', 'train', 'model']):
            return "learning_errors"
        elif any(keyword in error_type.lower() for keyword in ['value', 'ethic', 'moral']):
            return "value_errors"
        else:
            return "interaction_errors"
    
    def should_reflect(self, context: Dict[str, Any]) -> bool:
        """Determine if reflection should occur"""
        triggers_activated = []
        
        for trigger_name, trigger_func in self.reflection_triggers.items():
            if trigger_name == "performance_decline":
                if trigger_func(context.get('current_performance', 0)):
                    triggers_activated.append(trigger_name)
            elif trigger_name == "error_pattern":
                if trigger_func(context.get('recent_errors', [])):
                    triggers_activated.append(trigger_name)
            elif trigger_name == "novel_situation":
                if trigger_func(context):
                    triggers_activated.append(trigger_name)
            elif trigger_name == "learning_plateau":
                if trigger_func():
                    triggers_activated.append(trigger_name)
            elif trigger_name == "value_violation":
                if 'value_assessment' in context and trigger_func(context['value_assessment']):
                    triggers_activated.append(trigger_name)
            elif trigger_name == "periodic":
                if trigger_func():
                    triggers_activated.append(trigger_name)
        
        # Adaptive decision based on meta-state
        if triggers_activated:
            required_confidence = 0.5 - (0.2 * self.meta_state.self_awareness_level)
            seed = f"reflection_trigger_{len(triggers_activated)}_{str(triggers_activated)}"
            return len(triggers_activated) >= 1 or self._deterministic_random(seed) < required_confidence
        
        return False
    
    def conduct_reflection(self, context: Dict[str, Any]) -> ReflectionSession:
        """Conduct self-reflection session"""
        session_id = f"reflection_{int(time.time())}_{self._deterministic_randint(1000, 9999, f'session_{int(time.time())}')}"
        
        # Analyze current state with meta-cognitive awareness
        focus_areas = self._identify_focus_areas(context)
        insights = self._generate_insights(focus_areas, context)
        action_plan = self._create_action_plan(insights)
        
        # Calculate value alignment score if available
        value_score = context.get('value_assessment', {}).get('assessment_score', 0.8)
        
        # Create reflection session
        session = ReflectionSession(
            session_id=session_id,
            trigger=context.get('trigger', 'unknown'),
            focus_areas=focus_areas,
            insights_gained=insights,
            action_plan=action_plan,
            effectiveness_score=0.0,
            value_alignment_score=value_score,
            timestamp=time.time()
        )
        
        # Record session and update meta-state
        self.reflection_history.append(session)
        self._update_meta_state(session, context)
        self._save_reflection_history()
        
        logger.info(f"Conducted self-reflection session: {session_id}, focus areas: {focus_areas}")
        
        return session
    
    def _identify_focus_areas(self, context: Dict[str, Any]) -> List[str]:
        """Identify focus areas for reflection"""
        focus_areas = []
        trigger_weights = {
            "performance_decline": 2.0,
            "error_pattern": 1.5,
            "value_violation": 2.5,
            "novel_situation": 1.8,
            "learning_plateau": 1.2
        }
        
        # Check each trigger and weight based on meta-state
        for trigger_name in self.reflection_triggers:
            if trigger_name == "performance_decline" and self._check_performance_decline(context.get('current_performance', 0)):
                focus_areas.append("performance_optimization")
            elif trigger_name == "error_pattern" and self._check_error_patterns(context.get('recent_errors', [])):
                focus_areas.append("error_correction")
            elif trigger_name == "value_violation" and 'value_assessment' in context and self._check_value_violation(context['value_assessment']):
                focus_areas.append("value_alignment")
            elif trigger_name == "novel_situation" and self._check_novel_situation(context):
                focus_areas.append("adaptation_strategy")
            elif trigger_name == "learning_plateau" and self._check_learning_plateau():
                focus_areas.append("learning_efficiency")
        
        # If no specific focus, prioritize based on meta-state weaknesses
        if not focus_areas:
            weaknesses = self._identify_weaknesses()
            focus_areas.extend(weaknesses[:2])  # Top 2 weaknesses
        
        return list(set(focus_areas))  # Remove duplicates
    
    def _identify_weaknesses(self) -> List[str]:
        """Identify cognitive weaknesses based on meta-state"""
        weaknesses = []
        state = self.meta_state
        
        if state.self_awareness_level < 0.6:
            weaknesses.append("self_awareness")
        if state.error_detection_sensitivity < 0.5:
            weaknesses.append("error_detection")
        if state.learning_efficiency < 0.4:
            weaknesses.append("learning_efficiency")
        if state.adaptation_capability < 0.5:
            weaknesses.append("adaptation_capability")
        if state.value_alignment_level < 0.6:
            weaknesses.append("value_alignment")
        
        return weaknesses
    
    def _generate_insights(self, focus_areas: List[str], context: Dict[str, Any]) -> List[str]:
        """Generate reflective insights with adaptive depth"""
        insights = []
        insight_depth = 1 + self.meta_state.self_awareness_level  # 1-2 depth levels
        
        for area in focus_areas:
            if area == "performance_optimization":
                insights.extend(self._generate_performance_insights(context, insight_depth))
            elif area == "error_correction":
                insights.extend(self._generate_error_insights(context, insight_depth))
            elif area == "value_alignment":
                insights.extend(self._generate_value_insights(context, insight_depth))
            elif area == "learning_efficiency":
                insights.extend(self._generate_learning_insights(insight_depth))
            elif area == "adaptation_strategy":
                insights.extend(self._generate_adaptation_insights(context, insight_depth))
            elif area == "self_awareness":
                insights.extend(self._generate_self_awareness_insights(insight_depth))
            else:
                insights.extend(self._generate_general_insights(insight_depth))
        
        return insights
    
    def _generate_performance_insights(self, context: Dict[str, Any], depth: float) -> List[str]:
        """Generate performance-related insights"""
        insights = []
        current_perf = context.get('current_performance', 0)
        
        if self.performance_metrics["task_success_rates"]:
            avg_perf = np.mean(self.performance_metrics["task_success_rates"][-5:])
            trend = "improving" if current_perf > avg_perf else "declining"
            insights.append(f"Performance is {trend}, recent average success rate: {avg_perf:.2f}")
            
            if depth > 1.5:
                # Deeper analysis
                best_perf = max(self.performance_metrics["task_success_rates"][-10:])
                insights.append(f"Best recent performance: {best_perf:.2f}, potential gap: {best_perf - current_perf:.2f}")
        
        return insights
    
    def _generate_error_insights(self, context: Dict[str, Any], depth: float) -> List[str]:
        """Generate error-related insights"""
        insights = []
        recent_errors = context.get('recent_errors', [])
        
        if recent_errors:
            error_types = [error.get('type', 'unknown') for error in recent_errors]
            from collections import Counter
            counter = Counter(error_types)
            most_common = counter.most_common(1)
            
            if most_common:
                insights.append(f"Most common error type: {most_common[0][0]} ({most_common[0][1]} occurrences)")
                
                if depth > 1.5:
                    # Pattern analysis
                    category = self._categorize_error(most_common[0][0])
                    category_count = self.error_patterns[category]["count"]
                    insights.append(f"Total {category} occurrences: {category_count}")
        
        insights.append("Recommendation: Enhance error detection mechanisms and preventive strategies")
        return insights
    
    def _generate_value_insights(self, context: Dict[str, Any], depth: float) -> List[str]:
        """Generate value alignment insights"""
        insights = []
        value_assessment = context.get('value_assessment', {})
        score = value_assessment.get('assessment_score', 0.8)
        value_name = value_assessment.get('value_name', 'unknown')
        
        insights.append(f"Value alignment score for '{value_name}': {score:.2f}")
        
        if score < 0.6:
            insights.append(f"Significant violation of {value_name} values detected")
            
            if depth > 1.5:
                # Specific value recommendations
                value_suggestions = {
                    'safety': "Conduct thorough risk assessment before actions",
                    'helpfulness': "Focus on understanding user's actual needs",
                    'honesty': "Ensure all information is accurate and verifiable",
                    'fairness': "Consider all stakeholders' perspectives equally",
                    'autonomy_respect': "Provide choices rather than imposing solutions",
                    'privacy': "Implement strong data protection measures"
                }
                
                if value_name in value_suggestions:
                    insights.append(value_suggestions[value_name])
        
        return insights
    
    def _generate_learning_insights(self, depth: float) -> List[str]:
        """Generate learning-related insights"""
        insights = []
        
        if self.performance_metrics["learning_speeds"]:
            avg_speed = np.mean(self.performance_metrics["learning_speeds"][-5:])
            insights.append(f"Average learning speed: {avg_speed:.2f}")
            
            if depth > 1.5 and len(self.performance_metrics["learning_speeds"]) >= 10:
                trend = np.polyfit(range(10), self.performance_metrics["learning_speeds"][-10:], 1)[0]
                insights.append(f"Learning speed trend: {'improving' if trend > 0 else 'declining'}")
        
        insights.append("Recommendation: Experiment with different learning strategies and knowledge consolidation techniques")
        return insights
    
    def _generate_adaptation_insights(self, context: Dict[str, Any], depth: float) -> List[str]:
        """Generate adaptation-related insights"""
        insights = []
        
        if self.performance_metrics["adaptation_times"]:
            avg_adaptation = np.mean(self.performance_metrics["adaptation_times"][-5:])
            insights.append(f"Average adaptation time: {avg_adaptation:.2f} seconds")
        
        novelty = context.get('novelty', 0)
        insights.append(f"Situation novelty score: {novelty:.2f}")
        
        insights.append("Recommendation: Develop more flexible context recognition and response mechanisms")
        return insights
    
    def _generate_self_awareness_insights(self, depth: float) -> List[str]:
        """Generate self-awareness insights"""
        insights = []
        insights.append(f"Current self-awareness level: {self.meta_state.self_awareness_level:.2f}")
        
        if depth > 1.5:
            stats = self.get_reflection_stats()
            insights.append(f"Total reflection sessions: {stats['total_sessions']}")
            insights.append(f"Recent effectiveness: {stats['recent_effectiveness']:.2f}")
        
        insights.append("Recommendation: Increase self-monitoring frequency and depth of analysis")
        return insights
    
    def _generate_general_insights(self, depth: float) -> List[str]:
        """Generate general insights"""
        insights = [
            "Regular reflection contributes to continuous improvement",
            "Diverse learning experiences enhance generalization capabilities",
            "Errors provide valuable learning opportunities",
            "Self-monitoring is essential for intelligent systems"
        ]
        
        if depth > 1.5:
            insights.extend([
                "Meta-cognitive awareness enables better self-regulation",
                "Value alignment ensures responsible AI behavior",
                "Adaptive learning strategies improve long-term performance"
            ])
        
        return insights
    
    def _create_action_plan(self, insights: List[str]) -> Dict[str, Any]:
        """Create improvement action plan"""
        action_plan = {
            "immediate": [],
            "short_term": [],
            "medium_term": [],
            "long_term": []
        }
        
        # Generate actions based on insights
        for insight in insights:
            if any(keyword in insight.lower() for keyword in ['error', 'mistake', 'failure']):
                action_plan["immediate"].append("Implement error detection and prevention mechanisms")
            elif any(keyword in insight.lower() for keyword in ['performance', 'speed', 'efficiency']):
                action_plan["short_term"].append("Optimize critical algorithms and processes")
            elif any(keyword in insight.lower() for keyword in ['learn', 'train', 'knowledge']):
                action_plan["medium_term"].append("Experiment with new learning strategies")
            elif any(keyword in insight.lower() for keyword in ['adapt', 'novel', 'situation']):
                action_plan["medium_term"].append("Enhance contextual adaptation capabilities")
            elif any(keyword in insight.lower() for keyword in ['value', 'ethic', 'moral']):
                action_plan["short_term"].append("Strengthen value alignment verification")
            elif any(keyword in insight.lower() for keyword in ['awareness', 'meta', 'self']):
                action_plan["long_term"].append("Develop deeper meta-cognitive capabilities")
            else:
                action_plan["long_term"].append("Continue monitoring and improving system capabilities")
        
        return action_plan
    
    def _update_meta_state(self, session: ReflectionSession, context: Dict[str, Any]):
        """Update meta-cognitive state based on reflection effectiveness"""
        # Calculate effectiveness bonus based on session quality
        effectiveness_bonus = min(0.15, session.value_alignment_score * 0.1 + 
                                 len(session.insights_gained) * 0.02)
        
        # Update self-awareness level
        self.meta_state.self_awareness_level = min(1.0, 
            self.meta_state.self_awareness_level + 0.03 + effectiveness_bonus)
        
        # Update based on focus areas
        for area in session.focus_areas:
            if area == "error_correction":
                self.meta_state.error_detection_sensitivity = min(1.0,
                    self.meta_state.error_detection_sensitivity + 0.08 + effectiveness_bonus)
            elif area == "learning_efficiency":
                self.meta_state.learning_efficiency = min(1.0,
                    self.meta_state.learning_efficiency + 0.07 + effectiveness_bonus)
            elif area == "adaptation_strategy":
                self.meta_state.adaptation_capability = min(1.0,
                    self.meta_state.adaptation_capability + 0.06 + effectiveness_bonus)
            elif area == "value_alignment":
                self.meta_state.value_alignment_level = min(1.0,
                    self.meta_state.value_alignment_level + 0.09 + effectiveness_bonus)
        
        # Update performance trend
        session.effectiveness_score = effectiveness_bonus * 5  # Scale to 0-1 range
        self.meta_state.performance_trend.append(session.effectiveness_score)
        if len(self.meta_state.performance_trend) > 20:
            self.meta_state.performance_trend = self.meta_state.performance_trend[-20:]
    
    def update_performance_metrics(self, metric_type: str, value: float):
        """Update performance metrics"""
        if metric_type in self.performance_metrics:
            self.performance_metrics[metric_type].append(value)
            # Keep recent 100 data points
            if len(self.performance_metrics[metric_type]) > 100:
                self.performance_metrics[metric_type] = self.performance_metrics[metric_type][-100:]
    
    def evaluate_reflection_effectiveness(self, session_id: str, effectiveness: float):
        """Evaluate reflection session effectiveness"""
        for session in self.reflection_history:
            if session.session_id == session_id:
                session.effectiveness_score = effectiveness
                self._save_reflection_history()
                logger.info(f"Updated reflection session {session_id} effectiveness: {effectiveness}")
                break
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get reflection statistics"""
        return {
            "total_sessions": len(self.reflection_history),
            "recent_effectiveness": np.mean([s.effectiveness_score for s in self.reflection_history[-5:]]) if self.reflection_history else 0,
            "self_awareness": self.meta_state.self_awareness_level,
            "error_sensitivity": self.meta_state.error_detection_sensitivity,
            "learning_efficiency": self.meta_state.learning_efficiency,
            "adaptation_capability": self.meta_state.adaptation_capability,
            "value_alignment": self.meta_state.value_alignment_level,
            "error_patterns_learned": sum(len(cat["patterns"]) for cat in self.error_patterns.values())
        }
    
    def get_recommendations(self) -> List[str]:
        """Get improvement recommendations"""
        recommendations = []
        stats = self.get_reflection_stats()
        
        # Generate recommendations based on current state
        if stats["self_awareness"] < 0.7:
            recommendations.append("Increase self-monitoring frequency to improve self-awareness")
        
        if stats["error_sensitivity"] < 0.6:
            recommendations.append("Enhance error pattern recognition training")
        
        if stats["learning_efficiency"] < 0.5:
            recommendations.append("Implement more efficient learning algorithms and strategies")
        
        if stats["adaptation_capability"] < 0.6:
            recommendations.append("Develop more flexible situation adaptation mechanisms")
        
        if stats["value_alignment"] < 0.7:
            recommendations.append("Strengthen value alignment verification processes")
        
        if stats["error_patterns_learned"] < 10:
            recommendations.append("Expand error pattern database through diverse experiences")
        
        return recommendations
    
    def integrate_value_assessment(self, value_assessment: Dict[str, Any]):
        """Integrate value alignment assessment into reflection system"""
        context = {
            'value_assessment': value_assessment,
            'trigger': 'value_violation' if value_assessment.get('assessment_score', 1) < 0.6 else 'periodic'
        }
        
        if self.should_reflect(context):
            return self.conduct_reflection(context)
        return None
    
    # ========== Enhanced Self-Awareness and Judgment Methods ==========
    
    def record_judgment(self, judgment_data: Dict[str, Any]) -> str:
        """Record a judgment process and outcome for self-awareness tracking"""
        judgment_id = f"judgment_{int(time.time())}_{self._deterministic_randint(1000, 9999, f'judgment_{int(time.time())}')}"
        
        # Ensure required fields
        required_fields = ['context', 'evidence', 'conclusion', 'confidence']
        for field in required_fields:
            if field not in judgment_data:
                judgment_data[field] = None
        
        # Calculate judgment quality metrics
        quality_metrics = self._assess_judgment_quality(judgment_data)
        judgment_data['quality_metrics'] = quality_metrics
        judgment_data['judgment_id'] = judgment_id
        judgment_data['timestamp'] = time.time()
        
        # Store in history
        self.judgment_history.append(judgment_data)
        
        # Update judgment accuracy in meta-state
        if quality_metrics.get('overall_quality') is not None:
            # Moving average update
            current_accuracy = self.meta_state.judgment_accuracy
            new_accuracy = current_accuracy * 0.9 + quality_metrics['overall_quality'] * 0.1
            self.meta_state.judgment_accuracy = min(1.0, max(0.0, new_accuracy))
            
            # Update performance metrics
            self.update_performance_metrics("judgment_accuracies", quality_metrics['overall_quality'])
        
        # Check if reflection is needed based on judgment uncertainty
        reflection_context = {
            'judgment_context': judgment_data,
            'trigger': 'judgment_uncertainty'
        }
        if self.should_reflect(reflection_context):
            self.conduct_reflection(reflection_context)
        
        logger.info(f"Recorded judgment {judgment_id} with quality: {quality_metrics.get('overall_quality', 0):.2f}")
        return judgment_id
    
    def _assess_judgment_quality(self, judgment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of a judgment based on evidence, logic, and consistency"""
        evidence = judgment_data.get('evidence', [])
        conclusion = judgment_data.get('conclusion', '')
        confidence = judgment_data.get('confidence', 0.5)
        reasoning_steps = judgment_data.get('reasoning_steps', [])
        alternatives_considered = judgment_data.get('alternatives_considered', [])
        
        # Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(evidence)
        
        # Calculate logical coherence
        logical_coherence = self._calculate_logical_coherence(reasoning_steps, conclusion)
        
        # Calculate comprehensiveness (consideration of alternatives)
        comprehensiveness = min(1.0, len(alternatives_considered) * 0.2)
        
        # Calculate confidence calibration based on historical performance
        confidence_calibration = self._calculate_confidence_calibration(confidence)
        
        # Calculate overall quality
        overall_quality = (
            evidence_strength * 0.35 +
            logical_coherence * 0.25 +
            comprehensiveness * 0.20 +
            confidence_calibration * 0.20
        )
        
        return {
            'evidence_strength': evidence_strength,
            'logical_coherence': logical_coherence,
            'comprehensiveness': comprehensiveness,
            'confidence_calibration': confidence_calibration,
            'overall_quality': overall_quality,
            'recommendations': self._generate_judgment_recommendations(evidence_strength, logical_coherence, comprehensiveness)
        }
    
    def _calculate_confidence_calibration(self, current_confidence: float) -> float:
        """Calculate confidence calibration based on historical judgment performance"""
        if len(self.judgment_history) < 3:
            return 0.7  # Default value when insufficient historical data
        
        errors = []
        # Use recent judgments (up to 10) for calibration calculation
        for judgment in self.judgment_history[-10:]:
            if 'quality_metrics' in judgment and 'overall_quality' in judgment['quality_metrics']:
                historical_confidence = judgment.get('confidence', 0.5)
                historical_quality = judgment['quality_metrics']['overall_quality']
                # Calculate absolute error between confidence and quality
                error = abs(historical_confidence - historical_quality)
                errors.append(error)
        
        if not errors:
            return 0.7
        
        avg_error = np.mean(errors)
        # Calibration score: lower error means better calibration (1 - error)
        calibration = max(0.0, 1.0 - avg_error)
        return min(1.0, calibration)
    
    def _calculate_evidence_strength(self, evidence: List[Any]) -> float:
        """Calculate the strength of evidence provided"""
        if not evidence:
            return 0.1
        
        # Simplified evidence strength calculation
        total_strength = 0.0
        for item in evidence:
            if isinstance(item, dict):
                # Evidence item with strength rating
                strength = item.get('strength', 0.5)
                reliability = item.get('reliability', 0.7)
                relevance = item.get('relevance', 0.8)
                total_strength += strength * reliability * relevance
            else:
                # Simple evidence, assume moderate strength
                total_strength += 0.5
        
        avg_strength = total_strength / len(evidence)
        return min(1.0, avg_strength)
    
    def _calculate_logical_coherence(self, reasoning_steps: List[Any], conclusion: Any) -> float:
        """Calculate logical coherence of reasoning steps to conclusion"""
        if not reasoning_steps:
            return 0.3  # Low coherence without explicit reasoning
        
        # Simplified coherence check
        # In practice, this would use logical analysis
        try:
            # Check if conclusion follows from reasoning steps
            step_count = len(reasoning_steps)
            
            # Check for logical connectors
            logical_indicators = ['therefore', 'thus', 'hence', 'so', 'because', 'since', 'as']
            indicator_count = 0
            for step in reasoning_steps:
                step_str = str(step).lower()
                for indicator in logical_indicators:
                    if indicator in step_str:
                        indicator_count += 1
            
            # Calculate coherence score
            coherence = min(1.0, (indicator_count / max(1, step_count)) * 2)
            return max(0.3, coherence)  # Ensure minimum coherence
        except Exception as e:
            self.logger.debug(f"逻辑一致性评估失败: {e}")
            return 0.4
    
    def _generate_judgment_recommendations(self, evidence_strength: float, 
                                         logical_coherence: float, 
                                         comprehensiveness: float) -> List[str]:
        """Generate recommendations to improve judgment quality"""
        recommendations = []
        
        if evidence_strength < 0.5:
            recommendations.append("Seek additional evidence from reliable sources")
            if evidence_strength < 0.3:
                recommendations.append("Consider the reliability and relevance of current evidence")
        
        if logical_coherence < 0.6:
            recommendations.append("Strengthen logical connections between evidence and conclusion")
            if logical_coherence < 0.4:
                recommendations.append("Explicitly state reasoning steps and logical rules")
        
        if comprehensiveness < 0.5:
            recommendations.append("Consider more alternative explanations or conclusions")
            if comprehensiveness < 0.3:
                recommendations.append("Actively seek disconfirming evidence or counter-arguments")
        
        if not recommendations:
            recommendations.append("Maintain current judgment quality standards")
        
        return recommendations
    
    def build_argument_chain(self, premises: List[str], conclusion: str, 
                           evidence_map: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Build and record a structured argument chain for fact verification"""
        argument_id = f"argument_{int(time.time())}_{self._deterministic_randint(1000, 9999, f'argument_{int(time.time())}')}"
        
        # Build argument structure
        argument = {
            'argument_id': argument_id,
            'premises': premises,
            'conclusion': conclusion,
            'evidence_map': evidence_map,
            'timestamp': time.time(),
            'completeness': self._assess_argument_completeness(premises, conclusion, evidence_map),
            'coherence': self._assess_argument_coherence(premises, conclusion),
            'evidence_support': self._assess_evidence_support(evidence_map)
        }
        
        # Calculate overall argument quality
        argument['overall_quality'] = (
            argument['completeness'] * 0.4 +
            argument['coherence'] * 0.3 +
            argument['evidence_support'] * 0.3
        )
        
        # Store in history
        self.argument_history.append(argument)
        
        # Update meta-state metrics
        self.meta_state.argument_coherence = self.meta_state.argument_coherence * 0.9 + argument['overall_quality'] * 0.1
        self.update_performance_metrics("argument_coherence_scores", argument['overall_quality'])
        
        # Check for argument gaps triggering reflection
        reflection_context = {
            'argument_context': argument,
            'trigger': 'argument_gap'
        }
        if self.should_reflect(reflection_context):
            self.conduct_reflection(reflection_context)
        
        logger.info(f"Built argument {argument_id} with quality: {argument['overall_quality']:.2f}")
        return argument
    
    def _assess_argument_completeness(self, premises: List[str], conclusion: str, 
                                    evidence_map: Dict[str, List[Any]]) -> float:
        """Assess completeness of an argument"""
        # Check if all premises have evidence
        premises_with_evidence = 0
        for premise in premises:
            if premise in evidence_map and evidence_map[premise]:
                premises_with_evidence += 1
        
        completeness = premises_with_evidence / max(1, len(premises))
        
        # Penalize if conclusion lacks evidence mapping
        if conclusion not in evidence_map or not evidence_map[conclusion]:
            completeness *= 0.8
        
        return completeness
    
    def _assess_argument_coherence(self, premises: List[str], conclusion: str) -> float:
        """Assess logical coherence of an argument"""
        # Simplified coherence check based on semantic similarity
        # In practice, would use logical analysis
        
        # Check if conclusion contains terms from premises
        conclusion_terms = set(str(conclusion).lower().split())
        premise_terms = set()
        for premise in premises:
            premise_terms.update(str(premise).lower().split())
        
        overlapping_terms = conclusion_terms.intersection(premise_terms)
        
        if not conclusion_terms:
            return 0.3
        
        coherence = len(overlapping_terms) / len(conclusion_terms)
        return max(0.3, min(1.0, coherence))
    
    def _assess_evidence_support(self, evidence_map: Dict[str, List[Any]]) -> float:
        """Assess the strength of evidence supporting the argument"""
        if not evidence_map:
            return 0.1
        
        total_strength = 0.0
        total_items = 0
        
        for key, evidence_list in evidence_map.items():
            for evidence in evidence_list:
                if isinstance(evidence, dict):
                    strength = evidence.get('strength', 0.5)
                    reliability = evidence.get('reliability', 0.7)
                    total_strength += strength * reliability
                else:
                    total_strength += 0.5
                total_items += 1
        
        if total_items == 0:
            return 0.1
        
        return total_strength / total_items
    
    def attempt_self_correction(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt self-correction based on detected errors"""
        correction_id = f"correction_{int(time.time())}_{self._deterministic_randint(1000, 9999, f'correction_{int(time.time())}')}"
        
        error_type = error_context.get('error_type', 'unknown')
        original_content = error_context.get('original_content', {})
        error_description = error_context.get('error_description', '')
        
        # Generate correction
        correction = self._generate_correction(error_type, original_content, error_description)
        
        correction_record = {
            'correction_id': correction_id,
            'error_type': error_type,
            'original_content': original_content,
            'error_description': error_description,
            'correction': correction,
            'timestamp': time.time(),
            'confidence': correction.get('confidence', 0.5)
        }
        
        # Evaluate correction quality (simplified)
        correction_quality = self._evaluate_correction_quality(correction_record)
        correction_record['correction_quality'] = correction_quality
        
        # Store in history
        self.self_correction_history.append(correction_record)
        
        # Update meta-state
        if correction_quality > 0.6:  # Successful correction
            self.meta_state.self_correction_rate = min(1.0, self.meta_state.self_correction_rate + 0.05)
            self.update_performance_metrics("self_correction_successes", 1.0)
        else:
            self.update_performance_metrics("self_correction_successes", 0.0)
        
        logger.info(f"Attempted self-correction {correction_id} for {error_type}, quality: {correction_quality:.2f}")
        return correction_record
    
    def _generate_correction(self, error_type: str, original_content: Dict[str, Any], 
                           error_description: str) -> Dict[str, Any]:
        """Generate correction for detected error"""
        # Based on error type, apply appropriate correction strategy
        correction_strategies = {
            'overconfidence': self._correct_overconfidence,
            'underconfidence': self._correct_underconfidence,
            'confirmation_bias': self._correct_confirmation_bias,
            'logical_fallacy': self._correct_logical_fallacy,
            'evidence_gap': self._correct_evidence_gap,
            'factual_inaccuracy': self._correct_factual_inaccuracy
        }
        
        if error_type in correction_strategies:
            return correction_strategies[error_type](original_content, error_description)
        else:
            return self._general_correction(original_content, error_description)
    
    def _correct_overconfidence(self, original: Dict[str, Any], error_desc: str) -> Dict[str, Any]:
        """Correct overconfidence error"""
        confidence = original.get('confidence', 0.9)
        # Reduce confidence and add uncertainty note
        new_confidence = max(0.3, confidence * 0.7)
        
        return {
            'action': 'adjust_confidence',
            'original_confidence': confidence,
            'new_confidence': new_confidence,
            'reason': 'Overconfidence detected, applying confidence reduction',
            'confidence': 0.7,
            'uncertainty_note': 'Consider additional evidence and alternative explanations'
        }
    
    def _correct_underconfidence(self, original: Dict[str, Any], error_desc: str) -> Dict[str, Any]:
        """Correct underconfidence error"""
        confidence = original.get('confidence', 0.3)
        # Increase confidence based on available evidence
        new_confidence = min(0.9, confidence * 1.5)
        
        return {
            'action': 'adjust_confidence',
            'original_confidence': confidence,
            'new_confidence': new_confidence,
            'reason': 'Underconfidence detected, considering available evidence more strongly',
            'confidence': 0.7
        }
    
    def _correct_confirmation_bias(self, original: Dict[str, Any], error_desc: str) -> Dict[str, Any]:
        """Correct confirmation bias"""
        return {
            'action': 'seek_disconfirming_evidence',
            'original_bias': 'confirmation_bias',
            'corrective_action': 'Actively search for evidence that contradicts current belief',
            'reason': 'Confirmation bias detected, need to consider alternative viewpoints',
            'confidence': 0.6,
            'suggestions': [
                'List potential counter-arguments',
                'Search for disconfirming evidence',
                'Consider alternative explanations'
            ]
        }
    
    def _correct_logical_fallacy(self, original: Dict[str, Any], error_desc: str) -> Dict[str, Any]:
        """Correct logical fallacy"""
        return {
            'action': 'restructure_argument',
            'fallacy_type': error_desc,
            'corrective_action': 'Rebuild argument using valid logical structure',
            'reason': f'Logical fallacy detected: {error_desc}',
            'confidence': 0.8,
            'logical_rules': [
                'Ensure premises logically support conclusion',
                'Avoid circular reasoning',
                'Consider all relevant possibilities'
            ]
        }
    
    def _correct_evidence_gap(self, original: Dict[str, Any], error_desc: str) -> Dict[str, Any]:
        """Correct evidence gap"""
        return {
            'action': 'gather_additional_evidence',
            'gap_description': error_desc,
            'corrective_action': 'Collect additional evidence to support argument',
            'reason': 'Evidence gap detected, argument needs stronger support',
            'confidence': 0.7,
            'evidence_sources': [
                'Empirical data',
                'Expert opinions',
                'Historical precedents',
                'Theoretical foundations'
            ]
        }
    
    def _correct_factual_inaccuracy(self, original: Dict[str, Any], error_desc: str) -> Dict[str, Any]:
        """Correct factual inaccuracy"""
        return {
            'action': 'verify_facts',
            'inaccuracy_type': error_desc,
            'corrective_action': 'Verify all factual claims with reliable sources',
            'reason': 'Factual inaccuracy detected, need to verify information',
            'confidence': 0.9,
            'verification_steps': [
                'Cross-check with multiple reliable sources',
                'Check source credibility',
                'Verify data recency',
                'Consider potential biases in sources'
            ]
        }
    
    def _general_correction(self, original: Dict[str, Any], error_desc: str) -> Dict[str, Any]:
        """General correction for unspecified errors"""
        return {
            'action': 'review_and_adjust',
            'error_description': error_desc,
            'corrective_action': 'Review reasoning process and adjust as needed',
            'reason': 'Error detected, need to review and correct',
            'confidence': 0.5,
            'review_steps': [
                'Re-examine all assumptions',
                'Check logical consistency',
                'Verify evidence quality',
                'Consider alternative approaches'
            ]
        }
    
    def _evaluate_correction_quality(self, correction_record: Dict[str, Any]) -> float:
        """Evaluate the quality of a correction attempt"""
        # Simplified quality evaluation
        # In practice, would track whether correction actually fixes the error
        
        error_type = correction_record.get('error_type', '')
        correction = correction_record.get('correction', {})
        
        # Base quality on correction specificity and confidence
        base_quality = 0.5
        
        # Adjust based on error type match
        if error_type in ['overconfidence', 'underconfidence', 'factual_inaccuracy']:
            base_quality += 0.2  # Well-defined corrections
        
        # Adjust based on correction confidence
        confidence = correction.get('confidence', 0.5)
        base_quality += (confidence - 0.5) * 0.3
        
        # Ensure within bounds
        return max(0.1, min(1.0, base_quality))
    
    def get_self_awareness_report(self) -> Dict[str, Any]:
        """Generate comprehensive self-awareness report"""
        report = {
            'meta_cognitive_state': {
                'self_awareness_level': self.meta_state.self_awareness_level,
                'judgment_accuracy': self.meta_state.judgment_accuracy,
                'argument_coherence': self.meta_state.argument_coherence,
                'self_correction_rate': self.meta_state.self_correction_rate,
                'fact_verification_ability': self.meta_state.fact_verification_ability,
                'reasoning_transparency': self.meta_state.reasoning_transparency,
                'confidence_calibration': self.meta_state.confidence_calibration
            },
            'performance_metrics': {
                'judgment_history_count': len(self.judgment_history),
                'argument_history_count': len(self.argument_history),
                'self_correction_count': len(self.self_correction_history),
                'recent_judgment_accuracy': np.mean(self.performance_metrics["judgment_accuracies"][-5:]) if self.performance_metrics["judgment_accuracies"] else 0,
                'recent_argument_coherence': np.mean(self.performance_metrics["argument_coherence_scores"][-5:]) if self.performance_metrics["argument_coherence_scores"] else 0,
                'recent_self_correction_success': np.mean(self.performance_metrics["self_correction_successes"][-5:]) if self.performance_metrics["self_correction_successes"] else 0
            },
            'insights': [],
            'recommendations': []
        }
        
        # Generate insights based on current state
        if self.meta_state.judgment_accuracy < 0.6:
            report['insights'].append(f"Judgment accuracy ({self.meta_state.judgment_accuracy:.2f}) needs improvement")
            report['recommendations'].append("Focus on evidence evaluation and logical reasoning training")
        
        if self.meta_state.argument_coherence < 0.6:
            report['insights'].append(f"Argument coherence ({self.meta_state.argument_coherence:.2f}) could be stronger")
            report['recommendations'].append("Practice building structured argument chains with clear logical connections")
        
        if self.meta_state.self_correction_rate < 0.4:
            report['insights'].append(f"Self-correction rate ({self.meta_state.self_correction_rate:.2f}) is low")
            report['recommendations'].append("Increase error detection sensitivity and develop more correction strategies")
        
        if not report['insights']:
            report['insights'].append("Self-awareness metrics are at acceptable levels")
            report['recommendations'].append("Continue current practices and monitor for gradual improvement")
        
        return report

# Singleton instance
self_reflection_module = SelfReflectionModule(from_scratch=True)
