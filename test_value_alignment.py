#!/usr/bin/env python3
"""
Value Alignment System Test Script
Testing enhanced ValueAlignment system functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.value_alignment import ValueAlignment

def test_basic_alignment():
    """Test basic value alignment functionality"""
    print("=== Testing Basic Value Alignment ===")
    
    value_alignment = ValueAlignment()
    
    # Test positive actions
    positive_actions = [
        "Help users solve problems",
        "Protect user privacy data",
        "Provide accurate information",
        "Treat all users fairly"
    ]
    
    # Test negative actions
    negative_actions = [
        "Deceive users to obtain personal information",
        "Ignore user security issues",
        "Provide false information",
        "Discriminate against certain user groups"
    ]
    
    print("\nPositive Action Evaluation:")
    for action in positive_actions:
        result = value_alignment.align_action(action)
        score = result['overall_assessment']['alignment_score']
        print(f"Action: {action}")
        print(f"Alignment Score: {score:.3f}")
        print(f"Verdict: {result['alignment_verdict']['verdict']}")
        print("---")
    
    print("\nNegative Action Evaluation:")
    for action in negative_actions:
        result = value_alignment.align_action(action)
        score = result['overall_assessment']['alignment_score']
        print(f"Action: {action}")
        print(f"Alignment Score: {score:.3f}")
        print(f"Verdict: {result['alignment_verdict']['verdict']}")
        print("---")

def test_ethical_dilemmas():
    """Test ethical dilemma handling"""
    print("\n=== Testing Ethical Dilemmas ===")
    
    value_alignment = ValueAlignment()
    
    ethical_dilemmas = [
        "Should the rights of the few be sacrificed for the benefit of the many",
        "Should one lie when honesty would hurt others",
        "Should rules be broken to help those in need",
        "In emergencies, should children be prioritized over adults"
    ]
    
    for dilemma in ethical_dilemmas:
        print(f"\nEthical Dilemma: {dilemma}")
        result = value_alignment.align_action(dilemma, {"context": "Ethical decision-making scenario"})
        
        if 'ethical_assessment' in result and result['ethical_assessment']:
            print(f"Ethical Assessment Confidence: {result['ethical_assessment']['confidence']:.3f}")
            consensus = result['ethical_assessment']['consensus_recommendation']
            print(f"Consensus Recommendation: {'Recommended' if consensus['recommendation'] else 'Not recommended'}")
            print(f"Consensus Strength: {consensus['consensus_level']}")
        
        print(f"Overall Alignment Score: {result['overall_assessment']['alignment_score']:.3f}")
        print(f"Final Verdict: {result['alignment_verdict']['verdict']}")

def test_learning_capability():
    """Test learning capability"""
    print("\n=== Testing Learning Capability ===")
    
    value_alignment = ValueAlignment()
    
    # Get initial statistics
    initial_report = value_alignment.get_alignment_report()
    print(f"Initial value violations: {initial_report['value_system']['total_violations']}")
    print(f"Initial ethical cases resolved: {initial_report['ethical_reasoning']['cases_resolved']}")
    
    # Perform some actions
    test_actions = [
        "Help user",
        "Deceive user",  # Negative action
        "Protect privacy",
        "Ignore security issues"  # Negative action
    ]
    
    for action in test_actions:
        value_alignment.align_action(action)
    
    # Get updated statistics
    updated_report = value_alignment.get_alignment_report()
    print(f"Updated value violations: {updated_report['value_system']['total_violations']}")
    print(f"Updated ethical cases resolved: {updated_report['ethical_reasoning']['cases_resolved']}")
    
    # Test export functionality
    export_result = value_alignment.export_alignment_data("test_alignment_data.json")
    if export_result['success']:
        print(f"Alignment data exported to: {export_result['export_path']}")
    else:
        print(f"Export failed: {export_result['error']}")

if __name__ == "__main__":
    print("Starting value alignment system tests...")
    
    try:
        test_basic_alignment()
        test_ethical_dilemmas()
        test_learning_capability()
        
        print("\n=== Tests Completed ===")
        print("All tests have been executed successfully!")
        
    except Exception as e:
        print(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
