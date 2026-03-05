#!/usr/bin/env python3
"""
CMCP Demo - Cross-Model Communication Protocol Demonstration
==========================================================

This demo shows how to use the Cross-Model Communication Protocol (CMCP)
for seamless communication between different AI models.

Features demonstrated:
1. Model registration and discovery
2. Direct call communication
3. Message routing through gateway
4. Error handling and monitoring
5. Performance metrics collection
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cross_model_communication import (
    CMCPGateway, GatewayConfig, CommunicationProtocol,
    ModelCommunicationMixin, communication_handler,
    create_adapter_for_model, get_gateway, get_monitor
)
from core.cross_model_communication.message_format import (
    RequestMessage, ResponseMessage, MessagePriority
)


class DemoLanguageModel:
    """Demo language model with CMCP integration"""
    
    def __init__(self, model_id: str = "demo_language_model"):
        self.model_id = model_id
        self.model_type = "language"
        self.version = "1.0.0"
        
        # Initialize communication
        self.init_communication({
            'model_id': model_id,
            'model_type': 'language',
            'max_concurrent_requests': 5
        })
    
    @communication_handler(operation="generate_text")
    async def handle_generate_text(self, message: RequestMessage, context: Dict[str, Any]) -> ResponseMessage:
        """Handle text generation requests"""
        prompt = message.data.get("prompt", "")
        max_length = message.data.get("max_length", 100)
        
        # Simulate text generation
        generated_text = f"[DEMO] Generated text for prompt: '{prompt}' (max_length: {max_length})"
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        return ResponseMessage(
            response_to=message.message_id,
            source_model=self.model_id,
            original_source=message.source_model,
            operation=message.operation,
            data={
                "generated_text": generated_text,
                "prompt": prompt,
                "tokens_generated": len(generated_text.split()),
                "processing_time_ms": 100
            },
            status="success"
        )
    
    @communication_handler(operation="translate")
    async def handle_translate(self, message: RequestMessage, context: Dict[str, Any]) -> ResponseMessage:
        """Handle translation requests"""
        text = message.data.get("text", "")
        source_lang = message.data.get("source_lang", "en")
        target_lang = message.data.get("target_lang", "es")
        
        # Simulate translation
        translated_text = f"[DEMO TRANSLATED {source_lang}→{target_lang}]: {text}"
        
        await asyncio.sleep(0.05)
        
        return ResponseMessage(
            response_to=message.message_id,
            source_model=self.model_id,
            original_source=message.source_model,
            operation=message.operation,
            data={
                "translated_text": translated_text,
                "original_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "confidence": 0.95
            },
            status="success"
        )
    
    def get_capabilities(self) -> list:
        """Get model capabilities"""
        return ["generate_text", "translate", "summarize", "analyze_sentiment"]


class DemoVisionModel:
    """Demo vision model with CMCP integration"""
    
    def __init__(self, model_id: str = "demo_vision_model"):
        self.model_id = model_id
        self.model_type = "vision"
        self.version = "1.0.0"
        
        # Initialize communication
        self.init_communication({
            'model_id': model_id,
            'model_type': 'vision',
            'estimated_processing_time_ms': 200
        })
    
    @communication_handler(operation="analyze_image")
    async def handle_analyze_image(self, message: RequestMessage, context: Dict[str, Any]) -> ResponseMessage:
        """Handle image analysis requests"""
        image_data = message.data.get("image_data", "")
        analysis_type = message.data.get("analysis_type", "objects")
        
        # Simulate image analysis
        objects = ["person", "car", "tree", "building"]
        confidence_scores = [0.95, 0.87, 0.92, 0.78]
        
        await asyncio.sleep(0.2)
        
        return ResponseMessage(
            response_to=message.message_id,
            source_model=self.model_id,
            original_source=message.source_model,
            operation=message.operation,
            data={
                "analysis_type": analysis_type,
                "objects_detected": objects,
                "confidence_scores": confidence_scores,
                "image_size": len(image_data),
                "processing_time_ms": 200
            },
            status="success"
        )
    
    @communication_handler(operation="generate_caption")
    async def handle_generate_caption(self, message: RequestMessage, context: Dict[str, Any]) -> ResponseMessage:
        """Handle image caption generation"""
        image_data = message.data.get("image_data", "")
        
        # Simulate caption generation
        caption = "A beautiful sunset over mountains with trees in the foreground"
        
        await asyncio.sleep(0.15)
        
        return ResponseMessage(
            response_to=message.message_id,
            source_model=self.model_id,
            original_source=message.source_model,
            operation=message.operation,
            data={
                "caption": caption,
                "confidence": 0.88,
                "image_size": len(image_data),
                "processing_time_ms": 150
            },
            status="success"
        )
    
    def get_capabilities(self) -> list:
        """Get model capabilities"""
        return ["analyze_image", "generate_caption", "detect_faces", "segment_image"]


class DemoKnowledgeModel:
    """Demo knowledge model with CMCP integration"""
    
    def __init__(self, model_id: str = "demo_knowledge_model"):
        self.model_id = model_id
        self.model_type = "knowledge"
        self.version = "1.0.0"
        
        # Initialize communication
        self.init_communication({
            'model_id': model_id,
            'model_type': 'knowledge',
            'memory_requirement_mb': 1024
        })
    
    @communication_handler(operation="retrieve_facts")
    async def handle_retrieve_facts(self, message: RequestMessage, context: Dict[str, Any]) -> ResponseMessage:
        """Handle fact retrieval requests"""
        topic = message.data.get("topic", "")
        max_facts = message.data.get("max_facts", 5)
        
        # Simulate knowledge retrieval
        facts = [
            f"Fact 1 about {topic}: This is an important fact.",
            f"Fact 2 about {topic}: This is another important fact.",
            f"Fact 3 about {topic}: This is a third important fact."
        ][:max_facts]
        
        await asyncio.sleep(0.08)
        
        return ResponseMessage(
            response_to=message.message_id,
            source_model=self.model_id,
            original_source=message.source_model,
            operation=message.operation,
            data={
                "topic": topic,
                "facts": facts,
                "fact_count": len(facts),
                "source": "demo_knowledge_base",
                "processing_time_ms": 80
            },
            status="success"
        )
    
    @communication_handler(operation="answer_question")
    async def handle_answer_question(self, message: RequestMessage, context: Dict[str, Any]) -> ResponseMessage:
        """Handle question answering"""
        question = message.data.get("question", "")
        context_text = message.data.get("context", "")
        
        # Simulate question answering
        answer = f"The answer to '{question}' based on the provided context is: This is a simulated answer."
        
        await asyncio.sleep(0.12)
        
        return ResponseMessage(
            response_to=message.message_id,
            source_model=self.model_id,
            original_source=message.source_model,
            operation=message.operation,
            data={
                "question": question,
                "answer": answer,
                "confidence": 0.85,
                "context_used": bool(context_text),
                "processing_time_ms": 120
            },
            status="success"
        )
    
    def get_capabilities(self) -> list:
        """Get model capabilities"""
        return ["retrieve_facts", "answer_question", "summarize_document", "find_similar"]


async def demo_single_model_communication():
    """Demo communication between two models"""
    print("\n" + "="*60)
    print("DEMO 1: Single Model Communication")
    print("="*60)
    
    # Create models
    language_model = DemoLanguageModel()
    vision_model = DemoVisionModel()
    
    # Start gateway
    config = GatewayConfig(
        default_protocol=CommunicationProtocol.DIRECT_CALL,
        protocol_configs={
            "direct_call": {"max_workers": 10}
        }
    )
    
    gateway = get_gateway(config)
    await gateway.start()
    
    try:
        # Demo 1: Language model asks vision model to analyze an image
        print("\n1. Language → Vision: Analyze image")
        response = await language_model.send_to_model(
            target_model="demo_vision_model",
            operation="analyze_image",
            data={
                "image_data": "fake_image_data_here",
                "analysis_type": "objects"
            }
        )
        
        if response and response.is_success():
            print(f"   ✓ Success! Objects detected: {response.data.get('objects_detected', [])}")
            print(f"   Confidence scores: {response.data.get('confidence_scores', [])}")
        else:
            print(f"   ✗ Failed: {response.get_error_message() if response else 'No response'}")
        
        # Demo 2: Vision model asks language model for caption
        print("\n2. Vision → Language: Generate caption for analysis")
        response = await vision_model.send_to_model(
            target_model="demo_language_model",
            operation="generate_text",
            data={
                "prompt": "Describe the scene with detected objects",
                "max_length": 50
            }
        )
        
        if response and response.is_success():
            print(f"   ✓ Success! Generated text: {response.data.get('generated_text', '')}")
        else:
            print(f"   ✗ Failed: {response.get_error_message() if response else 'No response'}")
        
        # Demo 3: Chain of communication
        print("\n3. Chain: Vision → Language → Knowledge")
        # First, vision analyzes image
        vision_response = await vision_model.send_to_model(
            target_model="demo_vision_model",
            operation="analyze_image",
            data={
                "image_data": "another_fake_image",
                "analysis_type": "objects"
            }
        )
        
        if vision_response and vision_response.is_success():
            objects = vision_response.data.get("objects_detected", [])
            print(f"   ✓ Vision analysis complete. Objects: {objects}")
            
            # Then, language model uses vision results
            if objects:
                language_response = await language_model.send_to_model(
                    target_model="demo_language_model",
                    operation="generate_text",
                    data={
                        "prompt": f"Write a short story including: {', '.join(objects)}",
                        "max_length": 100
                    }
                )
                
                if language_response and language_response.is_success():
                    print(f"   ✓ Language generation complete.")
                    story = language_response.data.get("generated_text", "")
                    print(f"   Story: {story[:80]}..." if len(story) > 80 else f"   Story: {story}")
        
        # Demo 4: Error handling
        print("\n4. Error Handling: Request to non-existent model")
        response = await language_model.send_to_model(
            target_model="non_existent_model",
            operation="do_something",
            data={"test": "data"}
        )
        
        if response and response.is_error():
            print(f"   ✓ Error handled correctly: {response.get_error_message()}")
        else:
            print(f"   ✗ Expected error but got: {response}")
    
    finally:
        # Get metrics before stopping
        metrics = gateway.get_metrics()
        print("\n" + "-"*60)
        print("METRICS SUMMARY:")
        print(f"  Total requests: {metrics['gateway_metrics']['requests_total']}")
        print(f"  Successful: {metrics['gateway_metrics']['requests_successful']}")
        print(f"  Failed: {metrics['gateway_metrics']['requests_failed']}")
        print(f"  Average latency: {metrics['gateway_metrics']['average_latency_ms']:.2f}ms")
        print("-"*60)
        
        # Stop gateway
        await gateway.stop()


async def demo_multi_model_collaboration():
    """Demo multi-model collaboration"""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Model Collaboration")
    print("="*60)
    
    # Create all models
    language_model = DemoLanguageModel("collab_language_model")
    vision_model = DemoVisionModel("collab_vision_model")
    knowledge_model = DemoKnowledgeModel("collab_knowledge_model")
    
    # Start gateway
    config = GatewayConfig(
        default_protocol=CommunicationProtocol.DIRECT_CALL,
        protocol_configs={
            "direct_call": {"max_workers": 15}
        }
    )
    
    gateway = get_gateway(config)
    await gateway.start()
    
    try:
        # Demo: Complex multi-model task
        print("\nMulti-Model Task: Describe and analyze a scene")
        
        # Step 1: Vision model analyzes image
        print("\nStep 1: Vision analysis")
        vision_response = await vision_model.send_to_model(
            target_model="collab_vision_model",
            operation="analyze_image",
            data={
                "image_data": "scene_image_data",
                "analysis_type": "objects"
            }
        )
        
        if not vision_response or not vision_response.is_success():
            print("   ✗ Vision analysis failed")
            return
        
        objects = vision_response.data.get("objects_detected", [])
        print(f"   ✓ Detected objects: {objects}")
        
        # Step 2: Knowledge model retrieves facts about objects
        print("\nStep 2: Knowledge retrieval")
        knowledge_tasks = []
        for obj in objects[:3]:  # Limit to 3 objects for demo
            task = knowledge_model.send_to_model(
                target_model="collab_knowledge_model",
                operation="retrieve_facts",
                data={
                    "topic": obj,
                    "max_facts": 2
                }
            )
            knowledge_tasks.append(task)
        
        knowledge_responses = await asyncio.gather(*knowledge_tasks)
        
        all_facts = []
        for i, response in enumerate(knowledge_responses):
            if response and response.is_success():
                facts = response.data.get("facts", [])
                all_facts.extend(facts)
                print(f"   ✓ Facts for {objects[i]}: {len(facts)} facts")
        
        # Step 3: Language model generates comprehensive description
        print("\nStep 3: Generate comprehensive description")
        
        prompt = f"""
        Based on the following scene analysis:
        - Objects detected: {', '.join(objects)}
        - Relevant facts: {' '.join(all_facts[:3])}
        
        Write a detailed, engaging description of this scene.
        """
        
        language_response = await language_model.send_to_model(
            target_model="collab_language_model",
            operation="generate_text",
            data={
                "prompt": prompt,
                "max_length": 200
            }
        )
        
        if language_response and language_response.is_success():
            description = language_response.data.get("generated_text", "")
            print(f"   ✓ Generated description:")
            print(f"   \"{description}\"")
        
        # Step 4: All models collaborate on final analysis
        print("\nStep 4: Collaborative final analysis")
        
        # Create a complex prompt using all model outputs
        final_prompt = f"""
        Scene Analysis Summary:
        1. Vision Analysis: Detected {len(objects)} objects including {', '.join(objects[:3])}
        2. Knowledge Base: Retrieved {len(all_facts)} relevant facts
        3. Initial Description: {description[:100]}...
        
        Provide a comprehensive analysis including:
        - What this scene likely represents
        - Interesting facts about the objects
        - Cultural or historical context if applicable
        """
        
        final_response = await language_model.send_to_model(
            target_model="collab_language_model",
            operation="generate_text",
            data={
                "prompt": final_prompt,
                "max_length": 300
            }
        )
        
        if final_response and final_response.is_success():
            final_analysis = final_response.data.get("generated_text", "")
            print(f"   ✓ Final collaborative analysis:")
            print(f"   {final_analysis}")
        
        print("\n" + "="*60)
        print("COLLABORATION COMPLETE!")
        print(f"  Models involved: 3")
        print(f"  Operations performed: 5+")
        print(f"  Data exchanged: Objects, facts, descriptions")
        print("="*60)
    
    finally:
        # Get detailed metrics
        monitor = get_monitor()
        metrics_summary = monitor.get_metrics_summary()
        
        print("\n" + "-"*60)
        print("DETAILED METRICS:")
        for metric_name, data in metrics_summary.get("metrics", {}).items():
            if "cmcp" in metric_name:
                print(f"  {metric_name}:")
                print(f"    Latest: {data.get('latest', 0):.2f}")
                print(f"    Average: {data.get('average', 0):.2f}")
                print(f"    Count: {data.get('count', 0)}")
        
        print(f"  Active traces: {metrics_summary['traces']['active']}")
        print(f"  Completed traces: {metrics_summary['traces']['completed']}")
        print("-"*60)
        
        # Stop gateway
        await gateway.stop()


async def demo_monitoring_and_observability():
    """Demo monitoring and observability features"""
    print("\n" + "="*60)
    print("DEMO 3: Monitoring & Observability")
    print("="*60)
    
    # Create a simple model
    class MonitoredModel(DemoLanguageModel):
        def __init__(self):
            super().__init__("monitored_model")
            self.request_count = 0
        
        @communication_handler(operation="monitored_operation")
        async def handle_monitored_operation(self, message: RequestMessage, context: Dict[str, Any]) -> ResponseMessage:
            self.request_count += 1
            
            # Simulate occasional errors
            if self.request_count % 5 == 0:
                raise ValueError("Simulated error every 5th request")
            
            await asyncio.sleep(0.05)
            
            return ResponseMessage(
                response_to=message.message_id,
                source_model=self.model_id,
                original_source=message.source_model,
                operation=message.operation,
                data={
                    "request_number": self.request_count,
                    "status": "processed"
                },
                status="success"
            )
    
    model = MonitoredModel()
    
    # Start gateway with monitoring enabled
    config = GatewayConfig(
        default_protocol=CommunicationProtocol.DIRECT_CALL,
        monitoring_enabled=True,
        circuit_breaker_enabled=True
    )
    
    gateway = get_gateway(config)
    await gateway.start()
    
    try:
        print("\nSending monitored requests (some will fail):")
        
        tasks = []
        for i in range(1, 11):
            task = model.send_to_model(
                target_model="monitored_model",
                operation="monitored_operation",
                data={"request_id": i},
                priority="medium" if i % 2 == 0 else "high"
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        error_count = 0
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_count += 1
                print(f"  Request {i+1}: ✗ Error: {response}")
            elif response and response.is_success():
                success_count += 1
                print(f"  Request {i+1}: ✓ Success (req #{response.data.get('request_number', '?')})")
            else:
                error_count += 1
                print(f"  Request {i+1}: ✗ Failed: {response.get_error_message() if response else 'Unknown'}")
        
        print(f"\nResults: {success_count} successful, {error_count} failed")
        
        # Demo monitoring features
        monitor = get_monitor()
        
        print("\nMonitoring Features:")
        print("1. Metrics Collection:")
        metrics_summary = monitor.get_metrics_summary()
        print(f"   - Total metrics tracked: {len(metrics_summary.get('metrics', {}))}")
        
        # Export metrics
        print("\n2. Metrics Export (Prometheus format):")
        prometheus_metrics = monitor.export_metrics("prometheus")
        # Show first few lines
        lines = prometheus_metrics.split('\n')[:5]
        for line in lines:
            print(f"   {line}")
        if len(prometheus_metrics.split('\n')) > 5:
            print(f"   ... and {len(prometheus_metrics.split('\n')) - 5} more lines")
        
        # Alert demonstration
        print("\n3. Alert System:")
        
        # Add an alert rule
        monitor.add_alert_rule({
            "name": "high_error_rate",
            "metric": "cmcp_errors_total",
            "condition": ">",
            "threshold": 2,
            "handler": lambda alert: print(f"   ⚠️ ALERT: {alert['rule']['name']} triggered!")
        })
        
        # Check current alerts
        if monitor.active_alerts:
            print(f"   Active alerts: {len(monitor.active_alerts)}")
            for alert_id, alert in monitor.active_alerts.items():
                print(f"   - {alert_id}: {alert['metric_value']} (threshold: {alert['rule'].get('threshold')})")
        else:
            print("   No active alerts")
        
        print("\n" + "-"*60)
        print("MONITORING DEMO COMPLETE")
        print("-"*60)
    
    finally:
        await gateway.stop()


async def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("CROSS-MODEL COMMUNICATION PROTOCOL (CMCP) DEMO")
    print("="*60)
    print("Demonstrating seamless AI model collaboration\n")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run demos
        await demo_single_model_communication()
        await demo_multi_model_collaboration()
        await demo_monitoring_and_observability()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("1. Model registration and discovery")
        print("2. Direct model-to-model communication")
        print("3. Multi-model collaboration workflows")
        print("4. Error handling and circuit breakers")
        print("5. Comprehensive monitoring and metrics")
        print("6. Alert system for operational awareness")
        print("\nCMCP enables AGI systems to collaborate like never before!")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)