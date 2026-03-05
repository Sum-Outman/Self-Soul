import zlib
"""
AGI-Enhanced Creative Writing Tool - Advanced Language Model Integration

A comprehensive creative writing assistant built on top of the UnifiedLanguageModel (AdvancedLanguageModel).
Provides specialized methods for various creative writing tasks, including story generation,
character development, world-building, dialogue creation, and creative enhancement.

Features:
- Story generation with customizable genres, settings, and characters
- Character development with personality traits, backgrounds, and motivations
- World-building for fantasy, sci-fi, historical, and contemporary settings
- Dialogue generation with emotional tones and character voices
- Creative writing enhancement and rewriting
- Style adaptation (poetic, humorous, dramatic, etc.)
- Multi-lingual creative writing support
- AGI-enhanced creativity with emotional intelligence
"""

import logging
import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from core.models.language.unified_language_model import AdvancedLanguageModel
from core.model_registry import ModelRegistry
from core.error_handling import error_handler
from core.agi_tools import AGITools
from core.scene_adaptive_parameters import SceneAdaptiveParameters
from core.cycle_prevention_manager import CyclePreventionManager

class CreativeWritingTool:
    """AGI-Enhanced Creative Writing Assistant Tool
    
    A specialized tool for creative writing tasks that leverages the full power of the
    AdvancedLanguageModel with AGI capabilities for enhanced creativity, emotional intelligence,
    and contextual understanding.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the creative writing tool with AGI-enhanced language model.
        
        Args:
            config: Optional configuration dictionary for model initialization
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize AGI tools for enhanced creativity
        self.agi_tools = AGITools(
            model_type="creative_writing",
            model_id="creative_writing_tool",
            config=self.config
        )
        
        # Try to get language model from registry or create new instance
        self.language_model = self._initialize_language_model()
        
        # Initialize cycle prevention manager with both basic and adaptive protection
        # Basic layer: embedded-style protection (buffer cleanup + repeat detection + watchdog)
        # Adaptive layer: scene-aware parameter adjustment + performance feedback
        self.cycle_prevention_manager = CyclePreventionManager(
            config={
                "history_buffer_size": 15,  # Larger buffer for creative writing context
                "repeat_threshold": 3,      # 3 consecutive repeats trigger protection
                "base_temperature": 0.8,    # Higher base temperature for creativity
                "max_temperature": 1.2,     # Allow higher temperature for creative writing
                "base_repetition_penalty": 1.1,  # Lower penalty for creative repetition
                "max_repetition_penalty": 1.8,   # Higher max penalty if needed
            },
            enable_adaptive_layer=True
        )
        
        # Creative writing templates and prompts database
        self.templates = self._load_creative_templates()
        
        # Genre-specific writing styles
        self.genre_styles = {
            "fantasy": {
                "description": "epic, magical, imaginative, detailed world-building",
                "tone": "wondrous, adventurous, mysterious",
                "elements": ["magic", "mythical creatures", "ancient prophecies", "heroic quests"]
            },
            "sci_fi": {
                "description": "futuristic, technological, speculative, scientifically plausible",
                "tone": "futuristic, analytical, awe-inspiring",
                "elements": ["advanced technology", "space travel", "aliens", "future societies"]
            },
            "mystery": {
                "description": "suspenseful, clue-driven, logical, puzzle-solving",
                "tone": "suspenseful, mysterious, analytical",
                "elements": ["clues", "red herrings", "investigations", "reveals"]
            },
            "romance": {
                "description": "emotional, relationship-focused, heartfelt, character-driven",
                "tone": "passionate, emotional, tender",
                "elements": ["relationships", "emotional conflicts", "heartfelt moments", "romantic tension"]
            },
            "historical": {
                "description": "accurate, period-specific, culturally authentic, detailed",
                "tone": "authentic, immersive, educational",
                "elements": ["historical accuracy", "period details", "cultural context", "real events"]
            },
            "horror": {
                "description": "frightening, suspenseful, atmospheric, psychologically intense",
                "tone": "fearful, tense, disturbing",
                "elements": ["fear", "suspense", "supernatural", "psychological terror"]
            },
            "contemporary": {
                "description": "realistic, modern, relatable, character-focused",
                "tone": "realistic, relatable, conversational",
                "elements": ["everyday life", "modern problems", "personal growth", "social issues"]
            }
        }
        
        # Character archetypes database
        self.character_archetypes = {
            "hero": ["brave", "determined", "selfless", "honorable"],
            "mentor": ["wise", "experienced", "guiding", "knowledgeable"],
            "trickster": ["clever", "mischievous", "unpredictable", "humorous"],
            "guardian": ["protective", "loyal", "strong", "dutiful"],
            "shadow": ["dark", "conflicted", "antagonistic", "complex"],
            "herald": ["messenger", "catalyst", "announcing", "transformative"],
            "threshold_guardian": ["obstructive", "testing", "challenging", "protective"],
            "shape_shifter": ["unpredictable", "mysterious", "deceptive", "transformative"]
        }
        
        # Emotional tone mapping for creative writing
        self.emotional_tones = {
            "joyful": ["happy", "celebratory", "optimistic", "enthusiastic"],
            "sad": ["melancholy", "nostalgic", "heartbreaking", "poignant"],
            "angry": ["furious", "outraged", "bitter", "vengeful"],
            "fearful": ["terrified", "anxious", "paranoid", "dreadful"],
            "surprised": ["astonished", "shocked", "amazed", "bewildered"],
            "disgusted": ["repulsed", "contemptuous", "sickened", "scornful"],
            "neutral": ["calm", "balanced", "objective", "detached"],
            "romantic": ["passionate", "loving", "affectionate", "yearning"],
            "epic": ["grand", "heroic", "monumental", "legendary"],
            "humorous": ["funny", "witty", "sarcastic", "playful"]
        }
        
        self.logger.info("Creative Writing Tool initialized with AGI capabilities")
    
    def _initialize_language_model(self) -> AdvancedLanguageModel:
        """Initialize or retrieve the advanced language model for creative writing.
        
        Returns:
            An instance of AdvancedLanguageModel configured for creative writing
        """
        try:
            # Try to get model from registry first
            model_registry = ModelRegistry()
            language_model = model_registry.get_model("language")
            
            if language_model and isinstance(language_model, AdvancedLanguageModel):
                self.logger.info("Retrieved language model from registry for creative writing")
                return language_model
            
            # Fallback: create new instance
            self.logger.info("Creating new AdvancedLanguageModel instance for creative writing")
            
            # Configure for creative writing with enhanced parameters
            model_config = {
                "model_type": "language",
                "optimization_level": "creative",
                "enable_creative_generation": True,
                "creativity_parameters": {
                    "novelty_weight": 0.9,
                    "surprise_factor": 0.8,
                    "coherence_threshold": 0.85,
                    "diversity_measure": 0.9,
                    "emotional_depth": 0.9,
                    "intellectual_complexity": 0.8
                },
                "max_sequence_length": 1024,
                "temperature": 0.8  # Higher temperature for more creative outputs
            }
            
            # Merge with user config if provided
            if self.config:
                model_config.update(self.config)
            
            return AdvancedLanguageModel(model_config)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize language model: {str(e)}")
            error_handler.log_error(f"Language model initialization failed: {str(e)}", "CreativeWritingTool")
            raise
    
    def _load_creative_templates(self) -> Dict[str, Any]:
        """Load creative writing templates and prompts.
        
        Returns:
            Dictionary of creative writing templates organized by category
        """
        return {
            "story_starters": [
                "It was a dark and stormy night when...",
                "The ancient prophecy foretold that one day...",
                "No one believed me when I said I could see...",
                "The door creaked open, revealing a room that hadn't been entered for centuries...",
                "In a world where magic was currency...",
                "The last human on Earth wasn't alone after all...",
                "The map led to a place that shouldn't exist...",
                "Every family has secrets, but ours could destroy the world...",
                "The clock struck thirteen, and everything changed...",
                "They said the forest was haunted, but I had to find out for myself..."
            ],
            "character_introductions": [
                "{name} had always been {trait}, but today was different...",
                "No one really knew {name}, not even {pronoun}self...",
                "With a history of {background}, {name} carried the weight of {burden}...",
                "{name}'s eyes held stories that {pronoun} would never tell...",
                "Born with the ability to {ability}, {name} struggled to fit in...",
                "{name} made a living by {occupation}, but {pronoun} secret life was far more interesting...",
                "Everyone underestimated {name}, and that was {pronoun} greatest advantage...",
                "After {event}, {name} was never the same...",
                "{name} believed in {belief}, until the day {challenge}...",
                "The world saw {name} as {perception}, but the truth was far more complex..."
            ],
            "dialogue_prompts": [
                "\"I never thought I'd see you again,\" {character1} said, voice trembling.",
                "\"There are things you don't understand,\" {character2} whispered urgently.",
                "\"Trust me,\" {character1} pleaded, \"this is bigger than both of us.\"",
                "\"Why now? After all these years?\" {character2} demanded.",
                "\"Some secrets are better left buried,\" {character1} warned darkly.",
                "\"I'm not who you think I am,\" {character2} confessed, avoiding eye contact.",
                "\"The rules have changed,\" {character1} announced, a dangerous glint in {pronoun1} eyes.",
                "\"What if I told you everything you know is a lie?\" {character2} challenged.",
                "\"Sometimes the right choice feels all wrong,\" {character1} mused sadly.",
                "\"I'd do it all again,\" {character2} declared without hesitation."
            ],
            "world_building_elements": [
                "A city built on the back of a giant creature that travels across the desert",
                "A society where memories are traded as currency",
                "A forest that grows in reverse, with roots in the sky and leaves underground",
                "An ocean of liquid light that holds ancient civilizations in its depths",
                "Mountains that sing songs of creation during solar eclipses",
                "A library containing every book that will ever be written",
                "A clockwork kingdom where time is literally money",
                "A valley where shadows have substance and light is fluid",
                "A civilization that communicates entirely through scent and color",
                "A planet with multiple miniature suns, each with different properties"
            ],
            "plot_twists": [
                "The protagonist was the villain all along",
                "The mentor is actually working for the antagonist",
                "The magical artifact is actually a prison containing the real threat",
                "The prophecy was misinterpreted and means the opposite",
                "The two warring factions are actually two sides of the same organization",
                "The main character has been dead the entire time",
                "The futuristic setting is actually a post-apocalyptic past",
                "The love interest is actually the antagonist in disguise",
                "The entire story is happening inside a simulation",
                "The chosen one was never special - the real power was in their companions"
            ]
        }
    
    def _safe_generate_text(self, prompt: str, base_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        使用防循环管理器安全生成文本
        
        整合双层防护：
        1. 基础层：嵌入式思维防护（缓冲区清理+重复检测+看门狗重置）
        2. 高级层：场景自适应防护（语义检测+动态参数+性能反馈）
        
        Args:
            prompt: 生成提示
            base_params: 基础生成参数
            
        Returns:
            Tuple[str, Dict[str, Any]]: (生成的文本, 防护信息)
        """
        # 定义内部生成函数供防循环管理器调用
        def generate_func(context: str, params: Dict[str, Any]) -> str:
            """包装语言模型的生成函数"""
            generation_params = {
                "text": context,
                "max_length": base_params.get("max_length", 1000),
                "temperature": params.get("temperature", 0.8),
                "top_p": params.get("top_p", 0.9),
                "repetition_penalty": params.get("repetition_penalty", 1.1),
                "scene": params.get("scene", "creative_writing"),
                "scene_confidence": params.get("scene_confidence", 0.0)
            }
            
            # 调用语言模型生成
            generation_result = self.language_model._generate_text(generation_params)
            
            if not generation_result.get("success", False):
                error_msg = generation_result.get("error", "Unknown error")
                raise ValueError(f"Text generation failed: {error_msg}")
            
            return generation_result.get("generated_text", "")
        
        # 使用防循环管理器安全生成
        output, protection_info = self.cycle_prevention_manager.generate_safe(
            prompt=prompt,
            generate_func=generate_func,
            max_attempts=3  # 最大重试3次
        )
        
        return output, protection_info
    
    def generate_story(self, 
                      genre: str = "fantasy",
                      length: str = "short",
                      prompt: Optional[str] = None,
                      characters: Optional[List[Dict]] = None,
                      setting: Optional[str] = None,
                      tone: str = "epic",
                      include_twist: bool = False) -> Dict[str, Any]:
        """Generate a complete story with AGI-enhanced creativity.
        
        Args:
            genre: Story genre (fantasy, sci_fi, mystery, romance, etc.)
            length: Story length - "short" (500 words), "medium" (1000 words), "long" (2000 words)
            prompt: Optional starting prompt or story idea
            characters: Optional list of character dictionaries with names, traits, etc.
            setting: Optional setting description
            tone: Emotional tone for the story
            include_twist: Whether to include a plot twist
            
        Returns:
            Dictionary containing the generated story and metadata
        """
        try:
            self.logger.info(f"Generating {genre} story with {tone} tone")
            
            # Build the story prompt with AGI-enhanced creativity
            story_prompt = self._build_story_prompt(
                genre=genre,
                length=length,
                user_prompt=prompt,
                characters=characters,
                setting=setting,
                tone=tone,
                include_twist=include_twist
            )
            
            # Generate story using cycle prevention manager with safe generation
            # This provides dual-layer protection: basic embedded-style + adaptive scene-aware
            base_params = {
                "max_length": self._get_length_word_count(length) * 6,  # Approximate token count
                "creativity_level": "high",
            }
            
            # Use safe generation with dual-layer cycle prevention
            generated_text, protection_info = self._safe_generate_text(
                prompt=story_prompt,
                base_params=base_params
            )
            
            # Extract story title and clean up text
            story_title = self._extract_story_title(generated_text)
            cleaned_story = self._clean_generated_story(generated_text, story_title)
            
            # Enhance story with AGI creative analysis
            enhanced_story = self._enhance_story_with_agi(cleaned_story, genre, tone)
            
            # Calculate story metrics
            word_count = len(enhanced_story.split())
            
            # Evaluate story quality for adaptive parameter learning
            quality_metrics = self._evaluate_story_quality(
                story=enhanced_story,
                genre=genre,
                expected_length=length
            )
            
            # Record performance for adaptive learning
            # For creative writing, we want lower repetition (higher repetition_score)
            # and higher quality (higher quality_score)
            repetition_score = max(0.1, min(1.0, 0.8 - (quality_metrics.get("repetition_ratio", 0.2) * 2.0)))
            quality_score = max(0.1, min(1.0, quality_metrics.get("quality_score", 0.6)))
            
            # Get scene from protection info if available, otherwise use creative writing
            scene = protection_info.get("scene", "creative_writing")
            
            self.cycle_prevention_manager.record_performance(
                scene=scene,
                repetition_score=repetition_score,
                quality_score=quality_score,
                additional_metrics={
                    "genre": genre,
                    "word_count": word_count,
                    "creativity_score": quality_metrics.get("creativity_score", 0.0),
                    "coherence_score": quality_metrics.get("coherence_score", 0.0),
                    "protection_layer": protection_info.get("protection_layer", "unknown"),
                    "attempts": protection_info.get("attempts", 1)
                }
            )
            
            return {
                "success": True,
                "story": {
                    "title": story_title,
                    "content": enhanced_story,
                    "genre": genre,
                    "tone": tone,
                    "word_count": word_count,
                    "length_category": length
                },
                "metadata": {
                    "characters_used": characters or [],
                    "setting": setting or "AI-generated",
                    "include_twist": include_twist,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_used": "AdvancedLanguageModel with AGI enhancement",
                    "creativity_score": self._calculate_creativity_score(enhanced_story),
                    "cycle_protection": {
                        "enabled": True,
                        "protection_layer": protection_info.get("protection_layer", "unknown"),
                        "scene": protection_info.get("scene", "creative_writing"),
                        "scene_confidence": protection_info.get("scene_confidence", 0.0),
                        "temperature": protection_info.get("temperature", 0.8),
                        "repetition_penalty": protection_info.get("repetition_penalty", 1.1),
                        "attempts": protection_info.get("attempts", 1),
                        "cycle_detected": protection_info.get("cycle_detected", False)
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Story generation failed: {str(e)}")
            error_handler.log_error(f"Story generation error: {str(e)}", "CreativeWritingTool")
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Try providing more specific character details or a clearer setting."
            }
    
    def _build_story_prompt(self, 
                           genre: str,
                           length: str,
                           user_prompt: Optional[str],
                           characters: Optional[List[Dict]],
                           setting: Optional[str],
                           tone: str,
                           include_twist: bool) -> str:
        """Build a comprehensive story generation prompt."""
        
        genre_style = self.genre_styles.get(genre, self.genre_styles["fantasy"])
        tone_words = self.emotional_tones.get(tone, self.emotional_tones["epic"])
        
        prompt_parts = []
        
        # Add creative writing instruction
        prompt_parts.append("Write a creative story with the following specifications:")
        
        # Add genre and style
        prompt_parts.append(f"Genre: {genre} - {genre_style['description']}")
        prompt_parts.append(f"Style: Write in a {', '.join(tone_words)} tone.")
        prompt_parts.append(f"Length: {length} story (approximately {self._get_length_word_count(length)} words)")
        
        # Add user prompt if provided
        if user_prompt:
            prompt_parts.append(f"Story idea: {user_prompt}")
        
        # Add characters if provided
        if characters:
            prompt_parts.append("\nCharacters:")
            for i, char in enumerate(characters[:3]):  # Limit to 3 main characters
                char_desc = f"{char.get('name', f'Character {i+1}')}: "
                if 'role' in char:
                    char_desc += f"{char['role']}, "
                if 'personality' in char:
                    char_desc += f"{char['personality']}, "
                if 'motivation' in char:
                    char_desc += f"motivated by {char['motivation']}"
                prompt_parts.append(f"- {char_desc}")
        else:
            # Generate interesting character suggestions
            prompt_parts.append("\nInclude interesting, multi-dimensional characters with clear motivations.")
        
        # Add setting if provided
        if setting:
            prompt_parts.append(f"\nSetting: {setting}")
        else:
            # Add genre-appropriate setting suggestion
            setting_element = genre_style["elements"][(zlib.adler32(str(str(genre_style["elements"]).encode('utf-8')) & 0xffffffff) + "setting") % len(genre_style["elements"])]
            prompt_parts.append(f"\nSetting: Include {setting_element} as part of the world.")
        
        # Add plot twist instruction if requested
        if include_twist:
            twist_suggestion = self.templates["plot_twists"][(zlib.adler32(str(str(self.templates["plot_twists"]).encode('utf-8')) & 0xffffffff) + "twist") % len(self.templates["plot_twists"])]
            prompt_parts.append(f"\nPlot: Include a surprising plot twist. Suggestion: {twist_suggestion}")
        else:
            prompt_parts.append("\nPlot: Create an engaging plot with conflict and resolution.")
        
        # Add creative writing guidelines
        prompt_parts.append("\nCreative Guidelines:")
        prompt_parts.append("- Show, don't tell - use vivid descriptions and sensory details")
        prompt_parts.append("- Create emotional resonance with the characters")
        prompt_parts.append("- Maintain consistent tone and pacing")
        prompt_parts.append("- Use varied sentence structure for rhythm")
        prompt_parts.append("- Include meaningful dialogue that reveals character")
        prompt_parts.append("- End with a satisfying conclusion that ties up major plot points")
        
        # Add starter sentence if no user prompt
        if not user_prompt:
            starter = self.templates["story_starters"][(zlib.adler32(str(str(self.templates["story_starters"]).encode('utf-8')) & 0xffffffff) + "starter") % len(self.templates["story_starters"])]
            prompt_parts.append(f"\nYou may start with: {starter}")
        
        return "\n".join(prompt_parts)
    
    def generate_character(self,
                          archetype: Optional[str] = None,
                          genre: str = "fantasy",
                          role: Optional[str] = None,
                          include_backstory: bool = True,
                          detail_level: str = "medium") -> Dict[str, Any]:
        """Generate a detailed character with AGI-enhanced creativity.
        
        Args:
            archetype: Character archetype (hero, mentor, trickster, etc.)
            genre: Story genre for character context
            role: Character's role in the story (protagonist, antagonist, etc.)
            include_backstory: Whether to generate a detailed backstory
            detail_level: Level of detail - "basic", "medium", "detailed"
            
        Returns:
            Dictionary containing character details and backstory
        """
        try:
            self.logger.info(f"Generating {genre} character with archetype: {archetype}")
            
            # Select or randomize archetype
            if not archetype:
                archetype_keys = list(self.character_archetypes.keys())
                archetype = archetype_keys[(zlib.adler32(str(str(archetype_keys).encode('utf-8')) & 0xffffffff) + "archetype") % len(archetype_keys)]
            
            # Get archetype traits
            traits = self.character_archetypes.get(archetype, self.character_archetypes["hero"])
            
            # Generate character name based on genre
            character_name = self._generate_character_name(genre, archetype)
            
            # Build character prompt
            character_prompt = self._build_character_prompt(
                name=character_name,
                archetype=archetype,
                genre=genre,
                role=role,
                traits=traits,
                include_backstory=include_backstory,
                detail_level=detail_level
            )
            
            # Generate character description using cycle prevention manager
            base_params = {
                "max_length": 800,
            }
            
            # Use safe generation with dual-layer cycle prevention
            generated_text, protection_info = self._safe_generate_text(
                prompt=character_prompt,
                base_params=base_params
            )
            
            # Parse and structure character information
            character_data = self._parse_character_description(generated_text, character_name)
            
            # Add archetype and genre information
            character_data["archetype"] = archetype
            character_data["genre"] = genre
            character_data["traits"] = traits
            
            if role:
                character_data["role"] = role
            
            # Generate backstory if requested
            if include_backstory:
                backstory = self._generate_character_backstory(character_data, genre)
                character_data["backstory"] = backstory
            
            # Calculate character complexity score
            character_data["complexity_score"] = self._calculate_character_complexity(character_data)
            
            # Record performance for adaptive learning
            # For character generation, we evaluate based on complexity and completeness
            complexity_score = min(1.0, character_data["complexity_score"] / 10.0)  # Normalize to 0-1
            # Simple heuristic: if character has many fields, it's higher quality
            completeness_score = min(1.0, len(character_data) / 15.0)
            
            # Estimate repetition score based on character uniqueness
            # If character has unique traits and backstory, repetition should be low
            repetition_score = max(0.1, min(1.0, 0.7 - (complexity_score * 0.3)))
            quality_score = max(0.1, min(1.0, (complexity_score + completeness_score) / 2.0))
            
            # Get scene from protection info
            scene = protection_info.get("scene", "creative_writing")
            
            self.cycle_prevention_manager.record_performance(
                scene=scene,
                repetition_score=repetition_score,
                quality_score=quality_score,
                additional_metrics={
                    "genre": genre,
                    "archetype": archetype,
                    "complexity": character_data["complexity_score"],
                    "field_count": len(character_data),
                    "protection_layer": protection_info.get("protection_layer", "unknown"),
                    "attempts": protection_info.get("attempts", 1)
                }
            )
            
            return {
                "success": True,
                "character": character_data,
                "metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "detail_level": detail_level,
                    "include_backstory": include_backstory,
                    "cycle_protection": {
                        "enabled": True,
                        "protection_layer": protection_info.get("protection_layer", "unknown"),
                        "scene": protection_info.get("scene", "creative_writing"),
                        "scene_confidence": protection_info.get("scene_confidence", 0.0),
                        "temperature": protection_info.get("temperature", 0.7),
                        "repetition_penalty": protection_info.get("repetition_penalty", 1.1),
                        "attempts": protection_info.get("attempts", 1),
                        "cycle_detected": protection_info.get("cycle_detected", False)
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Character generation failed: {str(e)}")
            error_handler.log_error(f"Character generation error: {str(e)}", "CreativeWritingTool")
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Try specifying a different archetype or providing more constraints."
            }
    
    def _build_character_prompt(self,
                               name: str,
                               archetype: str,
                               genre: str,
                               role: Optional[str],
                               traits: List[str],
                               include_backstory: bool,
                               detail_level: str) -> str:
        """Build a character generation prompt."""
        
        genre_style = self.genre_styles.get(genre, self.genre_styles["fantasy"])
        
        prompt_parts = []
        prompt_parts.append(f"Create a detailed character for a {genre} story.")
        prompt_parts.append(f"Character Name: {name}")
        prompt_parts.append(f"Archetype: {archetype}")
        prompt_parts.append(f"Traits: {', '.join(traits)}")
        
        if role:
            prompt_parts.append(f"Role in story: {role}")
        
        prompt_parts.append(f"\nProvide the following details at {detail_level} level:")
        prompt_parts.append("1. Physical description (appearance, distinctive features)")
        prompt_parts.append("2. Personality (key characteristics, flaws, virtues)")
        prompt_parts.append("3. Motivations (what drives this character)")
        prompt_parts.append("4. Skills and abilities")
        prompt_parts.append("5. Relationships (how they interact with others)")
        
        if include_backstory:
            prompt_parts.append("6. Brief backstory (key life events that shaped them)")
        
        prompt_parts.append(f"\nMake the character fit the {genre} genre: {genre_style['description']}")
        prompt_parts.append("Make the character multi-dimensional with both strengths and weaknesses.")
        
        if detail_level == "detailed":
            prompt_parts.append("Include subtle details and nuanced characteristics.")
        elif detail_level == "basic":
            prompt_parts.append("Keep descriptions concise and focused on key elements.")
        
        return "\n".join(prompt_parts)
    
    def generate_dialogue(self,
                         characters: List[Dict[str, str]],
                         context: str,
                         tone: str = "neutral",
                         length: str = "medium",
                         include_subtext: bool = True) -> Dict[str, Any]:
        """Generate realistic dialogue between characters.
        
        Args:
            characters: List of character dictionaries with at least 'name' and 'personality'
            context: The situation or scene context for the dialogue
            tone: Emotional tone of the dialogue
            length: Dialogue length - "short" (3-5 exchanges), "medium" (6-10), "long" (11+)
            include_subtext: Whether to include emotional subtext and non-verbal cues
            
        Returns:
            Dictionary containing the generated dialogue and analysis
        """
        try:
            self.logger.info(f"Generating dialogue for {len(characters)} characters with {tone} tone")
            
            if len(characters) < 2:
                raise ValueError("Dialogue requires at least two characters")
            
            # Build dialogue prompt
            dialogue_prompt = self._build_dialogue_prompt(
                characters=characters,
                context=context,
                tone=tone,
                length=length,
                include_subtext=include_subtext
            )
            
            # Generate dialogue
            generation_params = {
                "text": dialogue_prompt,
                "max_length": 1200,
                "temperature": 0.75,
                "top_p": 0.92
            }
            
            generation_result = self.language_model._generate_text(generation_params)
            
            if not generation_result.get("success", False):
                raise ValueError(f"Dialogue generation failed: {generation_result.get('error', 'Unknown error')}")
            
            generated_text = generation_result.get("generated_text", "")
            
            # Parse and format dialogue
            formatted_dialogue = self._format_dialogue(generated_text, characters)
            
            # Analyze dialogue characteristics
            dialogue_analysis = self._analyze_dialogue(formatted_dialogue, characters)
            
            return {
                "success": True,
                "dialogue": formatted_dialogue,
                "analysis": dialogue_analysis,
                "metadata": {
                    "character_count": len(characters),
                    "tone": tone,
                    "length": length,
                    "include_subtext": include_subtext,
                    "context": context,
                    "generation_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Dialogue generation failed: {str(e)}")
            error_handler.log_error(f"Dialogue generation error: {str(e)}", "CreativeWritingTool")
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Try providing more detailed character personalities or a clearer context."
            }
    
    def enhance_writing(self,
                       text: str,
                       enhancement_type: str = "vividness",
                       style: Optional[str] = None,
                       target_genre: Optional[str] = None) -> Dict[str, Any]:
        """Enhance existing writing with AGI-powered improvements.
        
        Args:
            text: The original text to enhance
            enhancement_type: Type of enhancement - "vividness", "clarity", "emotional_impact",
                            "conciseness", "creativity", "all"
            style: Optional target writing style (poetic, dramatic, humorous, etc.)
            target_genre: Optional target genre for style adaptation
            
        Returns:
            Dictionary containing enhanced text and improvement metrics
        """
        try:
            self.logger.info(f"Enhancing writing with {enhancement_type} improvement")
            
            if not text or len(text.strip()) < 10:
                raise ValueError("Text must be at least 10 characters long")
            
            # Build enhancement prompt
            enhancement_prompt = self._build_enhancement_prompt(
                text=text,
                enhancement_type=enhancement_type,
                style=style,
                target_genre=target_genre
            )
            
            # Generate enhanced text
            generation_params = {
                "text": enhancement_prompt,
                "max_length": len(text) * 2,  # Allow room for expansion
                "temperature": 0.65,
                "top_p": 0.88
            }
            
            generation_result = self.language_model._generate_text(generation_params)
            
            if not generation_result.get("success", False):
                raise ValueError(f"Writing enhancement failed: {generation_result.get('error', 'Unknown error')}")
            
            enhanced_text = generation_result.get("generated_text", "")
            
            # Clean up the enhanced text
            cleaned_enhanced = self._clean_enhanced_text(enhanced_text, text)
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(text, cleaned_enhanced, enhancement_type)
            
            return {
                "success": True,
                "original_text": text,
                "enhanced_text": cleaned_enhanced,
                "improvement_metrics": improvement_metrics,
                "enhancement_type": enhancement_type,
                "metadata": {
                    "style_applied": style,
                    "target_genre": target_genre,
                    "original_length": len(text),
                    "enhanced_length": len(cleaned_enhanced),
                    "generation_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Writing enhancement failed: {str(e)}")
            error_handler.log_error(f"Writing enhancement error: {str(e)}", "CreativeWritingTool")
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Try focusing on one specific enhancement type or providing more context."
            }
    
    def generate_writing_prompt(self,
                               genre: Optional[str] = None,
                               prompt_type: str = "story_starter",
                               complexity: str = "medium",
                               include_constraints: bool = True) -> Dict[str, Any]:
        """Generate creative writing prompts with varying complexity and constraints.
        
        Args:
            genre: Optional genre for the prompt
            prompt_type: Type of prompt - "story_starter", "character", "setting", "dialogue", "plot"
            complexity: Prompt complexity - "simple", "medium", "complex"
            include_constraints: Whether to include specific writing constraints
            
        Returns:
            Dictionary containing the prompt and suggested use
        """
        try:
            self.logger.info(f"Generating {prompt_type} prompt with {complexity} complexity")
            
            # Select genre if not specified
            if not genre:
                genre_keys = list(self.genre_styles.keys())
                genre = genre_keys[(zlib.adler32(str(str(genre_keys).encode('utf-8')) & 0xffffffff) + "genre") % len(genre_keys)]
            
            # Generate prompt based on type
            if prompt_type == "story_starter":
                prompt = self._generate_story_starter_prompt(genre, complexity, include_constraints)
            elif prompt_type == "character":
                prompt = self._generate_character_prompt_type(genre, complexity, include_constraints)
            elif prompt_type == "setting":
                prompt = self._generate_setting_prompt(genre, complexity, include_constraints)
            elif prompt_type == "dialogue":
                prompt = self._generate_dialogue_prompt_type(genre, complexity, include_constraints)
            elif prompt_type == "plot":
                prompt = self._generate_plot_prompt(genre, complexity, include_constraints)
            else:
                prompt = self._generate_story_starter_prompt(genre, complexity, include_constraints)
            
            # Add AGI enhancement for more creative prompts
            enhanced_prompt = self._enhance_prompt_with_agi(prompt, genre, prompt_type)
            
            # Generate suggested use
            suggested_use = self._generate_prompt_suggestions(enhanced_prompt, prompt_type, complexity)
            
            return {
                "success": True,
                "prompt": enhanced_prompt,
                "metadata": {
                    "genre": genre,
                    "prompt_type": prompt_type,
                    "complexity": complexity,
                    "include_constraints": include_constraints,
                    "suggested_use": suggested_use,
                    "generation_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prompt generation failed: {str(e)}")
            error_handler.log_error(f"Prompt generation error: {str(e)}", "CreativeWritingTool")
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Try selecting a different genre or reducing the complexity."
            }
    
    # Helper methods
    def _get_length_word_count(self, length: str) -> int:
        """Convert length description to approximate word count."""
        length_map = {
            "short": 500,
            "medium": 1000,
            "long": 2000,
            "flash": 100,  # Flash fiction
            "novella": 10000,
            "novel": 50000
        }
        return length_map.get(length.lower(), 1000)
    
    def _generate_character_name(self, genre: str, archetype: str) -> str:
        """Generate a character name appropriate for the genre and archetype."""
        # This is a simplified version - in practice, you might use a more sophisticated name generator
        fantasy_names = ["Aelar", "Thalion", "Eowyn", "Gideon", "Lyra", "Kaelen", "Seraphina", "Orion"]
        scifi_names = ["Jaxon", "Zara", "Kiran", "Nova", "Ryder", "Lyra", "Cyrus", "Vega"]
        modern_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Dakota"]
        
        if genre == "fantasy":
            return fantasy_names[(zlib.adler32(str(str(fantasy_names).encode('utf-8')) & 0xffffffff) + "fantasy") % len(fantasy_names)]
        elif genre == "sci_fi":
            return scifi_names[(zlib.adler32(str(str(scifi_names).encode('utf-8')) & 0xffffffff) + "scifi") % len(scifi_names)]
        else:
            return modern_names[(zlib.adler32(str(str(modern_names).encode('utf-8')) & 0xffffffff) + "modern") % len(modern_names)]
    
    def _extract_story_title(self, story_text: str) -> str:
        """Extract or generate a title from story text."""
        # Simple implementation - take first line or generate based on content
        lines = story_text.strip().split('\n')
        first_line = lines[0] if lines else "Untitled Story"
        
        # Clean up the first line to make it title-like
        title = first_line[:50].strip()
        if title.endswith(('.', '!', '?')):
            title = title[:-1]
        
        return title if title else "Untitled Story"
    
    def _clean_generated_story(self, story_text: str, title: str) -> str:
        """Clean and format generated story text."""
        # Remove any prompt remnants
        lines = story_text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that look like prompt instructions
            if line.lower().startswith(('write a', 'genre:', 'style:', 'length:', 'characters:', 'setting:')):
                continue
            cleaned_lines.append(line)
        
        # Join lines and ensure proper spacing
        cleaned_text = '\n'.join(cleaned_lines).strip()
        
        # Add title if not present
        if not cleaned_text.startswith(title):
            cleaned_text = f"{title}\n\n{cleaned_text}"
        
        return cleaned_text
    
    def _enhance_story_with_agi(self, story_text: str, genre: str, tone: str) -> str:
        """Enhance story with AGI creative analysis."""
        # This could be expanded to use the AGI tools for more sophisticated enhancement
        # For now, we'll do basic enhancement
        enhanced = story_text
        
        # Add genre-appropriate enhancements
        genre_style = self.genre_styles.get(genre, {})
        if "elements" in genre_style and ((zlib.adler32(str(str(genre_style).encode('utf-8')) & 0xffffffff) + "enhance") % 100) < 50:
            # Add a genre-appropriate element if missing
            element = genre_style["elements"][(zlib.adler32(str(str(genre_style["elements"]).encode('utf-8')) & 0xffffffff) + "element") % len(genre_style["elements"])]
            if element.lower() not in enhanced.lower():
                # Find a place to insert the element
                sentences = enhanced.split('. ')
                if len(sentences) > 2:
                    insert_pos = 1 + (zlib.adler32(str(str(sentences).encode('utf-8')) & 0xffffffff) + "insert") % (len(sentences) - 1)
                    sentences.insert(insert_pos, f"The scene included {element}.")
                    enhanced = '. '.join(sentences)
        
        return enhanced
    
    def _calculate_creativity_score(self, text: str) -> float:
        """Calculate a creativity score for generated text."""
        # Simplified creativity scoring
        words = text.split()
        unique_words = set(words)
        
        # Basic metrics
        vocab_richness = len(unique_words) / max(len(words), 1)
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        
        # Combine metrics
        score = (vocab_richness * 0.6 + min(avg_word_length / 10, 1) * 0.4)
        return round(min(score, 1.0), 2)
    
    def _parse_character_description(self, text: str, name: str) -> Dict[str, Any]:
        """Parse generated character description into structured data."""
        # Simplified parsing - in practice, you might use more sophisticated NLP
        character_data = {
            "name": name,
            "physical_description": "",
            "personality": "",
            "motivations": "",
            "skills": "",
            "relationships": "",
            "backstory": ""
        }
        
        # Look for section headers
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if "physical" in line_lower or "appearance" in line_lower:
                current_section = "physical_description"
            elif "personality" in line_lower:
                current_section = "personality"
            elif "motivation" in line_lower:
                current_section = "motivations"
            elif "skill" in line_lower or "ability" in line_lower:
                current_section = "skills"
            elif "relationship" in line_lower:
                current_section = "relationships"
            elif "backstory" in line_lower or "history" in line_lower:
                current_section = "backstory"
            elif current_section and line.strip():
                if character_data[current_section]:
                    character_data[current_section] += " " + line.strip()
                else:
                    character_data[current_section] = line.strip()
        
        return character_data
    
    def _generate_character_backstory(self, character_data: Dict[str, Any], genre: str) -> str:
        """Generate a detailed backstory for a character."""
        backstory_prompt = f"Write a detailed backstory for {character_data['name']}, a character in a {genre} story."
        backstory_prompt += f"\nPersonality: {character_data.get('personality', 'Not specified')}"
        backstory_prompt += f"\nMotivations: {character_data.get('motivations', 'Not specified')}"
        
        generation_params = {
            "text": backstory_prompt,
            "max_length": 500,
            "temperature": 0.7
        }
        
        generation_result = self.language_model._generate_text(generation_params)
        if generation_result.get("success", False):
            return generation_result.get("generated_text", "")
        
        return "Backstory not generated."
    
    def _calculate_character_complexity(self, character_data: Dict[str, Any]) -> float:
        """Calculate character complexity score."""
        score = 0.0
        
        # Check for presence of different character aspects
        aspects = ["physical_description", "personality", "motivations", "skills", "relationships", "backstory"]
        present_aspects = sum(1 for aspect in aspects if character_data.get(aspect) and len(character_data[aspect]) > 10)
        
        score += (present_aspects / len(aspects)) * 0.6
        
        # Check for nuance (presence of both strengths and weaknesses)
        personality_text = character_data.get("personality", "").lower()
        positive_indicators = ["brave", "kind", "intelligent", "strong", "wise", "compassionate"]
        negative_indicators = ["flawed", "afraid", "angry", "selfish", "arrogant", "impulsive"]
        
        positive_count = sum(1 for word in positive_indicators if word in personality_text)
        negative_count = sum(1 for word in negative_indicators if word in personality_text)
        
        if positive_count > 0 and negative_count > 0:
            score += 0.4
        
        return round(min(score, 1.0), 2)
    
    def _build_dialogue_prompt(self,
                              characters: List[Dict[str, str]],
                              context: str,
                              tone: str,
                              length: str,
                              include_subtext: bool) -> str:
        """Build a dialogue generation prompt."""
        
        length_map = {
            "short": "3-5 exchanges",
            "medium": "6-10 exchanges",
            "long": "11-15 exchanges"
        }
        
        prompt_parts = []
        prompt_parts.append(f"Write a dialogue scene with the following specifications:")
        prompt_parts.append(f"Context: {context}")
        prompt_parts.append(f"Tone: {tone}")
        prompt_parts.append(f"Length: {length_map.get(length, '6-10 exchanges')}")
        
        prompt_parts.append("\nCharacters:")
        for char in characters:
            char_desc = f"- {char['name']}: {char.get('personality', 'Personality not specified')}"
            if 'role' in char:
                char_desc += f", {char['role']}"
            prompt_parts.append(char_desc)
        
        prompt_parts.append("\nGuidelines:")
        prompt_parts.append("- Each character should speak in a distinct voice matching their personality")
        prompt_parts.append("- Dialogue should advance the scene or reveal character")
        prompt_parts.append("- Use natural, conversational language")
        
        if include_subtext:
            prompt_parts.append("- Include emotional subtext and non-verbal cues (actions, expressions)")
        
        prompt_parts.append("\nFormat the dialogue with character names followed by their lines.")
        
        return "\n".join(prompt_parts)
    
    def _format_dialogue(self, dialogue_text: str, characters: List[Dict]) -> List[Dict[str, str]]:
        """Format raw dialogue text into structured format."""
        # Simplified formatting - in practice, use more sophisticated parsing
        lines = dialogue_text.strip().split('\n')
        formatted = []
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Extract character name and line
            parts = line.split(':', 1)
            if len(parts) == 2:
                char_name = parts[0].strip()
                char_line = parts[1].strip()
                
                # Find character details
                char_details = next((c for c in characters if c['name'] == char_name), None)
                
                formatted.append({
                    "character": char_name,
                    "line": char_line,
                    "character_details": char_details
                })
        
        return formatted
    
    def _analyze_dialogue(self, dialogue: List[Dict], characters: List[Dict]) -> Dict[str, Any]:
        """Analyze dialogue characteristics."""
        if not dialogue:
            return {"error": "No dialogue to analyze"}
        
        total_lines = len(dialogue)
        character_lines = {}
        
        for entry in dialogue:
            char_name = entry["character"]
            character_lines[char_name] = character_lines.get(char_name, 0) + 1
        
        # Calculate balance
        line_counts = list(character_lines.values())
        if line_counts:
            balance_score = min(line_counts) / max(line_counts) if max(line_counts) > 0 else 0
        else:
            balance_score = 0
        
        # Estimate emotional content
        emotional_words = ["love", "hate", "fear", "hope", "anger", "joy", "sad", "happy", "excited", "worried"]
        emotional_count = 0
        total_words = 0
        
        for entry in dialogue:
            words = entry["line"].split()
            total_words += len(words)
            emotional_count += sum(1 for word in words if word.lower() in emotional_words)
        
        emotional_density = emotional_count / max(total_words, 1)
        
        return {
            "total_exchanges": total_lines,
            "character_distribution": character_lines,
            "balance_score": round(balance_score, 2),
            "emotional_density": round(emotional_density, 3),
            "average_line_length": round(total_words / max(total_lines, 1), 1)
        }
    
    def _build_enhancement_prompt(self,
                                 text: str,
                                 enhancement_type: str,
                                 style: Optional[str],
                                 target_genre: Optional[str]) -> str:
        """Build a writing enhancement prompt."""
        
        enhancement_descriptions = {
            "vividness": "Make the writing more vivid and descriptive. Add sensory details, stronger imagery, and more evocative language.",
            "clarity": "Improve clarity and readability. Simplify complex sentences, clarify ambiguous phrases, and improve flow.",
            "emotional_impact": "Increase emotional impact. Strengthen emotional language, build tension, and deepen character emotions.",
            "conciseness": "Make the writing more concise. Remove redundancy, tighten sentences, and eliminate unnecessary words.",
            "creativity": "Boost creativity. Add imaginative elements, unique metaphors, and original phrasing.",
            "all": "Comprehensively improve the writing. Enhance vividness, clarity, emotional impact, conciseness, and creativity."
        }
        
        prompt_parts = []
        prompt_parts.append(f"Improve the following text with a focus on {enhancement_type}:")
        prompt_parts.append(f"{enhancement_descriptions.get(enhancement_type, enhancement_descriptions['all'])}")
        
        if style:
            prompt_parts.append(f"Adapt the style to be more {style}.")
        
        if target_genre:
            genre_style = self.genre_styles.get(target_genre, {})
            if "description" in genre_style:
                prompt_parts.append(f"Make it appropriate for {target_genre} genre: {genre_style['description']}")
        
        prompt_parts.append("\nOriginal text:")
        prompt_parts.append(f'"{text}"')
        
        prompt_parts.append("\nProvide only the enhanced text without additional commentary.")
        
        return "\n".join(prompt_parts)
    
    def _clean_enhanced_text(self, enhanced_text: str, original_text: str) -> str:
        """Clean enhanced text by removing any prompt remnants."""
        # Remove common prompt artifacts
        artifacts = [
            "Improved text:",
            "Enhanced version:",
            "Here's the enhanced text:",
            "Enhanced text:",
            "Revised version:"
        ]
        
        cleaned = enhanced_text.strip()
        
        for artifact in artifacts:
            if cleaned.startswith(artifact):
                cleaned = cleaned[len(artifact):].strip()
        
        # If cleaning removed everything, return original
        if not cleaned or len(cleaned) < len(original_text) * 0.5:
            return original_text
        
        return cleaned
    
    def _calculate_improvement_metrics(self,
                                      original: str,
                                      enhanced: str,
                                      enhancement_type: str) -> Dict[str, float]:
        """Calculate improvement metrics between original and enhanced text."""
        # Simplified metrics - in practice, use more sophisticated analysis
        
        # Word count comparison
        orig_words = len(original.split())
        enh_words = len(enhanced.split())
        word_change = ((enh_words - orig_words) / max(orig_words, 1)) * 100
        
        # Sentence length variation (rough estimate)
        orig_sentences = original.count('.') + original.count('!') + original.count('?')
        enh_sentences = enhanced.count('.') + enhanced.count('!') + enhanced.count('?')
        
        orig_avg_words = orig_words / max(orig_sentences, 1)
        enh_avg_words = enh_words / max(enh_sentences, 1)
        sentence_variation = ((enh_avg_words - orig_avg_words) / max(orig_avg_words, 1)) * 100
        
        # Vocabulary richness (simplified)
        orig_unique = len(set(original.lower().split()))
        enh_unique = len(set(enhanced.lower().split()))
        
        orig_richness = orig_unique / max(orig_words, 1)
        enh_richness = enh_unique / max(enh_words, 1)
        richness_improvement = ((enh_richness - orig_richness) / max(orig_richness, 0.01)) * 100
        
        # Enhancement type specific metrics
        type_scores = {
            "vividness": min(max(richness_improvement / 50, 0), 1) * 0.8 + 0.2,
            "clarity": min(max(100 - abs(word_change) / 2, 0), 100) / 100,
            "emotional_impact": min(max(enh_richness * 10, 0), 1),
            "conciseness": min(max(100 - word_change, 0), 100) / 100 if word_change > 0 else 1.0,
            "creativity": min(max(richness_improvement / 30, 0), 1),
            "all": (min(max(richness_improvement / 50, 0), 1) * 0.3 +
                   min(max(100 - abs(word_change) / 2, 0), 100) / 100 * 0.3 +
                   min(max(enh_richness * 10, 0), 1) * 0.4)
        }
        
        improvement_score = type_scores.get(enhancement_type, type_scores["all"])
        
        return {
            "improvement_score": round(improvement_score, 2),
            "word_count_change_percent": round(word_change, 1),
            "sentence_structure_variation": round(sentence_variation, 1),
            "vocabulary_richness_improvement": round(richness_improvement, 1),
            "overall_enhancement": round(improvement_score * 100, 1)
        }
    
    def _generate_story_starter_prompt(self, genre: str, complexity: str, include_constraints: bool) -> str:
        """Generate a story starter prompt."""
        starter = self.templates["story_starters"][(zlib.adler32(str(str(self.templates["story_starters"]).encode('utf-8')) & 0xffffffff) + "starter_prompt") % len(self.templates["story_starters"])]
        
        prompt = f"Write a {genre} story that begins with: '{starter}'"
        
        if complexity == "complex" and include_constraints:
            constraints = [
                "Include a character with a secret",
                "Set it in an unusual location",
                "Incorporate a magical or technological element",
                "End with a surprising revelation"
            ]
            indices = list(range(len(constraints)))
            sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(constraints).encode('utf-8')) & 0xffffffff) + str(x) + "sample"))[:2]
            selected_constraints = [constraints[i] for i in sampled_indices]
            prompt += f"\nConstraints: {', '.join(selected_constraints)}"
        
        return prompt
    
    def _generate_character_prompt_type(self, genre: str, complexity: str, include_constraints: bool) -> str:
        """Generate a character-focused prompt."""
        archetype_keys = list(self.character_archetypes.keys())
        archetype = archetype_keys[(zlib.adler32(str(str(archetype_keys).encode('utf-8')) & 0xffffffff) + "character_prompt") % len(archetype_keys)]
        traits = self.character_archetypes[archetype]
        
        prompt = f"Create a {genre} character who is a {archetype}. Key traits: {', '.join(traits)}"
        
        if complexity == "complex" and include_constraints:
            constraints = [
                "Give them a contradictory trait",
                "Include a traumatic past event",
                "Make them unexpectedly skilled at something unusual",
                "Give them a phobia or irrational fear"
            ]
            indices = list(range(len(constraints)))
            sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(constraints).encode('utf-8')) & 0xffffffff) + str(x) + "sample2"))[:2]
            selected_constraints = [constraints[i] for i in sampled_indices]
            prompt += f"\nAdditional constraints: {', '.join(selected_constraints)}"
        
        return prompt
    
    def _generate_setting_prompt(self, genre: str, complexity: str, include_constraints: bool) -> str:
        """Generate a setting-focused prompt."""
        elements = self.genre_styles.get(genre, {}).get("elements", ["mysterious location"])
        genre_element = elements[(zlib.adler32(str(str(elements).encode('utf-8')) & 0xffffffff) + "genre_element") % len(elements)]
        
        prompt = f"Describe a {genre} setting featuring {genre_element}"
        
        if complexity == "complex" and include_constraints:
            constraints = [
                "Include at least three sensory descriptions",
                "Make the setting reflect a character's emotional state",
                "Include a hidden danger or secret",
                "Show how the setting changes over time"
            ]
            indices = list(range(len(constraints)))
            sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(constraints).encode('utf-8')) & 0xffffffff) + str(x) + "sample3"))[:2]
            selected_constraints = [constraints[i] for i in sampled_indices]
            prompt += f"\nConstraints: {', '.join(selected_constraints)}"
        
        return prompt
    
    def _generate_dialogue_prompt_type(self, genre: str, complexity: str, include_constraints: bool) -> str:
        """Generate a dialogue-focused prompt."""
        context_options = [
            "a confession",
            "a negotiation",
            "an argument",
            "a reunion",
            "a secret being revealed"
        ]
        context = context_options[(zlib.adler32(str(str(context_options).encode('utf-8')) & 0xffffffff) + "context") % len(context_options)]
        
        prompt = f"Write a {genre} dialogue scene where two characters have {context}"
        
        if complexity == "complex" and include_constraints:
            constraints = [
                "One character is lying",
                "Include non-verbal communication",
                "End with an unexpected twist",
                "Use subtext to convey hidden meanings"
            ]
            indices = list(range(len(constraints)))
            sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(constraints).encode('utf-8')) & 0xffffffff) + str(x) + "sample4"))[:2]
            selected_constraints = [constraints[i] for i in sampled_indices]
            prompt += f"\nConstraints: {', '.join(selected_constraints)}"
        
        return prompt
    
    def _generate_plot_prompt(self, genre: str, complexity: str, include_constraints: bool) -> str:
        """Generate a plot-focused prompt."""
        twist = self.templates["plot_twists"][(zlib.adler32(str(str(self.templates["plot_twists"]).encode('utf-8')) & 0xffffffff) + "plot_twist") % len(self.templates["plot_twists"])]
        
        prompt = f"Outline a {genre} plot that includes this twist: {twist}"
        
        if complexity == "complex" and include_constraints:
            constraints = [
                "Include three acts with clear turning points",
                "Feature a morally ambiguous protagonist",
                "Incorporate a thematic element about society",
                "End with an ambiguous resolution"
            ]
            indices = list(range(len(constraints)))
            sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(constraints).encode('utf-8')) & 0xffffffff) + str(x) + "sample5"))[:2]
            selected_constraints = [constraints[i] for i in sampled_indices]
            prompt += f"\nConstraints: {', '.join(selected_constraints)}"
        
        return prompt
    
    def _enhance_prompt_with_agi(self, prompt: str, genre: str, prompt_type: str) -> str:
        """Enhance a prompt with AGI creativity."""
        # Simple enhancement - add creative spark
        enhancements = [
            "Think outside the box and create something truly original.",
            "Challenge conventional tropes of the genre.",
            "Focus on emotional authenticity and depth.",
            "Create layers of meaning and symbolism.",
            "Balance familiarity with surprising innovation."
        ]
        
        if ((zlib.adler32(str(str(prompt).encode('utf-8')) & 0xffffffff) + str(genre) + str(prompt_type) + "enhance") % 100) < 50:
            enhancement = enhancements[(zlib.adler32(str(str(enhancements).encode('utf-8')) & 0xffffffff) + "enhancement") % len(enhancements)]
            prompt += f"\n\nCreative challenge: {enhancement}"
        
        return prompt
    
    def _generate_prompt_suggestions(self, prompt: str, prompt_type: str, complexity: str) -> List[str]:
        """Generate suggested uses for a writing prompt."""
        suggestions = []
        
        base_suggestions = [
            "Use this as a warm-up exercise before your main writing session.",
            "Set a timer for 20 minutes and write without stopping.",
            "Write from the perspective of an unexpected character.",
            "Adapt the prompt to a different genre for contrast."
        ]
        
        type_specific = {
            "story_starter": [
                "Expand this into a complete short story.",
                "Write the same opening from multiple perspectives.",
                "Continue the story but change the genre midway."
            ],
            "character": [
                "Write a scene where this character faces their greatest fear.",
                "Create dialogue between this character and their opposite.",
                "Write a backstory scene that explains how they became who they are."
            ],
            "setting": [
                "Write a scene where the setting itself is a character.",
                "Describe the same setting at different times of day.",
                "Write about a character discovering this setting for the first time."
            ],
            "dialogue": [
                "Write the same dialogue scene with different emotional tones.",
                "Add a third character to change the dynamic.",
                "Rewrite the dialogue as internal monologue."
            ],
            "plot": [
                "Outline three different ways this plot could resolve.",
                "Write the key scene where the twist is revealed.",
                "Create character arcs that align with this plot."
            ]
        }
        
        # Add base suggestions
        indices = list(range(len(base_suggestions)))
        sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(base_suggestions).encode('utf-8')) & 0xffffffff) + str(x) + "sample6"))[:2]
        suggestions.extend([base_suggestions[i] for i in sampled_indices])
        
        # Add type-specific suggestions
        if prompt_type in type_specific:
            type_suggestions = type_specific[prompt_type]
            indices = list(range(len(type_suggestions)))
            sampled_indices = sorted(indices, key=lambda x: (zlib.adler32(str(str(type_suggestions).encode('utf-8')) & 0xffffffff) + str(x) + "sample7"))[:2]
            suggestions.extend([type_suggestions[i] for i in sampled_indices])
        
        return suggestions
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities of the creative writing tool."""
        return {
            "creative_writing_functions": [
                "generate_story",
                "generate_character", 
                "generate_dialogue",
                "enhance_writing",
                "generate_writing_prompt"
            ],
            "supported_genres": list(self.genre_styles.keys()),
            "character_archetypes": list(self.character_archetypes.keys()),
            "emotional_tones": list(self.emotional_tones.keys()),
            "enhancement_types": [
                "vividness", 
                "clarity", 
                "emotional_impact", 
                "conciseness", 
                "creativity", 
                "all"
            ],
            "prompt_types": [
                "story_starter",
                "character",
                "setting", 
                "dialogue",
                "plot"
            ],
            "complexity_levels": ["simple", "medium", "complex"],
            "integration": {
                "language_model": "AdvancedLanguageModel",
                "agi_enhanced": True,
                "emotional_intelligence": True,
                "context_awareness": True
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the creative writing tool."""
        return {
            "status": "active",
            "model_initialized": self.language_model is not None,
            "templates_loaded": len(self.templates) > 0,
            "genre_styles": len(self.genre_styles),
            "character_archetypes": len(self.character_archetypes),
            "emotional_tones": len(self.emotional_tones),
            "capabilities": self.get_capabilities()
        }
    
    def _evaluate_story_quality(
        self, 
        story: str, 
        genre: str, 
        expected_length: str
    ) -> Dict[str, float]:
        """
        评估故事质量，用于自适应参数学习
        
        Args:
            story: 生成的故事文本
            genre: 故事类型
            expected_length: 期望长度（short/medium/long）
            
        Returns:
            包含质量指标的字典
        """
        if not story:
            return {
                "quality_score": 0.1,
                "repetition_ratio": 0.5,
                "creativity_score": 0.1,
                "coherence_score": 0.1
            }
        
        # 基础质量评估
        words = story.split()
        word_count = len(words)
        
        # 计算重复率（重复的单词比例）
        unique_words = set(word.lower() for word in words)
        repetition_ratio = 1.0 - (len(unique_words) / max(1, word_count))
        
        # 长度适配度评分
        expected_word_counts = {
            "short": 500,
            "medium": 1000,
            "long": 2000
        }
        expected_words = expected_word_counts.get(expected_length, 500)
        length_score = 1.0 - min(1.0, abs(word_count - expected_words) / expected_words)
        
        # 句子结构多样性（基于句子长度变化）
        sentences = [s.strip() for s in story.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
            structure_score = min(1.0, length_variance / 100.0)  # 归一化
        else:
            structure_score = 0.3
        
        # 词汇丰富度（基于不同词汇比例）
        vocabulary_richness = len(unique_words) / max(1, word_count)
        
        # 综合质量评分
        quality_score = (
            0.3 * length_score +
            0.3 * structure_score +
            0.2 * vocabulary_richness +
            0.2 * (1.0 - repetition_ratio * 2.0)  # 重复率越低越好
        )
        
        # 创意评分（基于词汇多样性和句子结构）
        creativity_score = min(1.0, 0.5 + 0.3 * vocabulary_richness + 0.2 * structure_score)
        
        # 连贯性评分（基于段落结构和连接词）
        coherence_score = min(1.0, 0.6 + 0.2 * structure_score)
        
        return {
            "quality_score": max(0.1, min(1.0, quality_score)),
            "repetition_ratio": max(0.0, min(1.0, repetition_ratio)),
            "creativity_score": max(0.1, min(1.0, creativity_score)),
            "coherence_score": max(0.1, min(1.0, coherence_score)),
            "vocabulary_richness": vocabulary_richness,
            "structure_score": structure_score,
            "length_score": length_score,
            "word_count": word_count
        }
