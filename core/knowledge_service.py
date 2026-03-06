"""
Engineering Knowledge Service
Loads and queries engineering knowledge JSON data
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class EngineeringKnowledgeService:
    """Service for loading and querying engineering knowledge data"""

    def __init__(self, data_dir: str = None):
        """
        Initialize the knowledge service

        Args:
            data_dir: Directory containing knowledge JSON files
                     If None, uses default location
        """
        if data_dir is None:
            # Default to core/data/knowledge directory
            self.data_dir = Path(__file__).parent / "data" / "knowledge"
        else:
            self.data_dir = Path(data_dir)

        self.knowledge_bases: Dict[str, Dict[str, Any]] = {}
        self.load_all_knowledge()

    def load_all_knowledge(self) -> None:
        """Load all engineering knowledge JSON files"""
        if not self.data_dir.exists():
            print(f"Warning: Knowledge data directory does not exist: {self.data_dir}")
            return

        json_files = list(self.data_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON files found in {self.data_dir}")
            return

        for json_file in json_files:
            if json_file.name == "engineering.json":
                # Skip the generic engineering.json if exists
                continue
                
            # Skip schema and non-knowledge files
            if json_file.name in ["knowledge_schema.json", "self_learning_knowledge.json"]:
                # These are not knowledge base files
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Convert legacy format if needed
                data = self._convert_legacy_format(data, json_file.name)
                
                # Validate the loaded data
                if not self._validate_knowledge_data(data, json_file.name):
                    print(f"Warning: Invalid knowledge data structure in {json_file.name}, skipping")
                    continue

                # Extract knowledge base information
                knowledge_base = data.get("knowledge_base", {})
                domain = knowledge_base.get("domain", json_file.stem)

                # Store the knowledge base
                self.knowledge_bases[domain] = {
                    "domain": domain,
                    "name": knowledge_base.get("name", {}),
                    "description": knowledge_base.get("description", {}),
                    "categories": knowledge_base.get("categories", []),
                    "timestamp": knowledge_base.get("timestamp"),
                    "version": knowledge_base.get("version"),
                    "file_path": str(json_file),
                }

                print(f"Loaded knowledge base: {domain}")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file {json_file}: {e}")
            except Exception as e:
                print(f"Error loading knowledge file {json_file}: {e}")

    def _validate_knowledge_data(self, data: Dict[str, Any], filename: str) -> bool:
        """
        Validate knowledge data structure
        
        Args:
            data: Loaded JSON data
            filename: Source filename for error messages
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Basic structure validation
            if not isinstance(data, dict):
                print(f"Validation error in {filename}: Root must be a dictionary")
                return False
                
            if "knowledge_base" not in data:
                print(f"Validation error in {filename}: Missing 'knowledge_base' key")
                return False
                
            knowledge_base = data["knowledge_base"]
            
            # Check required fields
            required_fields = ["domain", "name", "description", "categories"]
            for field in required_fields:
                if field not in knowledge_base:
                    print(f"Validation error in {filename}: Missing required field '{field}' in knowledge_base")
                    return False
            
            # Validate domain
            if not isinstance(knowledge_base["domain"], str) or not knowledge_base["domain"]:
                print(f"Validation error in {filename}: 'domain' must be a non-empty string")
                return False
            
            # Validate name and description (must be dictionaries with at least 'en' key)
            for field in ["name", "description"]:
                field_value = knowledge_base[field]
                if not isinstance(field_value, dict):
                    print(f"Validation error in {filename}: '{field}' must be a dictionary")
                    return False
                    
                if "en" not in field_value:
                    print(f"Validation error in {filename}: '{field}' must contain 'en' (English) translation")
                    return False
                    
                for lang_code, text in field_value.items():
                    if not isinstance(lang_code, str) or len(lang_code) != 2:
                        print(f"Validation warning in {filename}: '{field}' contains invalid language code '{lang_code}'")
                        # Continue validation but warn
                    
                    if not isinstance(text, str) or not text:
                        print(f"Validation error in {filename}: '{field}.{lang_code}' must be a non-empty string")
                        return False
            
            # Validate categories
            categories = knowledge_base["categories"]
            if not isinstance(categories, list):
                print(f"Validation error in {filename}: 'categories' must be a list")
                return False
                
            if len(categories) == 0:
                print(f"Validation warning in {filename}: 'categories' list is empty")
                # Allow empty categories but warn
            
            for i, category in enumerate(categories):
                if not isinstance(category, dict):
                    print(f"Validation error in {filename}: category at index {i} must be a dictionary")
                    return False
                    
                # Check required category fields
                category_required = ["id", "name", "concepts"]
                for field in category_required:
                    if field not in category:
                        print(f"Validation error in {filename}: category {i} missing required field '{field}'")
                        return False
                
                # Validate category name (similar to domain name/description)
                if not isinstance(category["name"], dict) or "en" not in category["name"]:
                    print(f"Validation error in {filename}: category {i} 'name' must be a dictionary with 'en' key")
                    return False
                
                # Validate concepts
                concepts = category["concepts"]
                if not isinstance(concepts, list):
                    print(f"Validation error in {filename}: category {i} 'concepts' must be a list")
                    return False
                    
                for j, concept in enumerate(concepts):
                    if not isinstance(concept, dict):
                        print(f"Validation error in {filename}: concept {j} in category {i} must be a dictionary")
                        return False
                    
                    concept_required = ["id", "name", "description"]
                    for field in concept_required:
                        if field not in concept:
                            print(f"Validation error in {filename}: concept {j} in category {i} missing required field '{field}'")
                            return False
                    
                    # Validate concept name and description
                    for field in ["name", "description"]:
                        field_value = concept[field]
                        if not isinstance(field_value, dict) or "en" not in field_value:
                            print(f"Validation error in {filename}: concept {j}.{field} must be a dictionary with 'en' key")
                            return False
            
            # If all validation passed
            return True
            
        except Exception as e:
            print(f"Validation error in {filename}: Unexpected error during validation: {e}")
            return False

    def _convert_legacy_format(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Convert legacy knowledge base format to new format
        
        Args:
            data: Loaded JSON data
            filename: Source filename for logging
            
        Returns:
            Converted data in new format
        """
        # If already in new format (has knowledge_base key), return as-is
        if "knowledge_base" in data:
            return data
            
        # Check if this is legacy format (has domain but no knowledge_base)
        if "domain" not in data:
            # Not a valid knowledge base file
            return data
            
        print(f"Converting legacy format: {filename}")
        
        # Create knowledge_base structure
        knowledge_base = {
            "domain": data.get("domain"),
            "name": data.get("name", {}),
            "description": data.get("description", {}),
            "categories": [],
            "timestamp": data.get("timestamp"),
            "version": data.get("version", "1.0"),
        }
        
        # Convert concepts to a category if present
        if "concepts" in data and isinstance(data["concepts"], list):
            concepts_category = {
                "id": "general_concepts",
                "name": {
                    "en": "General Concepts",
                    "zh": "通用概念",
                    "de": "Allgemeine Konzepte",
                    "ja": "一般概念",
                    "ru": "Общие понятия"
                },
                "description": {
                    "en": "General concepts in this domain",
                    "zh": "本领域的一般概念",
                    "de": "Allgemeine Konzepte in diesem Bereich",
                    "ja": "この分野の一般概念",
                    "ru": "Общие понятия в этой области"
                },
                "concepts": data["concepts"]
            }
            knowledge_base["categories"].append(concepts_category)
        
        # Convert principles to a category if present
        if "principles" in data and isinstance(data["principles"], list):
            principles_category = {
                "id": "principles",
                "name": {
                    "en": "Principles",
                    "zh": "原则",
                    "de": "Prinzipien",
                    "ja": "原則",
                    "ru": "Принципы"
                },
                "description": {
                    "en": "Fundamental principles in this domain",
                    "zh": "本领域的基本原则",
                    "de": "Grundlegende Prinzipien in diesem Bereich",
                    "ja": "この分野の基本原理",
                    "ru": "Основные принципы в этой области"
                },
                "concepts": data["principles"]
            }
            knowledge_base["categories"].append(principles_category)
        
        # If no categories were created, create an empty categories list
        if len(knowledge_base["categories"]) == 0:
            knowledge_base["categories"] = []
        
        # Return new format data
        return {"knowledge_base": knowledge_base}

    def get_domains(self) -> List[str]:
        """Get list of available knowledge domains"""
        return list(self.knowledge_bases.keys())

    def get_domain_info(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific domain"""
        if domain not in self.knowledge_bases:
            return None

        domain_data = self.knowledge_bases[domain]
        return {
            "domain": domain_data["domain"],
            "name": domain_data["name"],
            "description": domain_data["description"],
            "category_count": len(domain_data["categories"]),
            "timestamp": domain_data["timestamp"],
            "version": domain_data["version"],
        }

    def search_concepts(self, query: str, domain: str = None) -> List[Dict[str, Any]]:
        """
        Search for concepts across knowledge bases

        Args:
            query: Search query string
            domain: Optional domain to restrict search to

        Returns:
            List of matching concepts with metadata
        """
        results = []
        query_lower = query.lower()

        domains_to_search = [domain] if domain else self.get_domains()

        for current_domain in domains_to_search:
            if current_domain not in self.knowledge_bases:
                continue

            knowledge_base = self.knowledge_bases[current_domain]

            for category in knowledge_base.get("categories", []):
                category_id = category.get("id")
                category_name = category.get("name", {})

                for concept in category.get("concepts", []):
                    concept_id = concept.get("id")
                    concept_name = concept.get("name", {})
                    description = concept.get("description", {})

                    # Search in English names and descriptions
                    match_found = False

                    # Check English name
                    en_name = concept_name.get("en", "").lower()
                    if query_lower in en_name:
                        match_found = True

                    # Check English description
                    en_desc = description.get("en", "").lower()
                    if query_lower in en_desc:
                        match_found = True

                    # Check concept ID
                    if query_lower in concept_id.lower():
                        match_found = True

                    if match_found:
                        results.append(
                            {
                                "domain": current_domain,
                                "category": {"id": category_id, "name": category_name},
                                "concept": {
                                    "id": concept_id,
                                    "name": concept_name,
                                    "description": description,
                                    "fields": {
                                        key: value
                                        for key, value in concept.items()
                                        if key not in ["id", "name", "description"]
                                    },
                                },
                            }
                        )

        return results

    def get_concept(self, domain: str, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific concept by ID"""
        if domain not in self.knowledge_bases:
            return None

        knowledge_base = self.knowledge_bases[domain]

        for category in knowledge_base.get("categories", []):
            for concept in category.get("concepts", []):
                if concept.get("id") == concept_id:
                    return {
                        "domain": domain,
                        "category": {
                            "id": category.get("id"),
                            "name": category.get("name", {}),
                        },
                        "concept": concept,
                    }

        return None

    def get_category_concepts(
        self, domain: str, category_id: str
    ) -> List[Dict[str, Any]]:
        """Get all concepts in a specific category"""
        if domain not in self.knowledge_bases:
            return []

        knowledge_base = self.knowledge_bases[domain]

        for category in knowledge_base.get("categories", []):
            if category.get("id") == category_id:
                return category.get("concepts", [])

        return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded knowledge bases"""
        total_concepts = 0
        total_categories = 0

        for knowledge_base in self.knowledge_bases.values():
            total_categories += len(knowledge_base.get("categories", []))
            for category in knowledge_base.get("categories", []):
                total_concepts += len(category.get("concepts", []))

        return {
            "total_domains": len(self.knowledge_bases),
            "total_categories": total_categories,
            "total_concepts": total_concepts,
            "domains": self.get_domains(),
        }


# Global instance for easy access
_knowledge_service_instance = None


def get_knowledge_service() -> EngineeringKnowledgeService:
    """Get or create the global knowledge service instance"""
    global _knowledge_service_instance
    if _knowledge_service_instance is None:
        _knowledge_service_instance = EngineeringKnowledgeService()
    return _knowledge_service_instance
