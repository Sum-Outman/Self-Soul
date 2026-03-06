// Model ID mapping utility for consistent model identification

// Model ID to name mapping
export const modelIdToName = {
  'manager': 'Manager Model',
  'language': 'Language Model',
  'knowledge': 'Knowledge Model',
  'vision': 'Vision Model',
  'audio': 'Audio Model',
  'autonomous': 'Autonomous Model',
  'programming': 'Programming Model',
  'planning': 'Planning Model',
  'emotion': 'Emotion Model',
  'spatial': 'Spatial Model',
  'computer_vision': 'Computer Vision Model',
  'sensor': 'Sensor Model',
  'motion': 'Motion Model',
  'prediction': 'Prediction Model',
  'advanced_reasoning': 'Advanced Reasoning Model',
  'multi_model_collaboration': 'Multi-Model Collaboration Model',
  'data_fusion': 'Data Fusion Model',
  'creative_problem_solving': 'Creative Problem Solving Model',
  'meta_cognition': 'Meta Cognition Model',
  'value_alignment': 'Value Alignment Model',
  'vision_image': 'Vision Image Model',
  'vision_video': 'Vision Video Model',
  'finance': 'Finance Model',
  'medical': 'Medical Model',
  'collaboration': 'Collaboration Model',
  'optimization': 'Optimization Model',
  'computer': 'Computer Model',
  'mathematics': 'Mathematics Model',
  'translation': 'Translation Model'
};

// Model ID to port mapping
export const modelIdToPort = {
  'manager': 8001,
  'language': 8002,
  'knowledge': 8003,
  'vision': 8004,
  'audio': 8005,
  'autonomous': 8006,
  'programming': 8007,
  'planning': 8008,
  'emotion': 8009,
  'spatial': 8010,
  'computer_vision': 8011,
  'sensor': 8012,
  'motion': 8013,
  'prediction': 8014,
  'advanced_reasoning': 8015,
  'multi_model_collaboration': 8016,
  'creative_problem_solving': 8017,
  'meta_cognition': 8018,
  'value_alignment': 8019,
  'vision_image': 8020,
  'vision_video': 8021,
  'finance': 8022,
  'medical': 8023,
  'collaboration': 8024,
  'optimization': 8025,
  'computer': 8026,
  'mathematics': 8027,
  'data_fusion': 8028,
  'translation': 8029
};

// Port to model ID mapping
export const portToModelId = {
  8001: 'manager',
  8002: 'language',
  8003: 'knowledge',
  8004: 'vision',
  8005: 'audio',
  8006: 'autonomous',
  8007: 'programming',
  8008: 'planning',
  8009: 'emotion',
  8010: 'spatial',
  8011: 'computer_vision',
  8012: 'sensor',
  8013: 'motion',
  8014: 'prediction',
  8015: 'advanced_reasoning',
  8016: 'multi_model_collaboration',
  8017: 'creative_problem_solving',
  8018: 'meta_cognition',
  8019: 'value_alignment',
  8020: 'vision_image',
  8021: 'vision_video',
  8022: 'finance',
  8023: 'medical',
  8024: 'collaboration',
  8025: 'optimization',
  8026: 'computer',
  8027: 'mathematics',
  8028: 'data_fusion',
  8029: 'translation'
};

// Get model name by ID
export const getModelNameById = (modelId) => {
  return modelIdToName[modelId] || modelId;
};

// Get model port by ID
export const getModelPortById = (modelId) => {
  return modelIdToPort[modelId] || null;
};

// Get model ID by port
export const getModelIdByPort = (port) => {
  return portToModelId[port] || null;
};

// Validate model ID
export const isValidModelId = (modelId) => {
  return modelIdToName.hasOwnProperty(modelId);
};

// Validate model port
export const isValidModelPort = (port) => {
  return portToModelId.hasOwnProperty(port);
};

// Get all model IDs
export const getAllModelIds = () => {
  return Object.keys(modelIdToName);
};

// Get all model ports
export const getAllModelPorts = () => {
  return Object.values(modelIdToPort);
};

// Get model info by ID
export const getModelInfoById = (modelId) => {
  return {
    id: modelId,
    name: getModelNameById(modelId),
    port: getModelPortById(modelId),
    valid: isValidModelId(modelId)
  };
};

// Get model info by port
export const getModelInfoByPort = (port) => {
  const modelId = getModelIdByPort(port);
  return {
    id: modelId,
    name: modelId ? getModelNameById(modelId) : null,
    port: port,
    valid: isValidModelPort(port)
  };
};

// Letter mapping utilities
 export const letterIds = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

export const modelIdToLetterMap = {
   'manager': 'A',
   'language': 'B',
   'knowledge': 'C',
   'vision': 'D',
   'audio': 'E',
   'autonomous': 'F',
   'programming': 'G',
   'planning': 'H',
   'emotion': 'I',
   'spatial': 'J',
   'computer_vision': 'K',
   'sensor': 'L',
   'motion': 'M',
   'prediction': 'N',
   'advanced_reasoning': 'O',
   'data_fusion': 'P',
   'creative_problem_solving': 'Q',
   'meta_cognition': 'R',
   'value_alignment': 'S',
   'vision_image': 'T',
   'vision_video': 'U',
   'finance': 'V',
   'medical': 'W',
   'collaboration': 'X',
   'optimization': 'Y',
   'computer': 'Z',
   'mathematics': 'AA'
 };

export const letterToIdMap = {};
Object.keys(modelIdToLetterMap).forEach(modelId => {
  const letter = modelIdToLetterMap[modelId];
  letterToIdMap[letter] = modelId;
});

// Convert model ID to letter
export const idToLetter = (modelId) => {
  return modelIdToLetterMap[modelId] || modelId;
};

// Convert letter to model ID
export const letterToId = (letter) => {
  return letterToIdMap[letter] || letter;
};

// Convert array of letters to model IDs
export const lettersToIds = (letters) => {
  return letters.map(letter => letterToId(letter));
};

// Convert array of model IDs to letters
export const idsToLetters = (modelIds) => {
  return modelIds.map(modelId => idToLetter(modelId));
};

// Get model display name (alias for getModelNameById)
export const getModelDisplayName = getModelNameById;

// Model descriptions
const modelDescriptions = {
  'manager': 'Coordinates all other models',
  'language': 'Natural language processing',
  'knowledge': 'Knowledge base management',
  'vision': 'Visual perception and analysis',
  'audio': 'Audio processing and speech recognition',
  'autonomous': 'Autonomous decision making',
  'programming': 'Code generation and analysis',
  'planning': 'Task planning and scheduling',
  'emotion': 'Emotion recognition and response',
  'spatial': 'Spatial awareness and navigation',
  'computer_vision': 'Advanced computer vision tasks',
  'sensor': 'Sensor data processing and fusion',
  'motion': 'Motion control and coordination',
  'prediction': 'Predictive analytics and forecasting',
  'advanced_reasoning': 'Complex reasoning tasks',
  'data_fusion': 'Multi-source data integration',
  'creative_problem_solving': 'Creative solution generation',
  'meta_cognition': 'Self-awareness and reflection',
  'value_alignment': 'Ethical decision making',
  'vision_image': 'Image-specific vision tasks',
  'vision_video': 'Video-specific vision tasks',
  'finance': 'Financial analysis and planning',
  'medical': 'Medical knowledge and analysis',
  'collaboration': 'Team collaboration and coordination',
  'optimization': 'Resource and process optimization',
  'computer': 'Computer system knowledge and control',
  'mathematics': 'Mathematical reasoning and problem solving'
};

// Get model description by ID
export const getModelDescription = (modelId) => {
  return modelDescriptions[modelId] || 'No description available';
};

export default {
  modelIdToName,
  modelIdToPort,
  portToModelId,
  getModelNameById,
  getModelPortById,
  getModelIdByPort,
  isValidModelId,
  isValidModelPort,
  getAllModelIds,
  getAllModelPorts,
  getModelInfoById,
  getModelInfoByPort,
  idToLetter,
  letterToId,
  modelIdToLetterMap,
  letterToIdMap,
  lettersToIds,
  idsToLetters,
  letterIds,
  getModelDisplayName,
  getModelDescription
};
