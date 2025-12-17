/**
 * Model ID Mapping Utility - Resolves inconsistency between frontend letter IDs and backend string IDs
 * 
 * Frontend uses letter IDs (A-K) for UI display and user interaction
 * Backend uses string IDs ('manager', 'language', etc.) for API calls and model management
 */

// Mapping from letter IDs to string IDs
export const letterToIdMap = {
  'A': 'manager',           // Manager Model
  'B': 'language',          // Language Model
  'C': 'audio',             // Audio Processing Model
  'D': 'vision_image',      // Image Vision Model
  'E': 'vision_video',      // Video Vision Model
  'F': 'spatial',           // Spatial Model
  'G': 'sensor',            // Sensor Model
  'H': 'computer',          // Computer Model
  'I': 'motion',            // Motion Model
  'J': 'knowledge',         // Knowledge Expert Model
  'K': 'programming',       // Programming Model
  'L': 'planning',          // Planning Model
  'M': 'autonomous',        // Autonomous Model
  'N': 'emotion',           // Emotion Model
  'O': 'spatial',           // Spatial Model (duplicate for compatibility)
  'P': 'vision_image',      // Computer Vision Model (using existing vision_image)
  'Q': 'sensor',            // Sensor Model (duplicate for compatibility)
  'R': 'motion',            // Motion Model (duplicate for compatibility)
  'S': 'prediction',        // Prediction Model
  'T': 'collaboration',     // Collaboration Model
  'U': 'optimization',      // Optimization Model
  'V': 'finance',           // Finance Model
  'W': 'medical',           // Medical Model
  'X': 'value_alignment'    // Value Alignment Model
};

// 字符串ID到字母ID的映射
export const idToLetterMap = {
  'manager': 'A',
  'language': 'B',
  'audio': 'C',
  'vision_image': 'D',
  'vision_video': 'E',
  'spatial': 'F',
  'sensor': 'G',
  'computer': 'H',
  'motion': 'I',
  'knowledge': 'J',
  'programming': 'K',
  'planning': 'L',
  'autonomous': 'M',
  'emotion': 'N',
  'prediction': 'S',
  'collaboration': 'T',
  'optimization': 'U',
  'finance': 'V',
  'medical': 'W',
  'value_alignment': 'X'
};

// 所有字母ID列表
export const letterIds = Object.keys(letterToIdMap);

// 所有字符串ID列表
export const stringIds = Object.keys(idToLetterMap);

/**
 * 将字母ID转换为字符串ID
 * Convert letter ID to string ID
 * @param {string} letterId - 字母ID (A-K)
 * @returns {string} 对应的字符串ID
 */
export function letterToId(letterId) {
  return letterToIdMap[letterId] || letterId;
}

/**
 * 将字符串ID转换为字母ID
 * Convert string ID to letter ID
 * @param {string} stringId - 字符串ID
 * @returns {string} 对应的字母ID
 */
export function idToLetter(stringId) {
  return idToLetterMap[stringId] || stringId;
}

/**
 * 批量转换字母ID数组为字符串ID数组
 * Batch convert array of letter IDs to string IDs
 * @param {string[]} letterIds - 字母ID数组
 * @returns {string[]} 字符串ID数组
 */
export function lettersToIds(letterIds) {
  return letterIds.map(letterToId);
}

/**
 * 批量转换字符串ID数组为字母ID数组
 * Batch convert array of string IDs to letter IDs
 * @param {string[]} stringIds - 字符串ID数组
 * @returns {string[]} 字母ID数组
 */
export function idsToLetters(stringIds) {
  return stringIds.map(idToLetter);
}

/**
 * 验证字母ID是否有效
 * Validate if letter ID is valid
 * @param {string} letterId - 字母ID
 * @returns {boolean} 是否有效
 */
export function isValidLetterId(letterId) {
  return letterId in letterToIdMap;
}

/**
 * 验证字符串ID是否有效
 * Validate if string ID is valid
 * @param {string} stringId - 字符串ID
 * @returns {boolean} 是否有效
 */
export function isValidStringId(stringId) {
  return stringId in idToLetterMap;
}

/**
 * 获取所有核心模型的字母ID（A-K）
 * Get all core model letter IDs (A-K)
 * @returns {string[]} 字母ID数组
 */
export function getAllCoreLetterIds() {
  return letterIds;
}

/**
 * 获取所有核心模型的字符串ID
 * Get all core model string IDs
 * @returns {string[]} 字符串ID数组
 */
export function getAllCoreStringIds() {
  return stringIds;
}

/**
 * Get model display name (using letter ID)
 * @param {string} id - Letter ID or string ID
 * @returns {string} Display name
 */
export function getModelDisplayName(id) {
  const letterId = idToLetterMap[id] || id;
  const displayNames = {
    'A': 'Unified Manager Model',
    'B': 'Unified Language Model',
    'C': 'Unified Audio Model',
    'D': 'Unified Image Vision Model',
    'E': 'Unified Video Vision Model',
    'F': 'Unified Spatial Model',
    'G': 'Unified Sensor Model',
    'H': 'Unified Computer Model',
    'I': 'Unified Motion Model',
    'J': 'Unified Knowledge Model',
    'K': 'Unified Programming Model',
    'L': 'Unified Planning Model',
    'M': 'Unified Autonomous Model',
    'N': 'Unified Emotion Model',
    'O': 'Spatial Model',
    'P': 'Computer Vision Model',
    'Q': 'Sensor Model',
    'R': 'Motion Model',
    'S': 'Unified Prediction Model',
    'T': 'Unified Collaboration Model',
    'U': 'Unified Optimization Model',
    'V': 'Unified Finance Model',
    'W': 'Unified Medical Model',
    'X': 'Unified Value Alignment Model'
  };
  return displayNames[letterId] || letterId;
}

/**
 * Get model detailed description
 * @param {string} id - Letter ID or string ID
 * @returns {string} Detailed description
 */
export function getModelDescription(id) {
  const letterId = idToLetterMap[id] || id;
  const descriptions = {
    'A': 'Core management model responsible for coordinating all other models',
    'B': 'Advanced language processing and understanding model',
    'C': 'Audio processing and voice recognition model',
    'D': 'Image vision processing and analysis model',
    'E': 'Video stream processing and analysis model',
    'F': 'Spatial awareness and positioning model',
    'G': 'Sensor data processing and perception model',
    'H': 'Computer control and interface model',
    'I': 'Motion control and actuator management model',
    'J': 'Knowledge base and expert system model',
    'K': 'Programming and code generation model',
    'L': 'Planning and decision-making model for task execution',
    'M': 'Autonomous operation and self-governance model',
    'N': 'Emotion recognition and response model',
    'O': 'Advanced spatial mapping and navigation model',
    'P': 'Computer vision and object recognition model',
    'Q': 'Sensor data fusion and analysis model',
    'R': 'Motion planning and execution model',
    'S': 'Predictive analysis and forecasting model',
    'T': 'Model collaboration and coordination model for joint task execution',
    'U': 'Model optimization and performance enhancement model',
    'V': 'Financial analysis and decision-making model',
    'W': 'Medical data analysis and healthcare model',
    'X': 'Value alignment and ethical decision-making model'
  };
  return descriptions[letterId] || 'Unknown model';
}

export default {
  letterToIdMap,
  idToLetterMap,
  letterIds,
  stringIds,
  letterToId,
  idToLetter,
  lettersToIds,
  idsToLetters,
  isValidLetterId,
  isValidStringId,
  getAllCoreLetterIds,
  getAllCoreStringIds,
  getModelDisplayName,
  getModelDescription
};
