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
  'F': 'spatial',           // Spatial Perception Model
  'G': 'sensor',            // Sensor Perception Model
  'H': 'computer',          // Computer Control Model
  'I': 'motion',            // Motion and Actuator Model
  'J': 'knowledge',         // Knowledge Expert Model
  'K': 'programming'        // Programming Model
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
  'programming': 'K'
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
    'A': 'Manager Model',
    'B': 'Language Model',
    'C': 'Audio Model',
    'D': 'Image Vision Model',
    'E': 'Video Vision Model',
    'F': 'Spatial Perception Model',
    'G': 'Sensor Perception Model',
    'H': 'Computer Control Model',
    'I': 'Motion and Actuator Model',
    'J': 'Knowledge Expert Model',
    'K': 'Programming Model'
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
    'K': 'Programming and code generation model'
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
