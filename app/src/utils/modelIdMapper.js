/**
 * 模型ID映射工具 - 解决前端字母ID和后端字符串ID不一致的问题
 * Model ID Mapping Utility - Resolves inconsistency between frontend letter IDs and backend string IDs
 * 
 * 前端使用字母ID (A-K) 用于界面显示和用户交互
 * Frontend uses letter IDs (A-K) for UI display and user interaction
 * 
 * 后端使用字符串ID ('manager', 'language', etc.) 用于API调用和模型管理
 * Backend uses string IDs ('manager', 'language', etc.) for API calls and model management
 */

// 字母ID到字符串ID的映射
export const letterToIdMap = {
  'A': 'manager',           // 管理模型 | Manager Model
  'B': 'language',          // 大语言模型 | Language Model
  'C': 'audio',             // 音频处理模型 | Audio Processing Model
  'D': 'vision_image',      // 图片视觉处理模型 | Image Vision Model
  'E': 'vision_video',      // 视频流视觉处理模型 | Video Vision Model
  'F': 'spatial',           // 双目空间定位感知模型 | Spatial Perception Model
  'G': 'sensor',            // 传感器感知模型 | Sensor Perception Model
  'H': 'computer',          // 计算机控制模型 | Computer Control Model
  'I': 'motion',            // 运动和执行器控制模型 | Motion and Actuator Model
  'J': 'knowledge',         // 知识库专家模型 | Knowledge Expert Model
  'K': 'programming'        // 编程模型 | Programming Model
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
 * 获取模型显示名称（使用字母ID）
 * Get model display name (using letter ID)
 * @param {string} id - 字母ID或字符串ID
 * @param {object} t - i18n翻译函数
 * @returns {string} 显示名称
 */
export function getModelDisplayName(id, t) {
  const letterId = idToLetterMap[id] || id;
  return t(`models.${letterId}`);
}

/**
 * 获取模型详细描述
 * Get model detailed description
 * @param {string} id - 字母ID或字符串ID
 * @param {object} t - i18n翻译函数
 * @returns {string} 详细描述
 */
export function getModelDescription(id, t) {
  const letterId = idToLetterMap[id] || id;
  return t(`models.description.${letterId}`);
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
