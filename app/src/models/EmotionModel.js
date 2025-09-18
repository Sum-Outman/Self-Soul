/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// 情感智能核心模型
class EmotionModel {
  constructor() {
    // 情感状态 (0-100)
    this.emotions = {
      joy: 50,
      sadness: 10,
      anger: 5,
      surprise: 15,
      fear: 5,
      trust: 60,
      anticipation: 40,
      disgust: 3
    }
    
    // 情感基线 (个性特征)
    this.baseline = {
      joy: 60,
      sadness: 15,
      anger: 10,
      surprise: 20,
      fear: 8,
      trust: 70,
      anticipation: 50,
      disgust: 5
    }
    
    // 情感记忆
    this.memory = []
    this.decayRate = 0.95 // 情感衰减率
  }
  
  // 分析文本情感
  analyzeText(text) {
    const positiveWords = ['高兴', '快乐', '喜欢', '爱', '成功', '美好', '满意']
    const negativeWords = ['悲伤', '难过', '恨', '生气', '失败', '糟糕', '失望']
    
    let positiveCount = 0
    let negativeCount = 0
    
    // 简单词频分析
    positiveWords.forEach(word => {
      const regex = new RegExp(word, 'g')
      const matches = text.match(regex)
      if (matches) positiveCount += matches.length
    })
    
    negativeWords.forEach(word => {
      const regex = new RegExp(word, 'g')
      const matches = text.match(regex)
      if (matches) negativeCount += matches.length
    })
    
    // 计算情感变化
    const total = positiveCount + negativeCount
    if (total > 0) {
      const joyChange = (positiveCount / total) * 20
      const sadnessChange = (negativeCount / total) * 15
      
      this.updateEmotion('joy', joyChange)
      this.updateEmotion('sadness', sadnessChange)
    }
    
    return this.getCurrentEmotion()
  }
  
  // 分析语音情感
  analyzeAudio(audioData) {
    // 模拟语音情感分析
    const pitch = this.calculatePitch(audioData)
    const volume = this.calculateVolume(audioData)
    
    if (pitch > 0.7) {
      this.updateEmotion('surprise', 15)
      this.updateEmotion('joy', 10)
    } else if (pitch < 0.3) {
      this.updateEmotion('sadness', 15)
      this.updateEmotion('fear', 10)
    }
    
    if (volume > 0.8) {
      this.updateEmotion('anger', 20)
    }
    
    return this.getCurrentEmotion()
  }
  
  // 分析视觉情感
  analyzeVisual(imageData) {
    // 模拟视觉情感分析
    const brightness = this.calculateBrightness(imageData)
    const colorDistribution = this.analyzeColors(imageData)
    
    if (brightness > 0.7) {
      this.updateEmotion('joy', 15)
    } else if (brightness < 0.3) {
      this.updateEmotion('sadness', 15)
    }
    
    if (colorDistribution.red > 0.4) {
      this.updateEmotion('anger', 10)
    } else if (colorDistribution.blue > 0.4) {
      this.updateEmotion('trust', 15)
    }
    
    return this.getCurrentEmotion()
  }
  
  // 更新情感状态
  updateEmotion(emotion, delta) {
    if (this.emotions[emotion] !== undefined) {
      // 应用变化，但不超过0-100范围
      this.emotions[emotion] = Math.max(0, Math.min(100, this.emotions[emotion] + delta))
      
      // 记录情感事件
      this.memory.push({
        emotion,
        delta,
        timestamp: Date.now()
      })
      
      // 应用情感衰减
      this.applyDecay()
    }
  }
  
  // 应用情感衰减（回归基线）
  applyDecay() {
    for (const emotion in this.emotions) {
      const diff = this.baseline[emotion] - this.emotions[emotion]
      this.emotions[emotion] += diff * (1 - this.decayRate)
    }
  }
  
  // 获取当前主导情感
  getDominantEmotion() {
    let maxEmotion = 'joy'
    let maxValue = 0
    
    for (const emotion in this.emotions) {
      if (this.emotions[emotion] > maxValue) {
        maxValue = this.emotions[emotion]
        maxEmotion = emotion
      }
    }
    
    return maxEmotion
  }
  
  // 获取当前情感状态
  getCurrentEmotion() {
    return {...this.emotions}
  }
  
  // 生成情感化响应
  generateResponse(text) {
    const dominant = this.getDominantEmotion()
    let response = text
    
    // 根据主导情感修饰响应
    switch(dominant) {
      case 'joy':
        response = `😊 ${response} 这真让人开心！`
        break
      case 'sadness':
        response = `😔 ${response} 我对此感到有些难过。`
        break
      case 'anger':
        response = `😠 ${response} 这让我感到生气！`
        break
      case 'surprise':
        response = `😲 ${response} 这真令人惊讶！`
        break
      case 'fear':
        response = `😨 ${response} 这让我有点害怕...`
        break
      case 'trust':
        response = `🤝 ${response} 我相信我们可以解决这个问题。`
        break
      case 'anticipation':
        response = `🤔 ${response} 我期待着看到结果。`
        break
      default:
        response = `🤖 ${response}`
    }
    
    return response
  }
  
  // 辅助方法 - 计算音频音高 (模拟)
  calculatePitch(audioData) {
    // 实际实现应使用FFT等音频分析技术
    return Math.random() * 0.5 + 0.3 // 返回0-1之间的值
  }
  
  // 辅助方法 - 计算音频音量 (模拟)
  calculateVolume(audioData) {
    // 实际实现应计算RMS值
    return Math.random() * 0.5 + 0.4 // 返回0-1之间的值
  }
  
  // 辅助方法 - 计算图像亮度 (模拟)
  calculateBrightness(imageData) {
    // 实际实现应分析像素亮度值
    return Math.random() * 0.5 + 0.4 // 返回0-1之间的值
  }
  
  // 辅助方法 - 分析图像颜色分布 (模拟)
  analyzeColors(imageData) {
    // 实际实现应分析颜色直方图
    return {
      red: Math.random() * 0.3,
      green: Math.random() * 0.4,
      blue: Math.random() * 0.3
    }
  }
}

export default EmotionModel