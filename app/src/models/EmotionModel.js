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

// Emotional Intelligence Core Model
class EmotionModel {
  constructor() {
    // Emotion states (0-100)
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
    
    // Emotion baselines (personality traits)
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
    
    // Emotion memory
    this.memory = []
    this.decayRate = 0.95 // Emotion decay rate
  }
  
  // Analyze text emotion
  analyzeText(text) {
    const positiveWords = ['happy', 'joyful', 'like', 'love', 'success', 'wonderful', 'satisfied']
    const negativeWords = ['sad', 'sorrow', 'hate', 'angry', 'failure', 'terrible', 'disappointed']
    
    let positiveCount = 0
    let negativeCount = 0
    
    // Simple word frequency analysis
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
    
    // Calculate emotion changes
    const total = positiveCount + negativeCount
    if (total > 0) {
      const joyChange = (positiveCount / total) * 20
      const sadnessChange = (negativeCount / total) * 15
      
      this.updateEmotion('joy', joyChange)
      this.updateEmotion('sadness', sadnessChange)
    }
    
    return this.getCurrentEmotion()
  }
  
  // Analyze voice emotion
  analyzeAudio(audioData) {
    // Simulate voice emotion analysis
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
  
  // Analyze visual emotion
  analyzeVisual(imageData) {
    // Simulate visual emotion analysis
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
  
  // Update emotion states
  updateEmotion(emotion, delta) {
    if (this.emotions[emotion] !== undefined) {
      // Apply changes, but keep within 0-100 range
      this.emotions[emotion] = Math.max(0, Math.min(100, this.emotions[emotion] + delta))
      
      // Record emotion events
      this.memory.push({
        emotion,
        delta,
        timestamp: Date.now()
      })
      
      // Apply emotion decay
      this.applyDecay()
    }
  }
  
  // Apply emotion decay (return to baseline)
  applyDecay() {
    for (const emotion in this.emotions) {
      const diff = this.baseline[emotion] - this.emotions[emotion]
      this.emotions[emotion] += diff * (1 - this.decayRate)
    }
  }
  
  // Get current dominant emotion
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
  
  // Get current emotion states
  getCurrentEmotion() {
    return {...this.emotions}
  }
  
  // Generate emotional response
  generateResponse(text) {
    const dominant = this.getDominantEmotion()
    let response = text
    
    // Modify response based on dominant emotion
    switch(dominant) {
      case 'joy':
        response = `😊 ${response} This makes me happy!`
        break
      case 'sadness':
        response = `😔 ${response} I feel a bit sad about this.`
        break
      case 'anger':
        response = `😠 ${response} This makes me angry!`
        break
      case 'surprise':
        response = `😲 ${response} This is surprising!`
        break
      case 'fear':
        response = `😨 ${response} This makes me a bit scared...`
        break
      case 'trust':
        response = `🤝 ${response} I believe we can solve this problem.`
        break
      case 'anticipation':
        response = `🤔 ${response} I'm looking forward to seeing the results.`
        break
      default:
        response = `🤖 ${response}`
    }
    
    return response
  }
  
  // Helper method - Calculate audio pitch (simulation)
  calculatePitch(audioData) {
    // Actual implementation should use audio analysis techniques like FFT
    return Math.random() * 0.5 + 0.3 // Return value between 0-1
  }
  
  // Helper method - Calculate audio volume (simulation)
  calculateVolume(audioData) {
    // Actual implementation should calculate RMS value
    return Math.random() * 0.5 + 0.4 // Return value between 0-1
  }
  
  // Helper method - Calculate image brightness (simulation)
  calculateBrightness(imageData) {
    // Actual implementation should analyze pixel brightness values
    return Math.random() * 0.5 + 0.4 // Return value between 0-1
  }
  
  // Helper method - Analyze image color distribution (simulation)
  analyzeColors(imageData) {
    // Actual implementation should analyze color histogram
    return {
      red: Math.random() * 0.3,
      green: Math.random() * 0.4,
      blue: Math.random() * 0.3
    }
  }
}

export default EmotionModel