# Emotion analysis models package
# 情感分析模型包
# Contains models for emotion analysis and awareness
# 包含情感分析和意识模型

from .unified_emotion_model import UnifiedEmotionModel

# Alias for compatibility
EmotionModel = UnifiedEmotionModel

__all__ = ['UnifiedEmotionModel', 'EmotionModel']
