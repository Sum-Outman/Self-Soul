from core.advanced_emotion_analysis import AdvancedEmotionAnalyzer

# 测试情感分析器初始化
try:
    print("Testing AdvancedEmotionAnalyzer initialization...")
    analyzer = AdvancedEmotionAnalyzer(from_scratch=True)
    print("✓ AdvancedEmotionAnalyzer initialized successfully")
    
    # 测试基本情感分析功能
    print("Testing basic emotion analysis...")
    test_text = "I am happy today"
    emotion_result = analyzer.analyze_text_emotion(test_text)
    print(f"✓ Emotion analysis result: {emotion_result}")
    
    # 测试多模态情感融合
    print("Testing multimodal emotion fusion...")
    text_emotion = {'happy': 0.8, 'sad': 0.1, 'neutral': 0.1}
    audio_emotion = {'happy': 0.6, 'sad': 0.3, 'neutral': 0.1}
    visual_emotion = {'happy': 0.7, 'sad': 0.2, 'neutral': 0.1}
    fused = analyzer.fuse_multimodal_emotions(text_emotion, audio_emotion, visual_emotion)
    print(f"✓ Multimodal fusion result: {fused}")
    
    print("✓ All emotion analyzer tests passed!")
    
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
