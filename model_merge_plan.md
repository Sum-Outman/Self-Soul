# 模型合并任务计划

- [ ] 分析空间模型重复功能并制定合并策略
- [ ] 合并 core/models/spatial/model.py 和 core/models/spatial/stereo_model.py
- [ ] 分析视觉模型重复功能并制定合并策略  
- [ ] 合并 core/models/vision/model.py 和 core/models/vision/image_model.py
- [ ] 删除重复的模型文件
- [ ] 更新模型注册表以反映合并后的模型
- [ ] 测试合并后的模型功能完整性
- [ ] 验证系统启动和模型加载正常

## 空间模型合并策略
- 保留 SpatialPerceptionModel 作为主类
- 整合 StereoVisionModel 的实时输入接口功能
- 合并物体检测和跟踪算法
- 统一API接口

## 视觉模型合并策略
- 保留 VisionModel 作为主类
- 整合 ImageVisionModel 的高级图像处理功能
- 合并外部API支持
- 统一情感生成功能
