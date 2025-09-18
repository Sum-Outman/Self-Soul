// 修复联合训练界面的多语言翻译
const fs = require('fs');
const path = require('path');

// 语言文件路径
const localesPath = path.join(__dirname, 'src', 'locales');
const languages = ['en', 'zh', 'de', 'ja', 'ru'];

  // 需要添加的完整翻译键和对应的翻译
  const missingTranslations = {
    en: {
      training: {
        title: "Training Control Panel",
        trainingMode: "Training Mode",
        individual: "Individual Training",
        joint: "Joint Training",
        selectModels: "Select Models for Training",
        recommendedCombinations: "Recommended Combinations",
        combinationValid: "Combination validated",
        dataset: "Dataset",
        parameters: "Parameters",
        epochs: "Epochs",
        batchSize: "Batch Size",
        learningRate: "Learning Rate",
        validationSplit: "Validation Split",
        trainingStrategy: "Training Strategy",
        startTraining: "Start Training",
        trainingInProgress: "Training in Progress",
        stopTraining: "Stop Training",
        progress: "Progress",
        requiresModels: "Requires models: {models}",
        missingDependencies: "Missing dependencies: {details}",
        uploadDataset: "Upload Dataset",
        dropoutRate: "Dropout Rate",
        weightDecay: "Weight Decay",
        momentum: "Momentum",
        optimizer: "Optimizer",
        knowledgeAssistOptions: "Knowledge Assist Options",
        domainKnowledge: "Domain Knowledge",
        commonSense: "Common Sense",
        proceduralKnowledge: "Procedural Knowledge",
        contextualLearning: "Contextual Learning",
        knowledgeIntensity: "Knowledge Intensity",
        logs: "Logs",
        epochCompleted: "Epoch completed - Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}%",
        trainingStarted: "Training started - Mode: {mode}, Models: {models}, Dataset: {dataset}",
        trainingCompleted: "Training completed successfully",
        trainingStopped: "Training stopped",
        websocketConnected: "WebSocket connection established",
        websocketDisconnected: "WebSocket connection disconnected",
        evaluation: "Evaluation",
        accuracy: "Accuracy",
        loss: "Loss",
        precision: "Precision",
        recall: "Recall",
        confusionMatrix: "Confusion Matrix",
        history: "History",
        date: "Date",
        models: "Models",
        duration: "Duration",
        actions: "Actions",
        view: "View",
        compare: "Compare",
        dependencies: "Model Dependencies"
      },
      models: {
        vision_image: "Image Vision Model",
        vision_video: "Video Vision Model",
        vision: "Vision Model",
        finance: "Finance Model",
        medical: "Medical Model",
        prediction: "Prediction Model",
        emotion: "Emotion Model",
        stereo_vision: "Stereo Vision Model"
      },
      errors: {
        selectAtLeastOneModel: "Please select at least one model for training",
        datasetUploadFailed: "Dataset upload failed: {error}",
        trainingStartFailed: "Training start failed: {error}",
        noActiveTraining: "No active training to stop",
        trainingStopFailed: "Training stop failed: {error}",
        websocketError: "WebSocket error: {error}",
        websocketConnectionFailed: "WebSocket connection failed: {error}"
      }
    },
    zh: {
      training: {
        title: "模型训练控制面板",
        trainingMode: "训练模式",
        individual: "单独训练",
        joint: "联合训练",
        selectModels: "选择参与训练的模型",
        recommendedCombinations: "推荐组合",
        combinationValid: "组合验证通过",
        dataset: "数据集",
        parameters: "参数",
        epochs: "训练轮次",
        batchSize: "批次大小",
        learningRate: "学习率",
        validationSplit: "验证集比例",
        trainingStrategy: "训练策略",
        startTraining: "开始训练",
        trainingInProgress: "训练进行中",
        stopTraining: "停止训练",
        progress: "进度",
        requiresModels: "需要模型: {models}",
        missingDependencies: "缺少依赖: {details}",
        uploadDataset: "上传数据集",
        dropoutRate: "丢弃率",
        weightDecay: "权重衰减",
        momentum: "动量",
        optimizer: "优化器",
        knowledgeAssistOptions: "知识库辅助选项",
        domainKnowledge: "领域知识",
        commonSense: "常识知识",
        proceduralKnowledge: "程序性知识",
        contextualLearning: "上下文学习",
        knowledgeIntensity: "知识强度",
        logs: "日志",
        epochCompleted: "轮次完成 - 轮次: {epoch}, 损失: {loss}, 准确率: {accuracy}%",
        trainingStarted: "训练已开始 - 模式: {mode}, 模型: {models}, 数据集: {dataset}",
        trainingCompleted: "训练完成成功",
        trainingStopped: "训练已停止",
        websocketConnected: "WebSocket连接已建立",
        websocketDisconnected: "WebSocket连接已断开",
        evaluation: "评估结果",
        accuracy: "准确率",
        loss: "损失值",
        precision: "精确率",
        recall: "召回率",
        confusionMatrix: "混淆矩阵",
        history: "历史记录",
        date: "日期",
        models: "模型",
        duration: "持续时间",
        actions: "操作",
        view: "查看",
        compare: "比较",
        dependencies: "模型依赖关系"
      },
      models: {
        vision_image: "图像视觉模型",
        vision_video: "视频视觉模型",
        vision: "视觉模型",
        finance: "金融模型",
        medical: "医疗模型",
        prediction: "预测模型",
        emotion: "情感模型",
        stereo_vision: "立体视觉模型"
      },
      errors: {
        selectAtLeastOneModel: "请至少选择一个模型进行训练",
        datasetUploadFailed: "数据集上传失败: {error}",
        trainingStartFailed: "训练启动失败: {error}",
        noActiveTraining: "没有活动的训练可以停止",
        trainingStopFailed: "训练停止失败: {error}",
        websocketError: "WebSocket错误: {error}",
        websocketConnectionFailed: "WebSocket连接失败: {error}"
      }
    },
  de: {
    training: {
      title: "Trainingssteuerung",
      trainingMode: "Trainingsmodus",
      individual: "Einzeltraining",
      joint: "Gemeinsames Training",
      selectModels: "Modelle für Training auswählen",
      recommendedCombinations: "Empfohlene Kombinationen",
      combinationValid: "Kombination validiert",
      dataset: "Datensatz",
      parameters: "Parameter",
      epochs: "Epochen",
      batchSize: "Batch-Größe",
      learningRate: "Lernrate",
      validationSplit: "Validierungsanteil",
      trainingStrategy: "Trainingsstrategie",
      startTraining: "Training starten",
      trainingInProgress: "Training läuft",
      stopTraining: "Training stoppen",
      progress: "Fortschritt",
      requiresModels: "Benötigt Modelle: {models}",
      missingDependencies: "Fehlende Abhängigkeiten: {details}",
      uploadDataset: "Datensatz hochladen",
      dropoutRate: "Dropout-Rate",
      weightDecay: "Gewichtsabnahme",
      momentum: "Impuls",
      optimizer: "Optimierer",
      knowledgeAssistOptions: "Wissensunterstützungsoptionen",
      domainKnowledge: "Domänenwissen",
      commonSense: "Allgemeinwissen",
      proceduralKnowledge: "Prozedurales Wissen",
      contextualLearning: "Kontextuelles Lernen",
      knowledgeIntensity: "Wissensintensität",
      logs: "Protokolle",
      epochCompleted: "Epoche abgeschlossen - Epoche: {epoch}, Verlust: {loss}, Genauigkeit: {accuracy}%",
      trainingStarted: "Training gestartet - Modus: {mode}, Modelle: {models}, Datensatz: {dataset}",
      trainingCompleted: "Training erfolgreich abgeschlossen",
      trainingStopped: "Training gestoppt",
      websocketConnected: "WebSocket-Verbindung hergestellt",
      websocketDisconnected: "WebSocket-Verbindung getrennt",
      evaluation: "Auswertung",
      accuracy: "Genauigkeit",
      loss: "Verlust",
      precision: "Präzision",
      recall: "Recall",
      confusionMatrix: "Konfusionsmatrix",
      history: "Verlauf",
      date: "Datum",
      models: "Modelle",
      duration: "Dauer",
      actions: "Aktionen",
      view: "Anzeigen",
      compare: "Vergleichen",
      dependencies: "Modellabhängigkeiten"
    },
    models: {
      vision_image: "Bild-Vision-Modell",
      vision_video: "Video-Vision-Modell",
      vision: "Vision-Modell",
      finance: "Finanzmodell",
      medical: "Medizinisches Modell",
      prediction: "Vorhersagemodell",
      emotion: "Emotionsmodell",
      stereo_vision: "Stereo-Vision-Modell"
    },
    errors: {
      selectAtLeastOneModel: "Bitte wählen Sie mindestens ein Modell für das Training aus",
      datasetUploadFailed: "Datensatz-Upload fehlgeschlagen: {error}",
      trainingStartFailed: "Trainingsstart fehlgeschlagen: {error}",
      noActiveTraining: "Kein aktives Training zum Stoppen",
      trainingStopFailed: "Trainingsstopp fehlgeschlagen: {error}",
      websocketError: "WebSocket-Fehler: {error}",
      websocketConnectionFailed: "WebSocket-Verbindung fehlgeschlagen: {error}"
    }
  },
  ja: {
    training: {
      title: "トレーニング制御パネル",
      trainingMode: "トレーニングモード",
      individual: "個別トレーニング",
      joint: "共同トレーニング",
      selectModels: "トレーニングするモデルを選択",
      recommendedCombinations: "推奨組み合わせ",
      combinationValid: "組み合わせが検証されました",
      dataset: "データセット",
      parameters: "パラメータ",
      epochs: "エポック数",
      batchSize: "バッチサイズ",
      learningRate: "学習率",
      validationSplit: "検証分割率",
      trainingStrategy: "トレーニング戦略",
      startTraining: "トレーニング開始",
      trainingInProgress: "トレーニング中",
      stopTraining: "トレーニング停止",
      progress: "進行状況",
      requiresModels: "必要なモデル: {models}",
      missingDependencies: "不足している依存関係: {details}",
      uploadDataset: "データセットをアップロード",
      dropoutRate: "ドロップアウト率",
      weightDecay: "重み減衰",
      momentum: "モーメンタム",
      optimizer: "オプティマイザ",
      knowledgeAssistOptions: "知識支援オプション",
      domainKnowledge: "ドメイン知識",
      commonSense: "常識",
      proceduralKnowledge: "手続き的知識",
      contextualLearning: "文脈学習",
      knowledgeIntensity: "知識強度",
      logs: "ログ",
      epochCompleted: "エポック完了 - エポック: {epoch}, 損失: {loss}, 精度: {accuracy}%",
      trainingStarted: "トレーニング開始 - モード: {mode}, モデル: {models}, データセット: {dataset}",
      trainingCompleted: "トレーニングが正常に完了しました",
      trainingStopped: "トレーニングが停止しました",
      websocketConnected: "WebSocket接続が確立されました",
      websocketDisconnected: "WebSocket接続が切断されました",
      evaluation: "評価",
      accuracy: "精度",
      loss: "損失",
      precision: "適合率",
      recall: "再現率",
      confusionMatrix: "混同行列",
      history: "履歴",
      date: "日付",
      models: "モデル",
      duration: "期間",
      actions: "操作",
      view: "表示",
      compare: "比較",
      dependencies: "モデル依存関係"
    },
    models: {
      vision_image: "画像視覚モデル",
      vision_video: "ビデオ視覚モデル",
      vision: "視覚モデル",
      finance: "金融モデル",
      medical: "医療モデル",
      prediction: "予測モデル",
      emotion: "感情モデル",
      stereo_vision: "ステレオ視覚モデル"
    },
    errors: {
      selectAtLeastOneModel: "トレーニングするモデルを少なくとも1つ選択してください",
      datasetUploadFailed: "データセットのアップロードに失敗しました: {error}",
      trainingStartFailed: "トレーニングの開始に失敗しました: {error}",
      noActiveTraining: "停止するアクティブなトレーニングがありません",
      trainingStopFailed: "トレーニングの停止に失敗しました: {error}",
      websocketError: "WebSocketエラー: {error}",
      websocketConnectionFailed: "WebSocket接続に失敗しました: {error}"
    }
  },
  ru: {
    training: {
      title: "Панель управления обучением",
      trainingMode: "Режим обучения",
      individual: "Индивидуальное обучение",
      joint: "Совместное обучение",
      selectModels: "Выбрать модели для обучения",
      recommendedCombinations: "Рекомендуемые комбинации",
      combinationValid: "Комбинация проверена",
      dataset: "Набор данных",
      parameters: "Параметры",
      epochs: "Эпохи",
      batchSize: "Размер пакета",
      learningRate: "Скорость обучения",
      validationSplit: "Доля валидации",
      trainingStrategy: "Стратегия обучения",
      startTraining: "Начать обучение",
      trainingInProgress: "Обучение в процессе",
      stopTraining: "Остановить обучение",
      progress: "Прогресс",
      requiresModels: "Требуются модели: {models}",
      missingDependencies: "Отсутствующие зависимости: {details}",
      uploadDataset: "Загрузить набор данных",
      dropoutRate: "Коэффициент отсева",
      weightDecay: "Затухание весов",
      momentum: "Импульс",
      optimizer: "Оптимизатор",
      knowledgeAssistOptions: "Опции помощи знаний",
      domainKnowledge: "Предметные знания",
      commonSense: "Здравый смысл",
      proceduralKnowledge: "Процедурные знания",
      contextualLearning: "Контекстное обучение",
      knowledgeIntensity: "Интенсивность знаний",
      logs: "Журналы",
      epochCompleted: "Эпоха завершена - Эпоха: {epoch}, Потери: {loss}, Точность: {accuracy}%",
      trainingStarted: "Обучение начато - Режим: {mode}, Модели: {models}, Набор данных: {dataset}",
      trainingCompleted: "Обучение успешно завершено",
      trainingStopped: "Обучение остановлено",
      websocketConnected: "WebSocket соединение установлено",
      websocketDisconnected: "WebSocket соединение разорвано",
      evaluation: "Оценка",
      accuracy: "Точность",
      loss: "Потери",
      precision: "Точность",
      recall: "Полнота",
      confusionMatrix: "Матрица ошибок",
      history: "История",
      date: "Дата",
      models: "Модели",
      duration: "Продолжительность",
      actions: "Действия",
      view: "Просмотр",
      compare: "Сравнить",
      dependencies: "Зависимости моделей"
    },
    models: {
      vision_image: "Модель обработки изображений",
      vision_video: "Модель обработки видео",
      vision: "Визуальная модель",
      finance: "Финансовая модель",
      medical: "Медицинская модель",
      prediction: "Модель прогнозирования",
      emotion: "Эмоциональная модель",
      stereo_vision: "Стереоскопическая модель"
    },
    errors: {
      selectAtLeastOneModel: "Пожалуйста, выберите хотя бы одну модель для обучения",
      datasetUploadFailed: "Ошибка загрузки набора данных: {error}",
      trainingStartFailed: "Ошибка запуска обучения: {error}",
      noActiveTraining: "Нет активного обучения для остановки",
      trainingStopFailed: "Ошибка остановки обучения: {error}",
      websocketError: "Ошибка WebSocket: {error}",
      websocketConnectionFailed: "Ошибка подключения WebSocket: {error}"
    }
  }
};

// 修复语言文件
function fixLanguageFiles() {
  console.log('🔧 修复联合训练界面的多语言翻译...\n');
  
  languages.forEach(lang => {
    console.log(`📝 修复 ${lang.toUpperCase()} 语言文件...`);
    
    try {
      const filePath = path.join(localesPath, `${lang}.json`);
      const content = fs.readFileSync(filePath, 'utf8');
      const translations = JSON.parse(content);
      
      // 添加缺失的翻译
      const langTranslations = missingTranslations[lang];
      
      // 添加training部分的翻译
      if (langTranslations.training) {
        if (!translations.training) {
          translations.training = {};
        }
        Object.assign(translations.training, langTranslations.training);
      }
      
      // 添加errors部分的翻译
      if (langTranslations.errors) {
        if (!translations.errors) {
          translations.errors = {};
        }
        Object.assign(translations.errors, langTranslations.errors);
      }
      
      // 添加models部分的翻译
      if (langTranslations.models) {
        if (!translations.models) {
          translations.models = {};
        }
        Object.assign(translations.models, langTranslations.models);
      }
      
      // 写回文件
      fs.writeFileSync(filePath, JSON.stringify(translations, null, 2));
      console.log(`✅ ${lang.toUpperCase()} - 翻译已添加`);
      
    } catch (error) {
      console.log(`❌ ${lang.toUpperCase()} - 修复失败: ${error.message}`);
    }
  });
  
  console.log('\n🎉 所有语言文件的翻译已修复完成！');
}

// 运行修复
fixLanguageFiles();
