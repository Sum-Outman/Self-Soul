// 提取SystemSettings.vue中使用的翻译键并检查完整性
const fs = require('fs');
const path = require('path');

// 从SystemSettings.vue中提取所有翻译键
function extractTranslationKeysFromFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const translationKeys = new Set();
    
    // 匹配 $t('key') 和 $t("key") 模式
    const regex = /\$t\(['"]([^'"]+)['"]\)/g;
    let match;
    
    while ((match = regex.exec(content)) !== null) {
      translationKeys.add(match[1]);
    }
    
    return Array.from(translationKeys);
  } catch (error) {
    console.error(`❌ 无法读取文件 ${filePath}:`, error.message);
    return [];
  }
}

// 检查翻译键在语言文件中的完整性
function checkTranslationCompleteness(translationKeys) {
  console.log('🔍 检查SystemSettings.vue中使用的翻译键完整性...\n');
  
  const localesPath = path.join(__dirname, 'src', 'locales');
  const languages = ['en', 'zh', 'de', 'ja', 'ru'];
  
  let allPassed = true;
  
  languages.forEach(lang => {
    console.log(`📖 检查 ${lang.toUpperCase()} 语言文件...`);
    
    try {
      const filePath = path.join(localesPath, `${lang}.json`);
      const content = fs.readFileSync(filePath, 'utf8');
      const translations = JSON.parse(content);
      
      let missingKeys = [];
      let langPassed = true;
      
      // 检查所有必需的键
      translationKeys.forEach(key => {
        const keys = key.split('.');
        let current = translations;
        
        for (const k of keys) {
          if (current && current[k] !== undefined) {
            current = current[k];
          } else {
            missingKeys.push(key);
            langPassed = false;
            break;
          }
        }
      });
      
      if (langPassed) {
        console.log(`✅ ${lang.toUpperCase()} - 所有翻译键都存在`);
      } else {
        console.log(`❌ ${lang.toUpperCase()} - 缺少 ${missingKeys.length} 个翻译键:`);
        missingKeys.forEach(key => console.log(`   - ${key}`));
        allPassed = false;
      }
      
    } catch (error) {
      console.log(`❌ ${lang.toUpperCase()} - 无法读取或解析语言文件: ${error.message}`);
      allPassed = false;
    }
    
    console.log('');
  });
  
  return allPassed;
}

// 主函数
function main() {
  const systemSettingsPath = path.join(__dirname, 'src', 'views', 'SystemSettings.vue');
  
  console.log('🔍 从SystemSettings.vue中提取翻译键...');
  const translationKeys = extractTranslationKeysFromFile(systemSettingsPath);
  
  console.log('📋 找到的翻译键:');
  translationKeys.forEach(key => console.log(`   - ${key}`));
  console.log('');
  
  // 检查完整性
  const allComplete = checkTranslationCompleteness(translationKeys);
  
  if (allComplete) {
    console.log('🎉 SystemSettings.vue中使用的所有翻译键在所有语言文件中都存在！');
  } else {
    console.log('⚠️  部分语言文件缺少翻译键，需要修复后才能完全支持多语言。');
  }
  
  return allComplete;
}

// 运行检查
main();
