// 提取SystemSettings.vue中使用的翻译键并检查完整性
const fs = require('fs');
const path = require('path');

// Extract all translation keys from SystemSettings.vue
function extractTranslationKeysFromFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const translationKeys = new Set();
    
    // Match $t('key') and $t("key") patterns
    const regex = /\$t\(['"]([^'"]+)['"]\)/g;
    let match;
    
    while ((match = regex.exec(content)) !== null) {
      translationKeys.add(match[1]);
    }
    
    return Array.from(translationKeys);
  } catch (error) {
    console.error(`❌ Unable to read file ${filePath}:`, error.message);
    return [];
  }
}

// Check translation keys completeness in language files
function checkTranslationCompleteness(translationKeys) {
    console.log('🔍 Checking translation keys completeness in SystemSettings.vue...\n');
    
    const localesPath = path.join(__dirname, 'src', 'locales');
    const languages = ['en']; // Only check English since interface is now English-only
    
    let allPassed = true;
    
    languages.forEach(lang => {
      console.log(`📖 Checking ${lang.toUpperCase()} language file...`);
    
    try {
      const filePath = path.join(localesPath, `${lang}.json`);
      const content = fs.readFileSync(filePath, 'utf8');
      const translations = JSON.parse(content);
      
      let missingKeys = [];
      let langPassed = true;
      
      // Check all required keys
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
        console.log(`✅ ${lang.toUpperCase()} - All translation keys exist`);
      } else {
        console.log(`❌ ${lang.toUpperCase()} - Missing ${missingKeys.length} translation keys:`);
        missingKeys.forEach(key => console.log(`   - ${key}`));
        allPassed = false;
      }
      
    } catch (error) {
      console.log(`❌ ${lang.toUpperCase()} - Unable to read or parse language file: ${error.message}`);
      allPassed = false;
    }
    
    console.log('');
  });
  
  return allPassed;
}

// Main function
function main() {
  const systemSettingsPath = path.join(__dirname, 'src', 'views', 'SystemSettings.vue');
  
  console.log('🔍 Extracting translation keys from SystemSettings.vue...');
  const translationKeys = extractTranslationKeysFromFile(systemSettingsPath);
  
  console.log('📋 Found translation keys:');
  translationKeys.forEach(key => console.log(`   - ${key}`));
  console.log('');
  
  // Check completeness
  const allComplete = checkTranslationCompleteness(translationKeys);
  
  if (allComplete) {
    console.log('🎉 All translation keys used in SystemSettings.vue exist in the English language file!');
  } else {
    console.log('⚠️  Some translation keys are missing in the English language file, please fix.');
  }
  
  return allComplete;
}

// Run check
main();
