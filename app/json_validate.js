const fs = require('fs');
try {
    const data = fs.readFileSync('src/locales/zh.json', 'utf8');
    JSON.parse(data);
    console.log('JSON is valid');
} catch (error) {
    console.error('JSON is invalid:', error.message);
    process.exit(1);
}