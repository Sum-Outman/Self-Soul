<!--
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
  -->

<template>
  <div class="language-switcher">
    <select v-model="currentLang" @change="onLanguageChange">
      <option value="zh">{{ $t('language.chinese') }}</option>
      <option value="en">{{ $t('language.english') }}</option>
      <option value="de">{{ $t('language.german') }}</option>
      <option value="ja">{{ $t('language.japanese') }}</option>
      <option value="ru">{{ $t('language.russian') }}</option>
    </select>
  </div>
</template>

<script>
import { switchLanguage } from '@/i18n'

export default {
  name: 'LanguageSwitcher',
  data() {
    return {
      currentLang: this.$i18n.locale
    }
  },
  methods: {
    onLanguageChange() {
      switchLanguage(this.currentLang)
    }
  },
  mounted() {
    // 监听语言变化
    window.addEventListener('language-changed', (event) => {
      this.currentLang = event.detail.lang
    })
  }
}
</script>

<style scoped>
.language-switcher {
  margin: 0 10px;
}

select {
  padding: 5px 10px;
  border-radius: 4px;
  border: 1px solid #ccc;
  background-color: white;
  cursor: pointer;
}
</style>