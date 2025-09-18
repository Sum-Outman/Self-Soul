"""
Finance Model: 金融分析专业模型
"""

"""
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
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


"""
FinanceModel类 - 中文类描述
FinanceModel Class - English class description
"""
class FinanceModel:
    """金融分析专业模型，提供市场分析、投资建议等功能
       Financial analysis professional model providing market analysis and investment advice
    """
    
    
"""
__init__函数 - 中文函数描述
__init__ Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def __init__(self):
        # 支持的金融市场
        # Supported financial markets
        self.supported_markets = ['stock', 'forex', 'crypto', 'bond', 'commodity']
        
        # 金融指标计算方法
        # Financial indicator calculation methods
        self.financial_indicators = {
            'moving_average': self._calculate_moving_average,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger_bands': self._calculate_bollinger_bands
        }
        
        # 投资组合优化参数
        # Portfolio optimization parameters
        self.portfolio_params = {
            'risk_free_rate': 0.02,  # 无风险收益率
            'max_portfolio_size': 20,  # 最大投资组合大小
            'min_diversification': 5   # 最小多样化程度
        }
        
        # 支持的语言
        # Supported languages
        self.supported_languages = ['zh', 'en']
        
        # 多语言金融术语映射
        # Multilingual financial term mapping
        self.finance_terms_translation = {
            'zh': {
                'analysis': '分析',
                'recommendation': '建议',
                'risk': '风险',
                'return': '回报',
                'portfolio': '投资组合'
            },
            'en': {
                'analysis': 'analysis',
                'recommendation': 'recommendation',
                'risk': 'risk',
                'return': 'return',
                'portfolio': 'portfolio'
            }
        }
    
    
"""
_calculate_moving_average函数 - 中文函数描述
_calculate_moving_average Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _calculate_moving_average(self, data, window=20):
        """计算移动平均线
           Calculate moving average
        """
        if len(data) < window:
            return None
        
        # 简单移动平均线计算
        # Simple moving average calculation
        result = []
        for i in range(len(data) - window + 1):
            window_data = data[i:i+window]
            avg = sum(window_data) / window
            result.append(avg)
        
        return result
    
    
"""
_calculate_rsi函数 - 中文函数描述
_calculate_rsi Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _calculate_rsi(self, data, period=14):
        """计算相对强弱指标
           Calculate Relative Strength Index
        """
        if len(data) < period + 1:
            return None
        
        # 简化版RSI计算
        # Simplified RSI calculation
        deltas = np.diff(data)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        if len(gains) == 0:
            return 0
        if len(losses) == 0:
            return 100
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    
"""
_calculate_macd函数 - 中文函数描述
_calculate_macd Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD指标
           Calculate MACD indicator
        """
        # 简化版MACD计算
        # Simplified MACD calculation
        fast_ma = self._calculate_moving_average(data, fast_period)
        slow_ma = self._calculate_moving_average(data, slow_period)
        
        if not fast_ma or not slow_ma:
            return None, None
        
        # 确保两个均线长度相同
        # Ensure both moving averages have the same length
        min_length = min(len(fast_ma), len(slow_ma))
        fast_ma = fast_ma[-min_length:]
        slow_ma = slow_ma[-min_length:]
        
        # 计算MACD线
        # Calculate MACD line
        macd_line = [f - s for f, s in zip(fast_ma, slow_ma)]
        
        # 计算信号线
        # Calculate signal line
        signal_line = self._calculate_moving_average(macd_line, signal_period)
        
        return macd_line, signal_line
    
    
"""
_calculate_bollinger_bands函数 - 中文函数描述
_calculate_bollinger_bands Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _calculate_bollinger_bands(self, data, window=20, num_std=2):
        """计算布林带
           Calculate Bollinger Bands
        """
        if len(data) < window:
            return None, None, None
        
        # 计算中轨（移动平均线）
        # Calculate middle band (moving average)
        middle_band = []
        upper_band = []
        lower_band = []
        
        for i in range(len(data) - window + 1):
            window_data = data[i:i+window]
            avg = sum(window_data) / window
            std = np.std(window_data)
            
            middle_band.append(avg)
            upper_band.append(avg + num_std * std)
            lower_band.append(avg - num_std * std)
        
        return upper_band, middle_band, lower_band
    
    
"""
analyze_market函数 - 中文函数描述
analyze_market Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def analyze_market(self, market_type, symbol, historical_data=None, lang='zh'):
        """分析金融市场
           Analyze financial market
        
        Args:
            market_type (str): 市场类型
            symbol (str): 交易品种代码
            historical_data (list): 历史价格数据
            lang (str): 语言代码
        
        Returns:
            dict: 分析结果
        """
        if market_type not in self.supported_markets:
            return {"error": f"不支持的市场类型: {market_type}"}
        
        # 如果没有提供历史数据，生成模拟数据
        # If no historical data provided, generate sample data
        if not historical_data:
            historical_data = self._generate_sample_data(30)
        
        # 计算各种技术指标
        # Calculate various technical indicators
        ma20 = self._calculate_moving_average(historical_data, 20)
        rsi = self._calculate_rsi(historical_data)
        macd_line, signal_line = self._calculate_macd(historical_data)
        upper_band, middle_band, lower_band = self._calculate_bollinger_bands(historical_data)
        
        # 生成分析结果
        # Generate analysis results
        result = {
            'market_type': market_type,
            'symbol': symbol,
            self._translate('analysis', lang): {
                'current_price': historical_data[-1] if historical_data else 0,
                'technical_indicators': {
                    'moving_average_20': ma20[-1] if ma20 else None,
                    'rsi': rsi,
                    'macd': macd_line[-1] if macd_line else None,
                    'signal_line': signal_line[-1] if signal_line else None,
                    'bollinger_upper': upper_band[-1] if upper_band else None,
                    'bollinger_middle': middle_band[-1] if middle_band else None,
                    'bollinger_lower': lower_band[-1] if lower_band else None
                }
            }
        }
        
        # 生成投资建议
        # Generate investment recommendations
        recommendation = self._generate_recommendation(result, lang)
        result[self._translate('recommendation', lang)] = recommendation
        
        return result
    
    
"""
_generate_sample_data函数 - 中文函数描述
_generate_sample_data Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _generate_sample_data(self, days=30):
        """生成样本价格数据
           Generate sample price data
        """
        # 生成模拟的价格数据
        # Generate simulated price data
        base_price = 100.0
        volatility = 0.02  # 2%的波动率
        
        prices = [base_price]
        for _ in range(days - 1):
            # 随机游走模型生成价格
            # Random walk model for price generation
            price_change = np.random.normal(0, base_price * volatility)
            new_price = max(prices[-1] + price_change, 0.01)  # 确保价格不为负
            prices.append(new_price)
        
        return prices
    
    
"""
_generate_recommendation函数 - 中文函数描述
_generate_recommendation Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _generate_recommendation(self, analysis_result, lang='zh'):
        """基于分析结果生成投资建议
           Generate investment recommendation based on analysis
        """
        indicators = analysis_result[self._translate('analysis', lang)]['technical_indicators']
        current_price = analysis_result[self._translate('analysis', lang)]['current_price']
        
        # 简单的建议逻辑
        # Simple recommendation logic
        recommendation = {}
        
        # 基于RSI的建议
        # RSI-based recommendation
        if indicators['rsi'] is not None:
            if indicators['rsi'] > 70:
                recommendation['rsi_based'] = "超买，考虑卖出" if lang == 'zh' else "Overbought, consider selling"
            elif indicators['rsi'] < 30:
                recommendation['rsi_based'] = "超卖，考虑买入" if lang == 'zh' else "Oversold, consider buying"
            else:
                recommendation['rsi_based'] = "中性" if lang == 'zh' else "Neutral"
        
        # 基于MACD的建议
        # MACD-based recommendation
        if indicators['macd'] is not None and indicators['signal_line'] is not None:
            if indicators['macd'] > indicators['signal_line']:
                recommendation['macd_based'] = "看涨交叉，考虑买入" if lang == 'zh' else "Bullish crossover, consider buying"
            else:
                recommendation['macd_based'] = "看跌交叉，考虑卖出" if lang == 'zh' else "Bearish crossover, consider selling"
        
        # 基于布林带的建议
        # Bollinger Bands-based recommendation
        if (indicators['bollinger_upper'] is not None and 
            indicators['bollinger_lower'] is not None):
            if current_price > indicators['bollinger_upper']:
                recommendation['bollinger_based'] = "突破上轨，可能超买" if lang == 'zh' else "Above upper band, possibly overbought"
            elif current_price < indicators['bollinger_lower']:
                recommendation['bollinger_based'] = "跌破下轨，可能超卖" if lang == 'zh' else "Below lower band, possibly oversold"
            else:
                recommendation['bollinger_based'] = "在轨道内，震荡行情" if lang == 'zh' else "Within bands, ranging market"
        
        # 添加风险提示
        # Add risk disclaimer
        recommendation['risk_disclaimer'] = "以上建议仅供参考，不构成投资建议" if lang == 'zh' else "The above recommendations are for reference only and do not constitute investment advice"
        
        return recommendation
    
    
"""
optimize_portfolio函数 - 中文函数描述
optimize_portfolio Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def optimize_portfolio(self, assets, risk_preference='moderate', lang='zh'):
        """优化投资组合
           Optimize investment portfolio
        
        Args:
            assets (list): 资产列表
            risk_preference (str): 风险偏好 ('conservative', 'moderate', 'aggressive')
            lang (str): 语言代码
        
        Returns:
            dict: 优化后的投资组合
        """
        # 简化版投资组合优化
        # Simplified portfolio optimization
        # 实际实现中应使用现代投资组合理论
        # In actual implementation, modern portfolio theory should be used
        
        # 为每种资产分配权重（基于简化规则）
        # Allocate weights to each asset (based on simplified rules)
        weights = []
        total_assets = len(assets)
        
        # 根据风险偏好调整权重分布
        # Adjust weight distribution based on risk preference
        if risk_preference == 'conservative':
            # 保守策略：更均匀的分布
            # Conservative strategy: more even distribution
            weights = [1.0 / total_assets for _ in assets]
        elif risk_preference == 'aggressive':
            # 激进策略：集中在少数资产
            # Aggressive strategy: concentrated in few assets
            weights = [0.2 / (total_assets - 1) for _ in assets]
            weights[0] = 0.8  # 最大权重给第一个资产
        else:
            # 适中策略：适度集中
            # Moderate strategy: moderately concentrated
            weights = [0.15 / (total_assets - 1) for _ in assets]
            weights[0] = 0.85 - 0.15 * (total_assets - 2)  # 第一个资产权重较大
        
        # 确保权重和为1
        # Ensure weights sum to 1
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # 构建优化后的投资组合
        # Build optimized portfolio
        portfolio = {
            self._translate('portfolio', lang): [],
            'risk_preference': risk_preference
        }
        
        for i, asset in enumerate(assets):
            portfolio[self._translate('portfolio', lang)].append({
                'asset': asset,
                'weight': round(weights[i] * 100, 2)  # 转换为百分比
            })
        
        # 添加投资组合建议
        # Add portfolio advice
        portfolio_advice = self._get_portfolio_advice(risk_preference, lang)
        portfolio[self._translate('recommendation', lang)] = portfolio_advice
        
        return portfolio
    
    
"""
_get_portfolio_advice函数 - 中文函数描述
_get_portfolio_advice Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _get_portfolio_advice(self, risk_preference, lang='zh'):
        """获取投资组合建议
           Get portfolio advice
        """
        advice_map = {
            'conservative': {
                'zh': ["保持投资组合多元化", "增加低风险资产比例", "定期再平衡投资组合"],
                'en': ["Keep portfolio diversified", "Increase allocation to low-risk assets", "Regularly rebalance your portfolio"]
            },
            'moderate': {
                'zh': ["适度配置成长型资产", "关注资产间相关性", "定期检视投资表现"],
                'en': ["Moderately allocate to growth assets", "Pay attention to asset correlations", "Regularly review investment performance"]
            },
            'aggressive': {
                'zh': ["密切监控市场变化", "设定止损策略", "预留充足现金储备"],
                'en': ["Closely monitor market changes", "Set stop-loss strategies", "Maintain adequate cash reserves"]
            }
        }
        
        return advice_map.get(risk_preference, advice_map['moderate']).get(lang, [])
    
    
"""
_translate函数 - 中文函数描述
_translate Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def _translate(self, term, lang):
        """翻译金融术语
           Translate financial terms
        """
        if lang in self.finance_terms_translation and term in self.finance_terms_translation[lang]:
            return self.finance_terms_translation[lang][term]
        return term
    
    
"""
train函数 - 中文函数描述
train Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def train(self, training_data, callback=None):
        """训练金融模型
           Train the finance model
        
        Args:
            training_data: 训练数据
            callback: 进度回调函数
        
        Returns:
            dict: 训练结果
        """
        # 模拟训练过程
        # Simulate training process
        import time
        start_time = time.time()
        
        # 模拟训练进度
        # Simulate training progress
        for i in range(10):
            time.sleep(0.7)  # 模拟训练时间
            progress = (i + 1) * 10
            
            # 计算模拟指标
            # Calculate simulated metrics
            loss = 0.6 - (i * 0.05)
            accuracy = 55 + (i * 3.5)
            
            # 调用回调函数更新进度
            # Call callback function to update progress
            if callback:
                callback(progress, {'loss': loss, 'accuracy': accuracy})
        
        # 返回训练结果
        # Return training results
        return {
            'status': 'completed',
            'training_time': time.time() - start_time,
            'final_metrics': {
                'loss': 0.1,
                'accuracy': 90
            }
        }
    
    
"""
process函数 - 中文函数描述
process Function - English function description

Args:
    params: 参数描述 (Parameter description)
    
Returns:
    返回值描述 (Return value description)
"""
def process(self, input_data):
        """处理输入数据
           Process input data
        
        Args:
            input_data (dict): 输入数据，包含查询类型和参数
        
        Returns:
            dict: 处理结果
        """
        query_type = input_data.get('type', 'market_analysis')
        lang = input_data.get('lang', 'zh')
        
        if query_type == 'market_analysis':
            market_type = input_data.get('market_type', 'stock')
            symbol = input_data.get('symbol', 'AAPL')
            historical_data = input_data.get('historical_data')
            return self.analyze_market(market_type, symbol, historical_data, lang)
        elif query_type == 'portfolio_optimization':
            assets = input_data.get('assets', ['AAPL', 'MSFT', 'GOOGL'])
            risk_preference = input_data.get('risk_preference', 'moderate')
            return self.optimize_portfolio(assets, risk_preference, lang)
        else:
            return {"error": "不支持的查询类型"}