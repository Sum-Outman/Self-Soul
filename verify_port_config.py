#!/usr/bin/env python3
"""
验证所有端口配置是否与README保持一致
"""

import json
import os
import re
from pathlib import Path

def read_readme_ports():
    """从README.md读取端口配置表"""
    readme_path = Path(__file__).parent / "README.md"
    ports = {}
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找端口配置表
    table_pattern = r'\| Port \| Service \| Description \|\s*\|\s*---+\s*\|\s*---+\s*\|\s*---+\s*\|\s*(.*?)\s*\|\s*5175 \| Frontend Dashboard'
    match = re.search(table_pattern, content, re.DOTALL)
    if match:
        table_content = match.group(1)
        # 解析每一行
        lines = table_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('|'):
                # 移除前后的|并分割
                parts = [p.strip() for p in line.strip('|').split('|')]
                if len(parts) >= 3 and parts[0].isdigit():
                    port = int(parts[0])
                    service = parts[1].strip()
                    description = parts[2].strip()
                    # 提取模型ID（从服务名中提取小写，移除" Model"等）
                    model_id = service.lower().replace(' model', '').replace(' ', '_')
                    ports[model_id] = port
                    print(f"README: {model_id:30} -> {port}")
    
    return ports

def read_config_ports():
    """从model_services_config.json读取端口配置"""
    config_path = Path(__file__).parent / "config" / "model_services_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_ports = config.get("model_ports", {})
    for model_id, port in sorted(model_ports.items()):
        print(f"CONFIG: {model_id:30} -> {port}")
    
    return model_ports

def compare_ports(readme_ports, config_ports):
    """比较README端口和配置端口"""
    print("\n" + "="*60)
    print("端口配置一致性检查")
    print("="*60)
    
    all_good = True
    
    # 检查README中的每个模型是否在配置中
    for model_id, readme_port in readme_ports.items():
        if model_id in config_ports:
            config_port = config_ports[model_id]
            if config_port == readme_port:
                print(f"✅ {model_id:30}: README={readme_port}, CONFIG={config_port} (匹配)")
            else:
                print(f"❌ {model_id:30}: README={readme_port}, CONFIG={config_port} (不匹配)")
                all_good = False
        else:
            print(f"⚠️  {model_id:30}: 在README中但不在配置中")
            all_good = False
    
    # 检查配置中的每个模型是否在README中
    for model_id, config_port in config_ports.items():
        if model_id not in readme_ports:
            print(f"⚠️  {model_id:30}: 在配置中但不在README中 (端口: {config_port})")
            # 这不一定是错误，可能是额外模型
    
    # 检查端口范围
    min_port = min(config_ports.values()) if config_ports else 0
    max_port = max(config_ports.values()) if config_ports else 0
    print(f"\n端口范围: {min_port} - {max_port}")
    
    # 检查端口冲突
    port_to_models = {}
    for model_id, port in config_ports.items():
        port_to_models.setdefault(port, []).append(model_id)
    
    conflicts = {port: models for port, models in port_to_models.items() if len(models) > 1}
    if conflicts:
        print(f"\n❌ 端口冲突:")
        for port, models in conflicts.items():
            print(f"   端口 {port}: {', '.join(models)}")
        all_good = False
    else:
        print(f"\n✅ 无端口冲突")
    
    return all_good

def main():
    print("读取README端口配置...")
    readme_ports = read_readme_ports()
    
    print("\n读取配置文件端口配置...")
    config_ports = read_config_ports()
    
    success = compare_ports(readme_ports, config_ports)
    
    if success:
        print("\n✅ 所有端口配置与README一致")
        return 0
    else:
        print("\n❌ 发现端口配置不一致")
        return 1

if __name__ == "__main__":
    exit(main())