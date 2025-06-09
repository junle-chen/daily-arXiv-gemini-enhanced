#!/usr/bin/env python
"""
测试 Gemini API 是否可以正确处理论文过滤任务
"""
import os
import sys
import json
import time
import signal
import threading
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import api_manager


class TimeoutError(Exception):
    """超时错误"""
    pass


def timeout_handler(signum, frame):
    """处理超时信号"""
    raise TimeoutError("API调用超时")


def test_paper_filtering(timeout_seconds=120):
    """使用超时机制测试论文过滤API功能"""
    # 设置超时处理器
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        success = run_paper_filtering_test()
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 取消超时计时器
        return success
    except TimeoutError as e:
        print(f"❌ 测试失败: {e}")
        return False
    except Exception as e:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 取消超时计时器
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        return False


def run_paper_filtering_test():
    """测试论文过滤功能"""
    print("🔍 开始测试论文过滤功能...")
    
    # 配置环境
    api_manager.setup_environment()
    
    # 创建模型实例
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    print(f"🤖 使用模型: {model_name}")
    
    try:
        llm = ChatGoogleGenerativeAI(model=model_name)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise
        
    # 创建与filter_papers.py相同的提示模板
    system_template = """你是一个学术论文过滤器。你的任务是判断一篇论文是否与轨迹预测（trajectory prediction）和大语言模型（Large Language Models）相关。
    
    论文内容可能涉及以下主题：
    1. 轨迹预测：行人轨迹预测、车辆轨迹预测、移动物体路径规划、动作预测等
    2. 大模型：大型语言模型（LLMs）、基础模型（foundation models）、大规模AI模型等
    
    请仔细分析论文标题和摘要，判断其与上述主题的相关性。如果论文同时涉及轨迹预测和大模型，或者将这两个领域结合起来的工作，相关性会更高。"""

    human_template = """分析以下论文，判断其与轨迹预测和大型语言模型的相关性：
    
    标题：{title}
    摘要：{summary}
    
    请输出一个JSON格式的回复，包含以下字段：
    1. relevance_score: 0.0-1.0之间的数字，表示相关性程度（1.0表示高度相关）
    2. explanation: 简短解释为什么这篇论文相关或不相关
    3. keywords: 提取的与轨迹预测或大模型相关的关键词列表
    
    只返回JSON对象，不要有其他文本。"""
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    
    # 准备一个相关的论文样例
    test_paper = {
        "title": "Trajectory Prediction for Autonomous Vehicles using Large Language Models",
        "summary": "In this paper, we propose a novel method for predicting the trajectories of vehicles and pedestrians by leveraging large language models (LLMs). Our approach combines traditional motion prediction techniques with the contextual understanding capabilities of foundation models. We demonstrate that by incorporating LLMs into the prediction pipeline, we can achieve more accurate trajectory forecasting in complex urban environments. Experimental results show a 15% improvement over state-of-the-art methods on standard benchmarks."
    }
    
    print(f"📝 测试论文: {test_paper['title']}")
    print("🔄 调用API中...")
    
    # 添加延迟以确保不会有速率限制问题
    api_manager.smart_delay()
    
    # 调用API
    start_time = time.time()
    try:
        print("📤 正在发送API请求...")
        response = llm.invoke(prompt_template.format(title=test_paper["title"], summary=test_paper["summary"]))
        elapsed = time.time() - start_time
        
        print(f"⏱️ API响应时间: {elapsed:.2f}秒")
        print(f"📄 原始API响应: {response.content}")
        
        # 尝试解析JSON响应
        try:
            content = response.content
            if isinstance(content, str):
                content = content.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(content)
            print(f"✅ 成功解析JSON响应")
            print(f"📊 相关性得分: {result.get('relevance_score', 'N/A')}")
            print(f"📝 解释: {result.get('explanation', 'N/A')}")
            print(f"🔑 关键词: {result.get('keywords', [])}")
            
            # 保存结果
            with open("paper_filtering_test_result.json", "w", encoding="utf-8") as f:
                json.dump({
                    "paper": test_paper,
                    "response": result,
                    "elapsed_seconds": elapsed
                }, f, ensure_ascii=False, indent=2)
            
            print("✅ 测试结果已保存到 paper_filtering_test_result.json")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ 无法解析JSON响应: {e}")
            print(f"原始响应: {response.content}")
            raise
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ API调用失败 (经过 {elapsed:.2f}秒): {type(e).__name__}: {e}")
        raise


def run_test_with_thread_timeout(timeout_seconds=120):
    """使用线程超时机制进行测试（适用于Windows等不支持信号的平台）"""
    result = {"success": None, "error": None}
    
    def worker():
        try:
            success = run_paper_filtering_test()
            result["success"] = success
        except Exception as e:
            result["error"] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print("❌ 测试失败: API调用超时")
        return False
    elif result["error"]:
        print(f"❌ 测试失败: {type(result['error']).__name__}: {result['error']}")
        return False
    elif result["success"]:
        return True
    else:
        print("❌ 测试失败: 未知错误")
        return False


if __name__ == "__main__":
    print("🚀 论文过滤 API 测试工具")
    print("=" * 50)
    
    timeout_sec = 120
    if len(sys.argv) > 1:
        try:
            timeout_sec = int(sys.argv[1])
        except ValueError:
            pass
    
    print(f"⏱️ 设置超时时间: {timeout_sec} 秒")
    
    # 根据操作系统选择不同的超时处理方式
    if hasattr(signal, 'SIGALRM'):  # Unix/Mac
        test_paper_filtering(timeout_sec)
    else:  # Windows
        run_test_with_thread_timeout(timeout_sec)
