#!/usr/bin/env python
"""
测试 Gemini API 是否正常工作
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


def test_with_timeout(timeout_seconds=60):
    """使用信号超时机制进行测试"""
    # 设置超时处理器
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        test_api()
        signal.alarm(0)  # 取消超时计时器
        print("✅ 测试成功完成，API正常工作")
        return True
    except TimeoutError as e:
        print(f"❌ 测试失败: {e}")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时计时器
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        return False


def test_with_thread_timeout(timeout_seconds=60):
    """使用线程超时机制进行测试（适用于Windows等不支持信号的平台）"""
    result = {"success": None, "error": None}
    
    def worker():
        try:
            test_api()
            result["success"] = True
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
        print("✅ 测试成功完成，API正常工作")
        return True
    else:
        print("❌ 测试失败: 未知错误")
        return False


def test_api():
    """测试Gemini API是否正常工作"""
    print("🔍 开始测试 Gemini API...")
    
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
        
    # 创建简单提示模板
    system_template = "你是一个简单的测试助手，请用简短的回答回复问题。"
    human_template = "请分析以下文本的主题是什么: {text}"
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    
    # 准备一个简单的文本样例
    test_text = "人工智能在自动驾驶领域的应用越来越广泛。最近的研究表明，通过使用大型语言模型可以更好地理解和预测道路上的行人行为。"
    
    print(f"📝 测试文本: {test_text[:30]}...")
    print("🔄 调用API中...")
    
    # 添加延迟以确保不会有速率限制问题
    api_manager.smart_delay()
    
    # 调用API
    start_time = time.time()
    try:
        response = llm.invoke(prompt_template.format(text=test_text))
        elapsed = time.time() - start_time
        
        print(f"⏱️ API响应时间: {elapsed:.2f}秒")
        print(f"📄 API响应内容: {response.content[:100]}...")
        
        # 尝试保存结果为JSON格式
        with open("api_test_result.json", "w") as f:
            json.dump({"response": response.content, "elapsed_seconds": elapsed}, f, ensure_ascii=False)
        
        print("✅ API测试结果已保存到 api_test_result.json")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ API调用失败 (经过 {elapsed:.2f}秒): {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    print("🚀 Gemini API 测试工具")
    print("=" * 50)
    
    timeout_sec = 60
    if len(sys.argv) > 1:
        try:
            timeout_sec = int(sys.argv[1])
        except ValueError:
            pass
    
    print(f"⏱️ 设置超时时间: {timeout_sec} 秒")
    
    # 根据操作系统选择不同的超时处理方式
    if hasattr(signal, 'SIGALRM'):  # Unix/Mac
        test_with_timeout(timeout_sec)
    else:  # Windows
        test_with_thread_timeout(timeout_sec)
