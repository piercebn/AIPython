import os
import openai
from openai import OpenAI

def deepseek_chat():
    # 配置环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",  # DeepSeek API 端点
    )

    # 初始化对话历史
    messages = []
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("\n对话结束")
                break
            
            # 添加用户消息
            messages.append({"role": "user", "content": user_input})
            
            # 创建API请求
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            
            # 解析响应
            assistant_message = response.choices[0].message
            reasoning_content = assistant_message.reasoning_content  # 直接获取推理内容
            final_content = assistant_message.content  # 获取最终内容
            
            # 打印结果
            print("\n[推理过程]")
            print(reasoning_content)
            print("\n[最终回答]")
            print(final_content)
            
            # 添加助手消息到历史（仅保留回答部分，不保留推理部分）
            messages.append({
                "role": "assistant",
                "content": final_content
            })
            
        except openai.APIError as e:
            print(f"API错误: {e.status_code} - {e.message}")
        except AttributeError as e:
            print(f"响应格式错误: {e}")
        except KeyboardInterrupt:
            print("\n对话终止")
            break
        except Exception as e:
            print(f"意外错误: {str(e)}")

if __name__ == "__main__":
    print("DeepSeek 多轮对话系统 (输入 exit 退出)")
    print("正在初始化...")
    deepseek_chat()
