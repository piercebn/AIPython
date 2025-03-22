import os
import openai
from openai import OpenAI

class DeepSeekChat:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
        self.messages = []
    
    def stream_chat(self):
        try:
            stream = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=self.messages,
                temperature=0.3,
                stream=True,
            )
            
            full_response = {"reasoning": "", "content": ""}
            
            # 实时输出标记
            has_reasoning = False
            has_content = False
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # 处理推理内容
                if getattr(delta, "reasoning_content", None):
                    reasoning_part = delta.reasoning_content
                    full_response["reasoning"] += reasoning_part
                    if not has_reasoning:
                        print("\n[推理过程]")
                        has_reasoning = True
                    print(reasoning_part, end="", flush=True)
                
                # 处理最终回答（实时流式输出）
                if getattr(delta, "content", None):
                    content_part = delta.content
                    full_response["content"] += content_part
                    if not has_content:
                        if has_reasoning:
                            print("\n\n[最终回答]")
                        else:
                            print("\n[回答]")
                        has_content = True
                    print(content_part, end="", flush=True)
            
            # 更新对话历史
            if full_response["content"]:
                self.messages.append({"role": "assistant", "content": full_response["content"]})
            
            print()  # 确保最后换行

        except openai.APIError as e:
            print(f"\nAPI错误: {e.status_code} - {e.message}")
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

    def run(self):
        print("DeepSeek 流式对话系统 (输入 exit 退出)")
        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("\n对话结束")
                    break
                
                self.messages.append({"role": "user", "content": user_input})
                self.stream_chat()
                
            except KeyboardInterrupt:
                print("\n对话终止")
                break

if __name__ == "__main__":
    DeepSeekChat().run()
