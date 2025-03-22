import gradio as gr
import os

# 初始化对话历史，用于存储用户和模型的对话记录
chat_history = []

def deepseek_response(user_input, chat_history):
    """
    调用DeepSeek模型生成回复
    :param user_input: 用户输入文本
    :param chat_history: 对话历史
    :return: 模型生成的回复（生成器）
    """
    # 导入OpenAI库，用于调用DeepSeek API
    from openai import OpenAI
    
    # 从环境变量获取API密钥，确保安全性
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
    
    # 初始化OpenAI客户端，配置API密钥和基础URL
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    # 将对话历史转换为OpenAI API所需的格式
    history_openai_format = []
    for human, ai in chat_history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": ai}) 
    history_openai_format.append({"role": "user", "content": user_input})
    
    try:
        # 调用DeepSeek API，启用流式响应
        response = client.chat.completions.create(
            model='deepseek-chat',  # 使用的模型名称
            messages=history_openai_format,  # 对话历史
            temperature=1.0,  # 控制生成文本的随机性
            stream=True  # 启用流式输出
        )
        
        # 逐块返回响应内容，实现流式输出
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        # 处理API调用错误，返回友好的错误信息
        print(f"API调用出错: {str(e)}")
        yield "抱歉,服务出现了一些问题,请稍后再试。"

# 创建Gradio界面
with gr.Blocks() as demo:
    # 创建聊天窗口，用于显示对话历史
    chatbot = gr.Chatbot()
    
    # 创建输入框，用于用户输入消息
    msg = gr.Textbox()
    
    # 创建清空按钮，用于重置对话历史
    clear = gr.Button("Clear")
    
    def respond(message, chat_history):
        """
        处理用户输入并生成模型回复
        :param message: 用户输入文本
        :param chat_history: 当前对话历史
        :return: 更新后的输入框和对话历史
        """
        # 立即显示用户输入，初始回复为空
        chat_history.append((message, ""))
        yield "", chat_history
        
        # 显示等待提示，告知用户模型正在生成回复
        chat_history[-1] = (message, "正在思考...")
        yield "", chat_history
        
        # 逐块接收流式响应，实时更新对话历史
        full_response = ""
        for chunk in deepseek_response(message, chat_history[:-1]):  # 排除等待提示
            if full_response == "":
                # 移除等待提示，准备显示模型回复
                chat_history[-1] = (message, "")
            full_response += chunk
            chat_history[-1] = (message, full_response)
            yield "", chat_history
        
        # 将完整对话添加到历史记录
        chat_history[-1] = (message, full_response)
        yield "", chat_history
    
    # 绑定输入框的提交事件到respond函数
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # 绑定清空按钮的点击事件，重置对话历史
    clear.click(lambda: [], None, chatbot, queue=False)

# 启动Gradio应用
demo.launch()
