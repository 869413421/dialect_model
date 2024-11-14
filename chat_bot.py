import os

# 指定模型下载缓存路径，不需要请注释
os.environ["HF_HOME"] = '/root/autodl-tmp/huggingface'
os.environ["TRANSFORMERS_CACHE"] = '/root/autodl-tmp/huggingface'
import torch
import streamlit as st
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## Qwen2.5 信宜话大模型")
    # 创建一个滑块，用于选择最大长度，范围在 0 到 8192 之间，默认值为 512（Qwen2.5 支持 128K 上下文，并能生成最多 8K tokens）
    max_length = st.slider("max_length", 0, 8192, 2500, step=1)

# 创建一个标题和一个副标题
st.title("💬 Qwen2.5 信宜话大模型")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

# 定义模型路径
mode_name_or_path = "Qwen/Qwen2.5-7B-Instruct"


# 定义一个函数，用于获取模型和 tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 确保 pad_token 设置正确

    # 从预训练的模型中获取模型，并设置模型参数
    model = AutoModelForCausalLM.from_pretrained(
        mode_name_or_path,
        torch_dtype=torch.float16,  # 修改为 float16 以确保更好的兼容性
        device_map="auto"  # 自动映射到可用的设备
    )

    # 确保 LoRA 权重路径正确
    lora_path = '/root/autodl-tmp/project/dialect_model/output/Qwen2.5_instruct_lora/checkpoint-2403/'
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA 权重路径不存在: {lora_path}")

    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, model_id=lora_path, load_in_fp16=True)  # 可尝试加载 8bit 权重

    return tokenizer, model


# 加载 Qwen2.5 的 model 和 tokenizer
tokenizer, model = get_model()

# 如果 session_state 中没有 "messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "我系识讲信宜话葛智能助手阿信。你可以同我讲信宜话，亦可以同我讲普通话。不过我训练数据矛几好，效果矛好矛叼我。"}]

# 遍历 session_state 中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 将用户输入添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    print(st.session_state.messages)
    # 将对话输入模型，获得返回
    inputs = tokenizer.apply_chat_template(
        st.session_state.messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    gen_kwargs = {
        "max_length": max_length,  # 调整为你期望的长度
        "do_sample": True,  # 启用采样以获得更多样的输出
        "top_k": 50,  # 增大top_k以提高多样性
        "temperature": 0.9  # 可调节生成的多样性，0.9相对较为保守，1.0更自由
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)  # 直接编码输入
    # generated_ids = model.generate(input_ids, max_new_tokens=max_length)  # 生成回答
    #
    # # 解码模型的输出并处理
    # response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 将模型的输出添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)

    # 打印 session_state 调试信息（可选）
    # print(st.session_state)
