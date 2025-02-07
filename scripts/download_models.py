import os
from transformers import AutoModel, AutoTokenizer

def download_models():
    """下载并保存模型到本地"""
    print("开始下载向量化模型...")
    
    # 设置模型保存目录
    model_path = "models/codebert-base"
    os.makedirs(model_path, exist_ok=True)
    
    # 下载模型和分词器
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # 保存到本地
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    download_models() 