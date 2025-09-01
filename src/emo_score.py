import json
import os
import time
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging

# 禁用 tokenizers 并行
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    def __init__(self, news_items):
        self.news_items = news_items

    def __len__(self):
        return len(self.news_items)

    def __getitem__(self, idx):
        item = self.news_items[idx]
        # 提取 title 和 description 的第一个片段
        title = item["title_segments"][0] if item["title_segments"] else ""
        description_segments = item["description_segments"] if item["description_segments"] else []
        return {
            "title": title,
            "description_segments": description_segments,
            "date": item["date"]
        }

def custom_collate_fn(batch):
    # 自定义 collate，处理变长字段
    titles = [item["title"] for item in batch]
    description_segments = [item["description_segments"] for item in batch]
    dates = [item["date"] for item in batch]
    return {
        "title": titles,
        "description_segments": description_segments,
        "date": dates
    }

def main(raw_data_path, output_path, model_path, symbol):
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 设置参数
    batch_size = 128  # 单卡批量大小
    max_length = 510  # 留 2 token 给 [CLS] 和 [SEP]
    num_workers = 4   # 数据加载线程

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"初始显存: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    # 加载 FinBERT 模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()
        logger.info(f"FinBERT 模型加载成功，显存: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 加载 Polygon 新闻数据
    try:
        with open(raw_data_path, "r", encoding="utf-8") as f:
            news_data = json.load(f)
        news_items = news_data.get("results", [])
        if not news_items:
            logger.error("新闻数据为空")
            return
        logger.info(f"加载 {len(news_items)} 条新闻数据")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return

    # 转换为 DataFrame
    df_news = pd.DataFrame(news_items)

    # 转换 published_utc 时间
    def safe_convert_datetime(dt):
        if pd.isna(dt) or not dt:
            return None
        try:
            return datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y/%m/%d")
        except ValueError:
            logger.warning(f"无效时间格式跳过: {dt}")
            return None

    df_news["date"] = df_news["published_utc"].apply(safe_convert_datetime)
    df_news = df_news.dropna(subset=["date"])

    # 分段处理长文本
    def segment_text(text, tokenizer, max_length=510):
        if not text or pd.isna(text):
            return []
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_length:
            return [text]
        segments = []
        words = text.split()
        current_segment = []
        current_length = 0
        for word in words:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if current_length + len(word_tokens) > max_length:
                segments.append(" ".join(current_segment))
                current_segment = [word]
                current_length = len(word_tokens)
            else:
                current_segment.append(word)
                current_length += len(word_tokens)
        if current_segment:
            segments.append(" ".join(current_segment))
        if len(tokens) > 512:
            logger.info(f"分段处理超长文本，token 数: {len(tokens)}")
        return segments

    # 预处理分段
    df_news["title_segments"] = df_news["title"].apply(lambda x: [x] if x else [])
    df_news["description_segments"] = df_news["description"].apply(
        lambda x: segment_text(x, tokenizer, max_length) if x else []
    )

    # 数据集和加载器
    dataset = NewsDataset(df_news.to_dict("records"))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 情感分析函数
    def batch_sentiment_analysis(texts, model, tokenizer, device):
        if not texts or all(not t for t in texts):
            return [0.0] * len(texts), [0.0] * len(texts)
        try:
            inputs = tokenizer(
                texts, return_tensors="pt", max_length=512, truncation=True, padding=True
            ).to(device)
            with torch.no_grad(), autocast():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                scores = probs[:, 1] - probs[:, 0]  # positive - negative
                confidences = torch.max(probs, dim=-1).values
            return scores.cpu().numpy(), confidences.cpu().numpy()
        except Exception as e:
            logger.error(f"批量情感分析错误: {e}")
            return [0.0] * len(texts), [0.0] * len(texts)

    # 注意力机制
    def compute_attention_weights(title_score, description_score, title_confidence, description_confidence):
        if title_confidence == 0.0 and description_confidence == 0.0:
            return 0.0, 0.0, 0.0
        if title_confidence == 0.0:
            return description_score, 0.0, 1.0
        if description_confidence == 0.0:
            return title_score, 1.0, 0.0
        total_confidence = title_confidence + description_confidence
        title_weight = title_confidence / total_confidence
        description_weight = description_confidence / total_confidence
        score = title_weight * title_score + description_weight * description_score
        return score, title_weight, description_weight

    # 处理批量数据
    results = []
    start_time = time.time()
    for batch in dataloader:
        batch_results = []
        titles = batch["title"]
        description_segments = batch["description_segments"]
        for i in range(len(titles)):
            title = titles[i]
            desc_segments = description_segments[i]
            
            title_score = 0.0
            title_confidence = 0.0
            description_score = 0.0
            description_confidence = 0.0
            
            if title:
                title_scores, title_confidences = batch_sentiment_analysis(
                    [title], model, tokenizer, device
                )
                title_score = title_scores[0]
                title_confidence = title_confidences[0]
            
            if desc_segments:
                desc_scores, desc_confidences = batch_sentiment_analysis(
                    desc_segments, model, tokenizer, device
                )
                description_score = np.mean(desc_scores) if desc_scores.size else 0.0
                description_confidence = np.mean(desc_confidences) if desc_confidences.size else 0.0
            
            score, title_weight, desc_weight = compute_attention_weights(
                title_score, description_score, title_confidence, description_confidence
            )
            batch_results.append((score, title_weight, desc_weight))
        
        results.extend(batch_results)
        if torch.cuda.is_available():
            logger.info(f"处理批次，显存: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    # 收集结果
    df_news["sentiment_score"] = 0.0
    df_news["title_weight"] = 0.0
    df_news["description_weight"] = 0.0
    for i, (score, tw, dw) in enumerate(results):
        df_news.at[i, "sentiment_score"] = score
        df_news.at[i, "title_weight"] = tw
        df_news.at[i, "description_weight"] = dw

    # 按日期聚合
    daily_scores = df_news.groupby("date")["sentiment_score"].mean().reset_index()
    daily_scores = daily_scores.sort_values("date")

    # 保存结果
    daily_scores.to_csv(output_path, index=False)
    logger.info(f"情感打分完成，已保存到 {output_path}")
    logger.info(f"总计 {len(df_news)} 条新闻，{len(daily_scores)} 个日期数据")
    logger.info(f"处理时间: {time.time() - start_time:.2f} 秒")
    logger.info(f"平均 title_weight: {df_news['title_weight'].mean():.3f}, 标准差: {df_news['title_weight'].std():.3f}")
    logger.info(f"平均 description_weight: {df_news['description_weight'].mean():.3f}, 标准差: {df_news['description_weight'].std():.3f}")

if __name__ == "__main__":
    # M7 公司股票代码和路径
    companies = [
        {"symbol": "GOOGL", "raw_data_path": "/mnt/data/wqk/AI+Fin/data/raw/googl_news_polygon.json", "output_path": "/mnt/data/wqk/AI+Fin/data/processed/GOOGL_scored_news_attention.csv"},
        {"symbol": "AMZN", "raw_data_path": "/mnt/data/wqk/AI+Fin/data/raw/amzn_news_polygon.json", "output_path": "/mnt/data/wqk/AI+Fin/data/processed/AMZN_scored_news_attention.csv"},
        {"symbol": "AAPL", "raw_data_path": "/mnt/data/wqk/AI+Fin/data/raw/apple_news_polygon.json", "output_path": "/mnt/data/wqk/AI+Fin/data/processed/AAPL_scored_news_attention.csv"},
        {"symbol": "META", "raw_data_path": "/mnt/data/wqk/AI+Fin/data/raw/meta_news_polygon.json", "output_path": "/mnt/data/wqk/AI+Fin/data/processed/META_scored_news_attention.csv"},
        {"symbol": "MSFT", "raw_data_path": "/mnt/data/wqk/AI+Fin/data/raw/msft_news_polygon.json", "output_path": "/mnt/data/wqk/AI+Fin/data/processed/MSFT_scored_news_attention.csv"},
        {"symbol": "NVDA", "raw_data_path": "/mnt/data/wqk/AI+Fin/data/raw/nvda_news_polygon.json", "output_path": "/mnt/data/wqk/AI+Fin/data/processed/NVDA_scored_news_attention.csv"},
        {"symbol": "TSLA", "raw_data_path": "/mnt/data/wqk/AI+Fin/data/raw/tsla_news_polygon.json", "output_path": "/mnt/data/wqk/AI+Fin/data/processed/TSLA_scored_news_attention.csv"}
    ]

    model_path = "/mnt/data/wqk/AI+Fin/models/finbert"
    
    for company in companies:
        logger.info(f"开始处理 {company['symbol']} 的新闻数据")
        main(
            raw_data_path=company["raw_data_path"],
            output_path=company["output_path"],
            model_path=model_path,
            symbol=company["symbol"]
        )
