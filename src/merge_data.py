import pandas as pd
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 定义时间范围
start_date = "2021/06/01"
end_date = "2025/05/31"

# M7 公司数据
companies = [
    {"symbol": "GOOGL", "stock_path": "/mnt/data/wqk/AI+Fin/data/raw/105.GOOGL_谷歌-A.csv", "sentiment_path": "/mnt/data/wqk/AI+Fin/data/processed/GOOGL_scored_news_attention.csv", "stock_output": "/mnt/data/wqk/AI+Fin/data/stock/105.GOOGL_谷歌.csv", "merge_output": "/mnt/data/wqk/AI+Fin/data/merge/105.GOOGL_谷歌_with_sentiment.csv"},
    {"symbol": "AMZN", "stock_path": "/mnt/data/wqk/AI+Fin/data/raw/105.AMZN_亚马逊.csv", "sentiment_path": "/mnt/data/wqk/AI+Fin/data/processed/AMZN_scored_news_attention.csv", "stock_output": "/mnt/data/wqk/AI+Fin/data/stock/105.AMZN_亚马逊.csv", "merge_output": "/mnt/data/wqk/AI+Fin/data/merge/105.AMZN_亚马逊_with_sentiment.csv"},
    {"symbol": "AAPL", "stock_path": "/mnt/data/wqk/AI+Fin/data/raw/105.AAPL_苹果.csv", "sentiment_path": "/mnt/data/wqk/AI+Fin/data/processed/AAPL_scored_news_attention.csv", "stock_output": "/mnt/data/wqk/AI+Fin/data/stock/105.AAPL_苹果.csv", "merge_output": "/mnt/data/wqk/AI+Fin/data/merge/105.AAPL_苹果_with_sentiment.csv"},
    {"symbol": "META", "stock_path": "/mnt/data/wqk/AI+Fin/data/raw/105.META_Meta Platforms Inc-A.csv", "sentiment_path": "/mnt/data/wqk/AI+Fin/data/processed/META_scored_news_attention.csv", "stock_output": "/mnt/data/wqk/AI+Fin/data/stock/105.META_Meta Platforms Inc.csv", "merge_output": "/mnt/data/wqk/AI+Fin/data/merge/105.META_Meta Platforms Inc_with_sentiment.csv"},
    {"symbol": "MSFT", "stock_path": "/mnt/data/wqk/AI+Fin/data/raw/105.MSFT_微软.csv", "sentiment_path": "/mnt/data/wqk/AI+Fin/data/processed/MSFT_scored_news_attention.csv", "stock_output": "/mnt/data/wqk/AI+Fin/data/stock/105.MSFT_微软.csv", "merge_output": "/mnt/data/wqk/AI+Fin/data/merge/105.MSFT_微软_with_sentiment.csv"},
    {"symbol": "NVDA", "stock_path": "/mnt/data/wqk/AI+Fin/data/raw/105.NVDA_英伟达.csv", "sentiment_path": "/mnt/data/wqk/AI+Fin/data/processed/NVDA_scored_news_attention.csv", "stock_output": "/mnt/data/wqk/AI+Fin/data/stock/105.NVDA_英伟达.csv", "merge_output": "/mnt/data/wqk/AI+Fin/data/merge/105.NVDA_英伟达_with_sentiment.csv"},
    {"symbol": "TSLA", "stock_path": "/mnt/data/wqk/AI+Fin/data/raw/105.TSLA_特斯拉.csv", "sentiment_path": "/mnt/data/wqk/AI+Fin/data/processed/TSLA_scored_news_attention.csv", "stock_output": "/mnt/data/wqk/AI+Fin/data/stock/105.TSLA_特斯拉.csv", "merge_output": "/mnt/data/wqk/AI+Fin/data/merge/105.TSLA_特斯拉_with_sentiment.csv"}
]

# 确保输出目录存在
os.makedirs("/mnt/data/wqk/AI+Fin/data/stock", exist_ok=True)
os.makedirs("/mnt/data/wqk/AI+Fin/data/merge", exist_ok=True)

for company in companies:
    logger.info(f"开始处理 {company['symbol']} 数据")

    # 读取股票数据
    try:
        stock_df = pd.read_csv(company['stock_path'])
        stock_df['日期'] = pd.to_datetime(stock_df['日期']).dt.strftime('%Y/%m/%d')
        stock_df = stock_df[(stock_df['日期'] >= start_date) & (stock_df['日期'] <= end_date)]
        stock_df.to_csv(company['stock_output'], index=False, encoding='utf-8-sig')
        logger.info(f"{company['symbol']} 股票数据已截取至 {company['stock_output']}")
    except Exception as e:
        logger.error(f"{company['symbol']} 股票数据读取或保存失败: {e}")
        continue

    # 读取情感得分数据
    try:
        sentiment_df = pd.read_csv(company['sentiment_path'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.strftime('%Y/%m/%d')
        sentiment_df = sentiment_df[(sentiment_df['date'] >= start_date) & (sentiment_df['date'] <= end_date)]
        sentiment_df.set_index('date', inplace=True)
    except Exception as e:
        logger.error(f"{company['symbol']} 情感数据读取失败: {e}")
        continue

    # 合并数据
    try:
        merged_df = stock_df.copy()
        merged_df['日期'] = pd.to_datetime(merged_df['日期']).dt.strftime('%Y/%m/%d')
        merged_df.set_index('日期', inplace=True)
        merged_df['情感分数'] = merged_df.index.map(lambda x: sentiment_df.loc[x, 'sentiment_score'] if x in sentiment_df.index else 0)
        merged_df.reset_index(inplace=True)
        merged_df.to_csv(company['merge_output'], index=False, encoding='utf-8-sig')
        logger.info(f"{company['symbol']} 数据合并完成，已保存至 {company['merge_output']}")
    except Exception as e:
        logger.error(f"{company['symbol']} 数据合并失败: {e}")
        continue
